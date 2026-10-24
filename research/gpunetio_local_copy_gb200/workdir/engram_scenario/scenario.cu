// SPDX-License-Identifier: Apache-2.0
#include "local_copy.cuh"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void *lc_open(void *, size_t, void *, size_t, int, const char *, int);
extern "C" const char *lc_error();
extern "C" void *lc_device(void *);
extern "C" int lc_close(void *);
constexpr int layers=2, heads=24, hidden=256, guard=128, maxq=8;
constexpr size_t rowbytes=hidden*sizeof(__nv_bfloat16);
void ck(cudaError_t e,const char *op) { if(e) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e)); }
void bc(cublasStatus_t e,const char *op) { if(e) throw std::runtime_error(std::string(op)+": "+std::to_string(e)); }
#define C(x) ck((x),#x)
#define B(x) bc((x),#x)
__host__ __device__ uint32_t hash(uint32_t x) { x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16); }
__host__ __device__ int value(int row,int col,int layer,int epoch) {
    // Injective source-row identity plus changing bounded integer payload.
    if(col<7){uint32_t r=row;for(int i=0;i<col;++i)r/=31;return r%31;}
    return hash(uint32_t(row)*0x9e3779b9u^uint32_t(col)*0x85ebca6bu^
        uint32_t(layer)*0xc2b2ae35u^uint32_t(epoch)*0x27d4eb2fu)%31;
}

__global__ void init_weight(__nv_bfloat16 *w,size_t n) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        int v=int(hash(uint32_t(i))%17)-8;w[i]=__float2bfloat16(float(v?v:1));
    }
}
__global__ void expected_a(__nv_bfloat16 *a,const int *idx,int count,int layer,int epoch) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count*hidden;i+=gridDim.x*blockDim.x)
        a[i]=__float2bfloat16(float(value(idx[i/hidden],i%hidden,layer,epoch)));
}
__global__ void issue(local_copy::Copy *copy,const uint8_t *src,uint8_t *dst,const int *idx,
                       int count,int q,int ways,int *status) {
    int width=copy->mcst?16:32;
    for(int batch=q;batch<(count+width-1)/width;batch+=ways) {
        int j=batch*width+threadIdx.x;bool active=threadIdx.x<width&&j<count;
        auto t=copy->copy_async_warp(active?dst+guard+size_t(j)*rowbytes:dst,
            active?src+guard+size_t(idx[j])*rowbytes:src,rowbytes,active);
        int s=0;if(threadIdx.x==0)s=copy->wait(t);
        s=__shfl_sync(0xffffffffu,s,0);
        if(s) {if(threadIdx.x==0)atomicCAS(status,0,s);return;}
    }
}
__global__ void sm_copy(const uint8_t *src,uint8_t *dst,const int *idx,int count) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int j=blockIdx.x*4+warp;j<count;j+=gridDim.x*4)
        *reinterpret_cast<uint4 *>(dst+guard+size_t(j)*rowbytes+lane*16)=
            *reinterpret_cast<const uint4 *>(src+guard+size_t(idx[j])*rowbytes+lane*16);
}
__global__ void consume(const __nv_bfloat16 *a,float *out,int count) {
    int t=threadIdx.x;if(t<128)out[t]=__bfloat162float(a[(t*7919)%(count*hidden)]);
}
__global__ void validate(const uint8_t *dst,size_t bytes,const int *idx,const float *out,
                          const float *gold,int count,int n,int layer,int epoch,int gemms,
                          unsigned long long *errors) {
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<bytes;i+=gridDim.x*blockDim.x) {
        uint8_t expected=0xcd;
        if(i>=guard&&i<guard+size_t(count)*rowbytes) {
            size_t p=i-guard;int j=p/rowbytes,col=(p%rowbytes)/2;
            uint16_t bits=uint16_t(__float_as_uint(float(value(idx[j],col,layer,epoch)))>>16);
            expected=(p&1)?bits>>8:bits&255;
        }
        if(dst[i]!=expected)atomicAdd(errors,1ULL);
    }
    size_t elems=gemms?size_t(count/heads)*n:128;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<elems;i+=gridDim.x*blockDim.x) {
        float expected;
        if(gemms)expected=gold[i]*gemms;
        else {int p=(i*7919)%(count*hidden);expected=float(value(idx[p/hidden],p%hidden,layer,epoch));}
        if(out[i]!=expected)atomicAdd(errors,1ULL);
    }
}

struct Stage {cudaEvent_t start=nullptr,ready=nullptr,begin=nullptr,done=nullptr;};
struct Case {
    std::string arm,mode;int gemms,q;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;cudaEvent_t start=nullptr,end=nullptr;
    Stage stage[layers];
    ~Case() {
        if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);
        cudaEventDestroy(start);cudaEventDestroy(end);
        for(auto &s:stage){cudaEventDestroy(s.start);cudaEventDestroy(s.ready);cudaEventDestroy(s.begin);cudaEventDestroy(s.done);}
    }
};
cudaGraphNode_t event_node(cudaGraph_t graph,cudaEvent_t e,std::vector<cudaGraphNode_t> deps) {
    cudaGraphNode_t n;C(cudaGraphAddEventRecordNode(&n,graph,deps.data(),deps.size(),e));return n;
}
cudaGraphNode_t kernel_node(cudaGraph_t graph,void *fn,dim3 grid,dim3 block,void **args,
                            std::vector<cudaGraphNode_t> deps) {
    cudaKernelNodeParams p{};p.func=fn;p.gridDim=grid;p.blockDim=block;p.kernelParams=args;
    cudaGraphNode_t n;C(cudaGraphAddKernelNode(&n,graph,deps.data(),deps.size(),&p));return n;
}
void gemm(cublasHandle_t h,const __nv_bfloat16 *w,const __nv_bfloat16 *a,float *out,int m,int k,int n,float beta) {
    float alpha=1;
    B(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&alpha,w,CUDA_R_16BF,n,
        a,CUDA_R_16BF,k,&beta,out,CUDA_R_32F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
float ms(cudaEvent_t a,cudaEvent_t b) {float x;C(cudaEventElapsedTime(&x,a,b));return x;}

int main(int argc,char **argv) try {
    int gpu=argc>1?atoi(argv[1]):6,tokens=argc>2?atoi(argv[2]):512;
    int pool=argc>3?atoi(argv[3]):524288,n=argc>4?atoi(argv[4]):8192;
    int blocks=argc>5?atoi(argv[5]):7,repeats=argc>6?atoi(argv[6]):5;
    unsigned seed=argc>7?strtoul(argv[7],nullptr,10):20261024;
    std::string filter=argc>8?argv[8]:"all";int gfilter=argc>9?atoi(argv[9]):-1;
    std::string mfilter=argc>10?argv[10]:"all";
    if(tokens<1||pool<heads||n<1||blocks<1||repeats<1)throw std::runtime_error("invalid shape");
    C(cudaSetDevice(gpu));
    int count=tokens*heads,k=heads*hidden;size_t srcbytes=2*guard+size_t(pool)*rowbytes;
    size_t dstbytes=2*guard+size_t(count)*rowbytes,outelems=size_t(tokens)*n;
    if(srcbytes>UINT32_MAX||dstbytes>UINT32_MAX)throw std::runtime_error("registered region exceeds Copy limit");
    size_t outalloc=std::max(outelems,size_t(128));
    uint8_t *src[layers]{},*dst[layers]{},*srcalias[layers]{};
    int *idx[layers]{},*status;__nv_bfloat16 *golda[layers]{},*weight;float *out[layers]{},*gold[layers]{};
    unsigned long long *errors;C(cudaMalloc(&errors,sizeof(*errors)));C(cudaMalloc(&status,sizeof(*status)));
    C(cudaMalloc(&weight,size_t(k)*n*sizeof(*weight)));init_weight<<<256,256>>>(weight,size_t(k)*n);
    std::vector<int> maps[layers];std::mt19937 rng(seed);
    for(int l=0;l<layers;++l) {
        C(cudaMallocHost(&src[l],srcbytes));
        C(cudaHostGetDevicePointer(&srcalias[l],src[l],0));C(cudaMalloc(&dst[l],dstbytes));
        C(cudaMalloc(&idx[l],count*sizeof(int)));maps[l].resize(count);
        for(int j=0;j<count;++j)maps[l][j]=(j%heads)*(pool/heads)+int(rng()%(pool/heads));
        C(cudaMemcpy(idx[l],maps[l].data(),count*sizeof(int),cudaMemcpyHostToDevice));
        C(cudaMalloc(&golda[l],size_t(count)*rowbytes));
        C(cudaMalloc(&out[l],outalloc*sizeof(float)));C(cudaMalloc(&gold[l],outalloc*sizeof(float)));
        C(cudaMemset(dst[l],0,dstbytes));
        printf("{\"kind\":\"source\",\"layer\":%d,\"address\":\"%p\",\"bytes\":%zu}\n",l,src[l],srcbytes);
    }
    {
        cudaStream_t probe;C(cudaStreamCreateWithFlags(&probe,cudaStreamNonBlocking));
        void *from=src[0]+guard,*to=dst[0]+guard;size_t bytes=rowbytes,at=0,failed=SIZE_MAX;
        cudaMemcpyAttributes a{};a.srcAccessOrder=cudaMemcpySrcAccessOrderStream;a.flags=cudaMemcpyFlagPreferOverlapWithCompute;
        auto result=cudaMemcpyBatchAsync(&to,&from,&bytes,1,&a,&at,1,&failed,probe);
        printf("{\"kind\":\"batch_support\",\"status\":%d,\"message\":\"%s\"}\n",int(result),cudaGetErrorString(result));
        if(result==cudaSuccess)throw std::runtime_error("new batch API available: add matched baseline before measurement");
        cudaGetLastError();C(cudaStreamDestroy(probe));
    }
    {std::ifstream in("/proc/self/numa_maps");std::ofstream out("numa-maps-initial.txt");out<<in.rdbuf();}
    std::vector<void *> handles[layers];std::vector<local_copy::Copy *> copies[layers];
    auto cold=std::chrono::steady_clock::now();
    for(int l=0;l<layers;++l)for(int q=0;q<maxq;++q) {
        auto *h=lc_open(src[l],srcbytes,dst[l],dstbytes,gpu,"mlx5_bond_3",3);
        if(!h)throw std::runtime_error(lc_error());
        handles[l].push_back(h);copies[l].push_back(static_cast<local_copy::Copy *>(lc_device(h)));
    }
    printf("{\"kind\":\"config\",\"gpu\":%d,\"nic\":\"mlx5_bond_3\",\"tokens\":%d,\"rows\":%d,\"rowbytes\":%zu,\"layers\":2,\"pool_rows\":%d,\"projection_n\":%d,\"seed\":%u,\"registration_seconds\":%.4f}\n",
        gpu,tokens,count,rowbytes,pool,n,seed,std::chrono::duration<double>(std::chrono::steady_clock::now()-cold).count());
    cublasHandle_t blas;B(cublasCreate(&blas));B(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED));
    cudaStream_t capture,run;C(cudaStreamCreateWithFlags(&capture,cudaStreamNonBlocking));C(cudaStreamCreateWithFlags(&run,cudaStreamNonBlocking));
    B(cublasSetStream(blas,capture));
    void *workspace;constexpr size_t workspace_bytes=32*1024*1024;
    C(cudaMalloc(&workspace,workspace_bytes));B(cublasSetWorkspace(blas,workspace,workspace_bytes));
    gemm(blas,weight,reinterpret_cast<__nv_bfloat16 *>(dst[0]+guard),out[0],tokens,k,n,0);
    C(cudaStreamSynchronize(capture));
    cudaGraph_t compute[layers][3];int amounts[]={0,1,4};
    for(int l=0;l<layers;++l)for(int gi=0;gi<3;++gi) {
        C(cudaStreamBeginCapture(capture,cudaStreamCaptureModeThreadLocal));
        if(amounts[gi])for(int r=0;r<amounts[gi];++r)
            gemm(blas,weight,reinterpret_cast<__nv_bfloat16 *>(dst[l]+guard),out[l],tokens,k,n,1);
        else consume<<<1,128,0,capture>>>(reinterpret_cast<__nv_bfloat16 *>(dst[l]+guard),out[l],count);
        C(cudaStreamEndCapture(capture,&compute[l][gi]));
        size_t node_count=0;C(cudaGraphGetNodes(compute[l][gi],nullptr,&node_count));
        std::vector<cudaGraphNode_t> ns(node_count);C(cudaGraphGetNodes(compute[l][gi],ns.data(),&node_count));
        printf("{\"kind\":\"compute_graph\",\"layer\":%d,\"gemms\":%d,\"node_types\":[",l,amounts[gi]);
        for(size_t i=0;i<node_count;++i){cudaGraphNodeType t;C(cudaGraphNodeGetType(ns[i],&t));printf("%s%d",i?",":"",int(t));}puts("]}");
    }
    int smcount;C(cudaDeviceGetAttribute(&smcount,cudaDevAttrMultiProcessorCount,gpu));
    std::vector<std::string> arms={"CE_SERIAL","CE_PARALLEL","CE_2D","CE_2D_PARALLEL","SM_1","SM_8","SM_SATURATED","NIC_1","NIC_4","NIC_8"};
    std::vector<std::unique_ptr<Case>> cases;
    for(const auto &arm:arms)for(int gi=0;gi<3;++gi)for(const std::string mode:{"SERIAL","LOOKAHEAD"}) {
        if((filter!="all"&&filter!=arm)||(gfilter>=0&&gfilter!=amounts[gi])||(mfilter!="all"&&mfilter!=mode))continue;
        auto c=std::make_unique<Case>();c->arm=arm;c->mode=mode;c->gemms=amounts[gi];c->q=arm=="NIC_8"?8:arm=="NIC_4"?4:1;
        C(cudaGraphCreate(&c->graph,0));C(cudaEventCreate(&c->start));C(cudaEventCreate(&c->end));
        for(auto &s:c->stage){C(cudaEventCreate(&s.start));C(cudaEventCreate(&s.ready));C(cudaEventCreate(&s.begin));C(cudaEventCreate(&s.done));}
        auto root=event_node(c->graph,c->start,{});cudaGraphNode_t previous_ready=root,previous_done=root;
        for(int l=0;l<layers;++l) {
            auto start=event_node(c->graph,c->stage[l].start,{mode=="SERIAL"?previous_done:previous_ready});
            std::vector<cudaGraphNode_t> nodes;
            if(arm.starts_with("NIC"))for(int q=0;q<c->q;++q) {
                auto *copy=copies[l][q];int ways=c->q;
                void *args[]={&copy,&src[l],&dst[l],&idx[l],&count,&q,&ways,&status};
                nodes.push_back(kernel_node(c->graph,reinterpret_cast<void *>(issue),1,32,args,{start}));
            } else if(arm.starts_with("SM")) {
                int grid=arm=="SM_1"?1:arm=="SM_8"?8:smcount;
                void *args[]={&srcalias[l],&dst[l],&idx[l],&count};
                nodes.push_back(kernel_node(c->graph,reinterpret_cast<void *>(sm_copy),grid,128,args,{start}));
            } else for(int j=0;j<count;) {
                auto deps=std::vector<cudaGraphNode_t>{start};
                if((arm=="CE_SERIAL"||arm=="CE_2D")&&!nodes.empty())deps={nodes.back()};
                cudaGraphNode_t node;
                auto *from=src[l]+guard+size_t(maps[l][j])*rowbytes,*to=dst[l]+guard+size_t(j)*rowbytes;
                if(arm=="CE_2D"||arm=="CE_2D_PARALLEL") {
                    int d=j+1<count?maps[l][j+1]-maps[l][j]:0,height=1;
                    if(d>0)while(j+height<count&&maps[l][j+height]==maps[l][j]+height*d)++height;
                    cudaMemcpy3DParms p{};p.srcPtr=make_cudaPitchedPtr(from,(height>1?d:1)*rowbytes,rowbytes,height);
                    p.dstPtr=make_cudaPitchedPtr(to,rowbytes,rowbytes,height);p.extent=make_cudaExtent(rowbytes,height,1);p.kind=cudaMemcpyHostToDevice;
                    C(cudaGraphAddMemcpyNode(&node,c->graph,deps.data(),deps.size(),&p));j+=height;
                } else {C(cudaGraphAddMemcpyNode1D(&node,c->graph,deps.data(),deps.size(),to,from,rowbytes,cudaMemcpyHostToDevice));++j;}
                nodes.push_back(node);
            }
            previous_ready=event_node(c->graph,c->stage[l].ready,nodes);
            auto begin=event_node(c->graph,c->stage[l].begin,{previous_ready,previous_done});
            cudaGraphNode_t child;auto child_status=cudaGraphAddChildGraphNode(&child,c->graph,&begin,1,compute[l][gi]);
            if(child_status)throw std::runtime_error("child graph arm="+arm+" G="+std::to_string(c->gemms)+" layer="+std::to_string(l)+": "+cudaGetErrorString(child_status));
            previous_done=event_node(c->graph,c->stage[l].done,{child});
        }
        event_node(c->graph,c->end,{previous_done});
        C(cudaGraphInstantiate(&c->exec,c->graph,nullptr,nullptr,0));cases.push_back(std::move(c));
    }
    printf("{\"kind\":\"graphs_ready\",\"cases\":%zu}\n",cases.size());fflush(stdout);
    if(cases.empty())throw std::runtime_error("no selected case");
    size_t free_before,total;C(cudaMemGetInfo(&free_before,&total));
    for(int block=0;block<blocks;++block) {
        int epoch=block+1;
        for(int l=0;l<layers;++l) {
            std::fill(src[l],src[l]+srcbytes,0xa5);
            auto *p=reinterpret_cast<uint16_t *>(src[l]+guard);
            for(int r=0;r<pool;++r)for(int col=0;col<hidden;++col)
                p[size_t(r)*hidden+col]=uint16_t(std::bit_cast<uint32_t>(float(value(r,col,l,epoch)))>>16);
            expected_a<<<256,256,0,capture>>>(golda[l],idx[l],count,l,epoch);
            gemm(blas,weight,golda[l],gold[l],tokens,k,n,0);
        }
        C(cudaStreamSynchronize(capture));
        if(block==0){std::ifstream in("/proc/self/numa_maps");std::ofstream f("numa-maps.txt");f<<in.rdbuf();}
        if(block==0) {
            std::vector<int> shifted=maps[0];int delta=pool>31?31:1;
            for(auto &v:shifted)v=(v+delta)%pool;
            int *wrong;C(cudaMalloc(&wrong,count*sizeof(int)));
            C(cudaMemcpy(wrong,shifted.data(),count*sizeof(int),cudaMemcpyHostToDevice));
            C(cudaMemset(errors,0,sizeof(*errors)));C(cudaMemset(dst[0],0xcd,dstbytes));
            sm_copy<<<8,128,0,run>>>(srcalias[0],dst[0],wrong,count);
            consume<<<1,128,0,run>>>(reinterpret_cast<__nv_bfloat16 *>(dst[0]+guard),out[0],count);
            validate<<<256,256,0,run>>>(dst[0],dstbytes,idx[0],out[0],gold[0],count,n,0,epoch,0,errors);
            C(cudaStreamSynchronize(run));unsigned long long detected;
            C(cudaMemcpy(&detected,errors,sizeof(detected),cudaMemcpyDeviceToHost));C(cudaFree(wrong));
            if(!detected)throw std::runtime_error("FAIL validator accepted shifted source rows");
            printf("{\"kind\":\"negative_validation\",\"source_row_shift\":%d,\"detected_mismatches\":%llu}\n",delta,detected);
        }
        for(size_t order=0;order<cases.size();++order) {
            auto &c=*cases[block%2?cases.size()-1-order:order];
            std::array<double,9> sum{};double wall=0;
            for(int trial=-2;trial<repeats;++trial) {
                size_t free_now;C(cudaMemGetInfo(&free_now,&total));
                if(trial>=0&&free_now+32*1024*1024<free_before)throw std::runtime_error("INVALID GPU occupancy changed");
                C(cudaMemsetAsync(status,0,sizeof(*status),run));C(cudaMemsetAsync(errors,0,sizeof(*errors),run));
                for(int l=0;l<layers;++l){C(cudaMemsetAsync(dst[l],0xcd,dstbytes,run));C(cudaMemsetAsync(out[l],0,outalloc*sizeof(float),run));}
                C(cudaStreamSynchronize(run));auto begin=std::chrono::steady_clock::now();
                C(cudaGraphLaunch(c.exec,run));C(cudaStreamSynchronize(run));
                wall+=trial>=0?std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count():0;
                std::array<double,9> t={ms(c.start,c.end),ms(c.stage[0].start,c.stage[0].ready),ms(c.stage[1].start,c.stage[1].ready),
                    ms(c.stage[0].begin,c.stage[0].done),ms(c.stage[1].begin,c.stage[1].done),
                    ms(c.start,c.stage[0].begin),ms(c.stage[0].done,c.stage[1].begin),
                    ms(c.start,c.stage[1].start),ms(c.start,c.stage[1].ready)};
                for(int l=0;l<layers;++l)validate<<<256,256,0,run>>>(dst[l],dstbytes,idx[l],out[l],gold[l],count,n,l,epoch,c.gemms,errors);
                C(cudaStreamSynchronize(run));unsigned long long mismatches;int st;
                C(cudaMemcpy(&mismatches,errors,sizeof(mismatches),cudaMemcpyDeviceToHost));C(cudaMemcpy(&st,status,sizeof(st),cudaMemcpyDeviceToHost));
                if(mismatches||st)throw std::runtime_error("FAIL data/projection status="+std::to_string(st)+" errors="+std::to_string(mismatches));
                for(int l=0;l<layers;++l)for(int j=0;j<guard;++j)
                    if(src[l][j]!=0xa5||src[l][srcbytes-1-j]!=0xa5)throw std::runtime_error("FAIL Host guard");
                // Capture this case's baseline after its own graph/module uploads.
                if(trial==-1)C(cudaMemGetInfo(&free_before,&total));
                if(trial>=0)for(int i=0;i<9;++i)sum[i]+=t[i];
            }
            for(auto &v:sum)v/=repeats;
            printf("{\"kind\":\"result\",\"block\":%d,\"arm\":\"%s\",\"mode\":\"%s\",\"gemms\":%d,\"qps_per_layer\":%d,\"e2e_ms\":%.6f,\"copy0_ms\":%.6f,\"copy1_ms\":%.6f,\"compute0_ms\":%.6f,\"compute1_ms\":%.6f,\"wait0_ms\":%.6f,\"wait1_ms\":%.6f,\"copy1_start_ms\":%.6f,\"copy1_ready_ms\":%.6f,\"wall_ms\":%.6f,\"mismatch\":0}\n",
                block,c.arm.c_str(),c.mode.c_str(),c.gemms,c.q,sum[0],sum[1],sum[2],sum[3],sum[4],sum[5],sum[6],sum[7],sum[8],wall/repeats);fflush(stdout);
        }
    }
    cases.clear();for(int l=0;l<layers;++l)for(int gi=0;gi<3;++gi)C(cudaGraphDestroy(compute[l][gi]));
    B(cublasDestroy(blas));C(cudaFree(workspace));C(cudaStreamDestroy(capture));C(cudaStreamDestroy(run));
    for(int l=0;l<layers;++l)for(auto it=handles[l].rbegin();it!=handles[l].rend();++it)
        if(lc_close(*it))throw std::runtime_error(lc_error());
    for(int l=0;l<layers;++l){C(cudaFree(idx[l]));C(cudaFree(golda[l]));C(cudaFree(out[l]));C(cudaFree(gold[l]));C(cudaFree(dst[l]));C(cudaFreeHost(src[l]));}
    C(cudaFree(weight));C(cudaFree(errors));C(cudaFree(status));puts("{\"kind\":\"complete\",\"status\":\"PASS\"}");
    return 0;
} catch(const std::exception &e) {fprintf(stderr,"%s\n",e.what());return 1;}
