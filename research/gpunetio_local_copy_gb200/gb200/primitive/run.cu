// SPDX-License-Identifier: Apache-2.0
// Host harness uses the existing VMM allocator; GPUNetIO3.5 device code remains
// in its separate DSO, not mixed with the helper header's NCCL4.1 device code.
#include "common.cuh"
#include "abi.cuh"
__global__ void validate(const uint32_t* out,const uint32_t* independent,const uint8_t* outputBase,const uint8_t* dst,
                          size_t bytes,int fragments,int rounds,unsigned seed,unsigned long long* errors){
    size_t words=bytes*fragments/4,total=words*rounds,pitch=bytes+64;unsigned long long bad=0;
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=gridDim.x*blockDim.x){
        int round=i/words;size_t w=i%words,f=w*4/bytes,col=w*4%bytes;
        uint32_t expected=sourceWord((sourceRow(round,f,seed)*pitch+col)/4,seed)^independent[round*128+w%128];bad+=out[i]!=expected;
    }
    for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<guard;i+=gridDim.x*blockDim.x)bad+=(outputBase[i]!=0xa5)+(outputBase[guard+total*4+i]!=0xa5)+(dst[i]!=0xa5)+(dst[guard+bytes*fragments+i]!=0xa5);
    if(bad)atomicAdd(errors,bad);
}
struct Case{int kind,serial,fragments,work;size_t bytes;};
int main(int argc,char** argv)try{
    std::string phase=argc>1?argv[1]:"pilot";unsigned seed=argc>2?strtoul(argv[2],nullptr,10):20261029;
    int blocks=argc>3?atoi(argv[3]):3,reps=argc>4?atoi(argv[4]):3,rounds=argc>5?atoi(argv[5]):128;bool profile=phase=="profile";
    if(blocks<1||reps<1||rounds<1||rounds>4096)throw std::runtime_error("invalid config");
    C(cudaSetDevice(0));C(cudaFree(nullptr));int numa;D(cuDeviceGetAttribute(&numa,CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,0));
    const size_t sourceBytes=8*1024*1024,dstBytes=32*4096,outBytes=dstBytes*rounds;
    HostVmm backing;uint8_t* src=backing.allocate(sourceBytes+2*guard,0,numa),*dst,*outputBase;
    N(ncclMemAlloc(reinterpret_cast<void**>(&dst),dstBytes+2*guard));C(cudaMalloc(&outputBase,outBytes+2*guard));auto* out=reinterpret_cast<uint32_t*>(outputBase+guard);
    void* module=dlopen(getenv("DIRECT_COPY_LIBRARY"),RTLD_NOW|RTLD_LOCAL);if(!module)throw std::runtime_error(dlerror());
    auto open=reinterpret_cast<void*(*)(void*,size_t,void*,size_t,int,const char*,int)>(dlsym(module,"lc_open"));auto device=reinterpret_cast<void*(*)(void*)>(dlsym(module,"lc_device"));auto close=reinterpret_cast<int(*)(void*)>(dlsym(module,"lc_close"));auto err=reinterpret_cast<const char*(*)()>(dlsym(module,"lc_error"));
    auto kernel=reinterpret_cast<void*(*)(int,int,int)>(dlsym(module,"lc_primitive_kernel"));auto facade=reinterpret_cast<int(*)(void*,const void*,void*,int,int,void*)>(dlsym(module,"lc_facade_test"));auto warpTest=reinterpret_cast<int(*)(void*,const void*,void*,int,void*)>(dlsym(module,"lc_warp_contract"));
    if(!open||!device||!close||!err||!kernel||!facade||!warpTest)throw std::runtime_error("missing exports");
    void* handle=open(src,sourceBytes+2*guard,dst,dstBytes+2*guard,0,"mlx5_0",3);if(!handle)throw std::runtime_error(err());void* copy=device(handle);
    PrimitiveStats* stats;unsigned long long* errors;uint32_t* independent;C(cudaMalloc(&stats,sizeof(PrimitiveStats)));C(cudaMalloc(&errors,8));C(cudaMalloc(&independent,size_t(rounds)*128*4));
    std::fill(src,src+sourceBytes+2*guard,0xa5);C(cudaMemset(dst,0xa5,dstBytes+2*guard));C(cudaMemset(errors,0,8));
    for(size_t i=0;i<sourceBytes/4;++i)reinterpret_cast<uint32_t*>(src+guard)[i]=sourceWord(i,seed);
    auto example=reinterpret_cast<int(*)(void*,const void*,void*,void*,void*)>(dlsym(module,"lc_thin_example"));if(!example)throw std::runtime_error("missing thin example");
    C(static_cast<cudaError_t>(example(copy,src+guard,dst+guard,out,errors)));C(cudaDeviceSynchronize());
    unsigned long long exampleError;C(cudaMemcpy(&exampleError,errors,8,cudaMemcpyDeviceToHost));if(exampleError)throw std::runtime_error("thin example error");
    uint32_t exampleOut[128];C(cudaMemcpy(exampleOut,out,sizeof(exampleOut),cudaMemcpyDeviceToHost));
    for(int t=0;t<128;++t)if(exampleOut[t]!=(sourceWord(t,seed)^independentWork(42,0,t,128)))throw std::runtime_error("thin example consumer mismatch");
    puts("{\"kind\":\"thin_example\",\"status\":\"PASS\",\"elements\":128}");
    auto warpExample=reinterpret_cast<int(*)(void*,const void*,void*,void*,void*)>(dlsym(module,"lc_thin_warp_example"));if(!warpExample)throw std::runtime_error("missing warp example");
    C(static_cast<cudaError_t>(warpExample(copy,src+guard,dst+guard,out,errors)));C(cudaDeviceSynchronize());C(cudaMemcpy(&exampleError,errors,8,cudaMemcpyDeviceToHost));if(exampleError)throw std::runtime_error("warp example error");
    uint32_t warpOut[512];C(cudaMemcpy(warpOut,out,sizeof(warpOut),cudaMemcpyDeviceToHost));for(int i=0;i<512;++i)if(warpOut[i]!=(sourceWord(i,seed)^independentWork(43,0,i/16,128)))throw std::runtime_error("warp example consumer mismatch");puts("{\"kind\":\"warp_example\",\"status\":\"PASS\",\"elements\":512}");
    for(int kind=0;kind<3;++kind)for(int mode=0;mode<2;++mode){C(static_cast<cudaError_t>(facade(copy,src,dst,kind,mode,errors)));C(cudaDeviceSynchronize());}
    for(int mode:{0,2}){C(static_cast<cudaError_t>(warpTest(copy,src,dst,mode,errors)));C(cudaDeviceSynchronize());}
    unsigned long long bad;C(cudaMemcpy(&bad,errors,8,cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("facade/warp contract failed "+std::to_string(bad));
    puts("{\"kind\":\"contracts\",\"status\":\"PASS\",\"refusal_drain_reuse\":true,\"warp_partial_mismatch_stale\":true,\"forced_mcst\":true}");
    std::vector<Case> cases;
    std::vector<size_t> sizes=phase=="pilot"?std::vector<size_t>{64,512}:std::vector<size_t>{64,512,4096};
    for(size_t bytes:sizes)for(int fragments:{1,8,31,32})for(int work:{0,128,1024})for(int kind=0;kind<7;++kind)for(int serial=0;serial<2;++serial){
        if(kind==5&&serial)continue;if(kind==6&&bytes*fragments>16384)continue;
        cases.push_back({kind,serial,fragments,work,bytes});
    }
    if(phase=="pilot")rounds=std::min(rounds,8);
    printf("{\"kind\":\"config\",\"phase\":\"%s\",\"seed\":%u,\"blocks\":%d,\"reps\":%d,\"rounds\":%d,\"threads\":128,\"cases\":%zu,\"numa\":%d,\"profile\":%s}\n",phase.c_str(),seed,blocks,reps,rounds,cases.size(),numa,profile?"true":"false");fflush(stdout);
    for(int kind=0;kind<7;++kind){cudaFuncAttributes a;C(cudaFuncGetAttributes(&a,kernel(kind,0,profile)));printf("{\"kind\":\"resource\",\"path\":%d,\"registers\":%d,\"shared\":%zu,\"local\":%zu}\n",kind,a.numRegs,a.sharedSizeBytes,a.localSizeBytes);}
    cudaStream_t stream;C(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));cudaEvent_t begin,end;C(cudaEventCreate(&begin));C(cudaEventCreate(&end));
    bool negative=false;
    for(int block=0;block<blocks;++block){
        unsigned epoch=seed+block;auto* words=reinterpret_cast<uint32_t*>(src+guard);for(size_t i=0;i<sourceBytes/4;++i)words[i]=sourceWord(i,epoch);
        std::vector<uint32_t> refs[3];int workLevels[]={0,128,1024};
        for(int wi=0;wi<3;++wi){refs[wi].resize(size_t(rounds)*128);for(int r=0;r<rounds;++r)for(int t=0;t<128;++t)refs[wi][r*128+t]=independentWork(epoch,r,t,workLevels[wi]);}
        auto order=cases;std::mt19937 rng(epoch);std::shuffle(order.begin(),order.end(),rng);
        for(auto c:order){
            int wi=c.work==0?0:(c.work==128?1:2);C(cudaMemcpy(independent,refs[wi].data(),refs[wi].size()*4,cudaMemcpyHostToDevice));
            void* args[]={&copy,&src,&dst,&out,&c.bytes,&c.fragments,&c.work,&rounds,&epoch,&stats};cudaGraph_t g;C(cudaGraphCreate(&g,0));kernelNode(g,kernel(c.kind,c.serial,profile),1,128,args);cudaGraphExec_t exec;C(cudaGraphInstantiate(&exec,g,nullptr,nullptr,0));
            for(int trial=-1;trial<reps;++trial){
                C(cudaMemsetAsync(dst,0xa5,dstBytes+2*guard,stream));C(cudaMemsetAsync(outputBase,0xa5,outBytes+2*guard,stream));C(cudaMemsetAsync(stats,0,sizeof(PrimitiveStats),stream));C(cudaMemsetAsync(errors,0,8,stream));C(cudaStreamSynchronize(stream));
                auto wallStart=std::chrono::steady_clock::now();C(cudaEventRecord(begin,stream));C(cudaGraphLaunch(exec,stream));C(cudaEventRecord(end,stream));C(cudaStreamSynchronize(stream));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wallStart).count();
                PrimitiveStats state;C(cudaMemcpy(&state,stats,sizeof(state),cudaMemcpyDeviceToHost));if(state.status)throw std::runtime_error("primitive error "+std::to_string(state.status)+" path="+std::to_string(c.kind)+" serial="+std::to_string(c.serial)+" bytes="+std::to_string(c.bytes)+" fragments="+std::to_string(c.fragments)+" work="+std::to_string(c.work)+" trial="+std::to_string(trial));
                if(c.kind<5&&(state.mcst||state.issued!=state.retired||state.cqci!=state.retired||state.owner))throw std::runtime_error("incomplete retirement");
                validate<<<256,256,0,stream>>>(out,independent,outputBase,dst,c.bytes,c.fragments,rounds,epoch,errors);C(cudaStreamSynchronize(stream));C(cudaMemcpy(&bad,errors,8,cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("consumer/guard mismatch "+std::to_string(bad));
                if(!negative){uint32_t changed;C(cudaMemcpy(&changed,out,4,cudaMemcpyDeviceToHost));changed^=1;C(cudaMemcpy(out,&changed,4,cudaMemcpyHostToDevice));validate<<<256,256,0,stream>>>(out,independent,outputBase,dst,c.bytes,c.fragments,rounds,epoch,errors);C(cudaStreamSynchronize(stream));C(cudaMemcpy(&bad,errors,8,cudaMemcpyDeviceToHost));if(!bad)throw std::runtime_error("negative failed");negative=true;puts("{\"kind\":\"negative\",\"status\":\"PASS\"}");}
                for(size_t i=0;i<guard;++i)if(src[i]!=0xa5||src[guard+sourceBytes+i]!=0xa5)throw std::runtime_error("source guards");
                if(trial>=0)printf("{\"kind\":\"result\",\"block\":%d,\"trial\":%d,\"path\":%d,\"serialized\":%d,\"bytes\":%zu,\"fragments\":%d,\"work\":%d,\"e2e_ms\":%.6f,\"wall_ms\":%.6f,\"submit_cycles\":%llu,\"compute_cycles\":%llu,\"wait_cycles\":%llu,\"consume_cycles\":%llu,\"issued\":%llu,\"retired\":%llu,\"mismatch\":0}\n",block,trial,c.kind,c.serial,c.bytes,c.fragments,c.work,elapsed(begin,end),wall,state.submit,state.compute,state.wait,state.consume,state.issued,state.retired);
            }
            fflush(stdout);C(cudaGraphExecDestroy(exec));C(cudaGraphDestroy(g));
        }
    }
    // Real intermediate-CQE failure uses a disposable context after timings.
    C(cudaMemset(errors,0,8));C(static_cast<cudaError_t>(warpTest(copy,src,dst,3,errors)));C(cudaDeviceSynchronize());C(cudaMemcpy(&bad,errors,8,cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("intermediate CQ error escaped");puts("{\"kind\":\"intermediate_cqe_error\",\"status\":\"PASS\"}");
    if(close(handle))throw std::runtime_error(err());
    if(getenv("PRIMITIVE_TEST_TIMEOUT")){
        handle=open(src,sourceBytes+2*guard,dst,dstBytes+2*guard,0,"mlx5_0",3);if(!handle)throw std::runtime_error(err());copy=device(handle);
        C(cudaMemset(errors,0,8));C(static_cast<cudaError_t>(warpTest(copy,src,dst,4,errors)));C(cudaDeviceSynchronize());C(cudaMemcpy(&bad,errors,8,cudaMemcpyDeviceToHost));if(bad)throw std::runtime_error("timeout contract failed");if(close(handle))throw std::runtime_error(err());puts("{\"kind\":\"injected_timeout\",\"status\":\"PASS\"}");
    }
    C(cudaEventDestroy(begin));C(cudaEventDestroy(end));C(cudaStreamDestroy(stream));for(void* p:{static_cast<void*>(outputBase),static_cast<void*>(stats),static_cast<void*>(errors),static_cast<void*>(independent)})C(cudaFree(p));N(ncclMemFree(dst));backing.close();dlclose(module);puts("{\"kind\":\"complete\",\"status\":\"PASS\"}");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
