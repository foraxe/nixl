// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Two-rank GDAKI-only versus NIXL device GET, remote Host-NUMA -> VRAM.
#include <nccl.h>
#include <nccl_device.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <vector>
#include <time.h>

#if !NCCL_GIN_GDAKI_ENABLE || NCCL_GIN_PROXY_ENABLE || NCCL_GIN_GPI_ENABLE || NCCL_GIN_EFA_GDA_ENABLE
#error Build with only NCCL_GIN_GDAKI_ENABLE=1; disable the other GIN backends.
#endif
#ifndef BENCH_GIN_SHARING
#define BENCH_GIN_SHARING NCCL_GIN_RESOURCE_SHARING_GPU
#endif

#define CUCHK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(r,&s);fprintf(stderr,"%s: %d %s\n",#x,int(r),s?s:"");exit(1);} }while(0)
#define RTCHK(x) do { cudaError_t r=(x); if(r!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(r));exit(1);} }while(0)
#define NCCHECK(x) do { ncclResult_t r=(x); if(r!=ncclSuccess){fprintf(stderr,"%s: %s\n",#x,ncclGetErrorString(r));exit(1);} }while(0)
static void transfer(int fd,void*buf,size_t n,bool writeit){char*p=static_cast<char*>(buf);while(n){ssize_t k=writeit?send(fd,p,n,0):recv(fd,p,n,0);if(k<=0){perror(writeit?"send":"recv");exit(1);}p+=k;n-=size_t(k);}}
static int endpoint(int rank,const char*address,int port,ncclUniqueId*id){int fd=socket(AF_INET,SOCK_STREAM,0);if(fd<0){perror("socket");exit(1);}sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons(port);if(rank==0){int yes=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));addr.sin_addr.s_addr=INADDR_ANY;if(bind(fd,(sockaddr*)&addr,sizeof(addr))||listen(fd,1)){perror("bind/listen");exit(1);}printf("BOOTSTRAP_READY\n");fflush(stdout);int peer=accept(fd,nullptr,nullptr);if(peer<0){perror("accept");exit(1);}close(fd);NCCHECK(ncclGetUniqueId(id));transfer(peer,id,sizeof(*id),true);return peer;}if(inet_pton(AF_INET,address,&addr.sin_addr)!=1){fprintf(stderr,"invalid IPv4\n");exit(1);}if(connect(fd,(sockaddr*)&addr,sizeof(addr))){perror("connect");exit(1);}transfer(fd,id,sizeof(*id),false);return fd;}
struct HostMem { CUdeviceptr ptr{}; CUmemGenericAllocationHandle handle{}; size_t size{}; };
static HostMem alloc_host(size_t requested){CUdevice dev;CUCHK(cuDeviceGet(&dev,0));int numa=-1;CUCHK(cuDeviceGetAttribute(&numa,CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,dev));if(numa<0)numa=0;CUmemAllocationProp p{};p.type=CU_MEM_ALLOCATION_TYPE_PINNED;p.location.type=CU_MEM_LOCATION_TYPE_HOST_NUMA;p.location.id=numa;p.requestedHandleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;size_t gran=0;CUCHK(cuMemGetAllocationGranularity(&gran,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM));HostMem m{};m.size=((requested+gran-1)/gran)*gran;CUCHK(cuMemCreate(&m.handle,m.size,&p,0));CUCHK(cuMemAddressReserve(&m.ptr,m.size,0,0,0));CUCHK(cuMemMap(m.ptr,m.size,0,m.handle,0));CUmemAccessDesc d[2]{};d[0].location.type=CU_MEM_LOCATION_TYPE_DEVICE;d[0].location.id=dev;d[0].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;d[1].location.type=CU_MEM_LOCATION_TYPE_HOST_NUMA;d[1].location.id=numa;d[1].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;CUCHK(cuMemSetAccess(m.ptr,m.size,d,2));printf("HOST_NUMA_READY node=%d bytes=%zu\n",numa,m.size);return m;}

struct Nixl {
 void*lib{}; void*ctx{};
 void*(*create)(uint32_t,uint16_t,const char*){};
 int(*reg)(void*,int,const uint64_t*,const uint64_t*,uint32_t){};
 int64_t(*metadata)(void*,char*,size_t){};
 int(*connect)(void*,const char*,size_t,const uint64_t*,const uint64_t*,uint32_t,uint32_t){};
 int(*fetch)(void*,uint64_t,uint32_t,size_t,size_t,size_t,uint32_t,uint32_t,uint32_t,uint64_t,uint64_t,uint64_t,uint64_t){};
 int(*consumed)(void*,uint64_t){}; int(*destroy)(void*){}; const char*(*error)(){};
 template<typename T> void sym(T&out,const char*name){out=reinterpret_cast<T>(dlsym(lib,name));if(!out){fprintf(stderr,"missing %s\n",name);exit(1);}}
 void check(int rc,const char*where){if(rc<0){fprintf(stderr,"NIXL %s: %s\n",where,error());exit(1);}}
 Nixl(const char*path,uint16_t port,const char*name){lib=dlopen(path,RTLD_NOW|RTLD_GLOBAL);if(!lib){fprintf(stderr,"dlopen: %s\n",dlerror());exit(1);}sym(create,"engram_create");sym(reg,"engram_register");sym(metadata,"engram_metadata");sym(connect,"engram_connect");sym(fetch,"engram_fetch_row264");sym(consumed,"engram_record_consumed");sym(destroy,"engram_destroy");sym(error,"engram_last_error");ctx=create(0,port,name);if(!ctx){fprintf(stderr,"NIXL create: %s\n",error());exit(1);}}
 void regist(bool host,void*p,uint64_t n){uint64_t a=(uint64_t)p;check(reg(ctx,host,&a,&n,1),"register");}
 std::vector<char> md(){int64_t n=metadata(ctx,nullptr,0);if(n<0){check(-1,"metadata length");}std::vector<char> v(n);check(int(metadata(ctx,v.data(),v.size())),"metadata");return v;}
 void attach(const std::vector<char>&v,uint64_t addr,uint64_t length){check(connect(ctx,v.data(),v.size(),&addr,&length,1,0),"connect");}
 void read(int64_t*ids,uint32_t n,uint64_t*stats,cudaStream_t s){check(fetch(ctx,(uint64_t)ids,n,64,0,64+4096*264+64,0,0,512,(uint64_t)stats,(uint64_t)s,0,8192),"fetch");}
 void retire(cudaStream_t s){check(consumed(ctx,(uint64_t)s),"consumer");}
 void close(){if(ctx){check(destroy(ctx),"destroy");ctx=nullptr;}if(lib){dlclose(lib);lib=nullptr;}}
};
static uint64_t ns(clockid_t clock){timespec t{};if(clock_gettime(clock,&t)){perror("clock_gettime");exit(1);}return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;}
static uint8_t value(unsigned row,unsigned col){if(col==0)return row&255;if(col==1)return row>>8;return (row*131+col*17)%251;}
__global__ void generate_ids(int64_t*ids,unsigned n,const unsigned*seed){for(unsigned i=threadIdx.x;i<n;i+=blockDim.x)ids[i]=(i*109u+*seed)%8192u;}
__global__ void gin_read(ncclDevComm dc,ncclWindow_t remote,ncclWindow_t local,const int64_t*ids,unsigned n){ncclGin g{dc,0,BENCH_GIN_SHARING};for(unsigned i=threadIdx.x;i<n;i+=blockDim.x)g.get(ncclTeamWorld(dc),0,remote,size_t(ids[i])*264,local,64+size_t(i)*264,264,ncclCoopThread{},ncclGin_None{},ncclGinOptFlagsDefault,ncclGin_SegmentMixed{});g.flush(ncclCoopCta{});}
__global__ void consume(const uint8_t*data,unsigned n,unsigned long long*sum){unsigned long long x=0;for(unsigned i=threadIdx.x;i<n*264;i+=blockDim.x)x+=data[64+i];atomicAdd(sum,x);}
struct Sample{uint64_t latency,cpu,thread;float gpu_us;};
int main(int argc,char**argv){
 if(argc!=9){fprintf(stderr,"usage: %s rank address control-port nixl-port shim output warmups samples\n",argv[0]);return 2;}
 int rank=atoi(argv[1]),port=atoi(argv[3]),np=atoi(argv[4]),warm=atoi(argv[7]),samples=atoi(argv[8]);if((rank!=0&&rank!=1)||warm<1||samples<2)return 2;
 // Refuse a host doorbell proxy as well as the NCCL GIN proxy backend.
 const char*handler=getenv("NCCL_GIN_GDAKI_NIC_HANDLER");
 if(!handler||strcmp(handler,"2")){fprintf(stderr,"require NCCL_GIN_GDAKI_NIC_HANDLER=2 (GPU SM doorbell)\n");return 3;}
 setenv("NCCL_GIN_TYPE","3",1);
 RTCHK(cudaSetDevice(0));RTCHK(cudaFree(nullptr));ncclUniqueId id{};int fd=endpoint(rank,argv[2],port,&id);ncclComm_t comm;NCCHECK(ncclCommInitRank(&comm,2,id,rank));ncclCommProperties_t props=NCCL_COMM_PROPERTIES_INITIALIZER;NCCHECK(ncclCommQueryProperties(comm,&props));if(!props.deviceApiSupport||!props.ginSupport[NCCL_GIN_TYPE_GDAKI]){fprintf(stderr,"GDAKI unavailable; no proxy fallback allowed\n");return 3;}
 auto source=alloc_host(8192*264);auto*bytes=reinterpret_cast<uint8_t*>(source.ptr);for(unsigned i=0;i<8192;i++)for(unsigned j=0;j<264;j++)bytes[i*264+j]=value(i,j)^(rank?128:0);
 constexpr size_t destsize=64+4096*264+128;void*dest=nullptr;NCCHECK(ncclMemAlloc(&dest,destsize));ncclWindow_t sw{},dw{};NCCHECK(ncclCommWindowRegister(comm,reinterpret_cast<void*>(source.ptr),source.size,&sw,NCCL_WIN_COLL_SYMMETRIC));NCCHECK(ncclCommWindowRegister(comm,dest,destsize,&dw,NCCL_WIN_COLL_SYMMETRIC));ncclDevCommRequirements req=NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;req.ginConnectionType=NCCL_GIN_CONNECTION_FULL;req.ginType=NCCL_GIN_TYPE_GDAKI;req.ginForceEnable=true;req.ginContextCount=1;req.ginSignalCount=1;ncclDevComm dc{};NCCHECK(ncclDevCommCreate(comm,&req,&dc));
 if(dc.ginConnectionCount!=1||dc.ginNetDeviceTypes[0]!=NCCL_NET_DEVICE_GIN_GDAKI||dc.lsaSize!=1){fprintf(stderr,"require GDAKI with one connection and LSA size 1 (no local-copy bypass)\n");return 3;}
 int runtime_version=0;NCCHECK(ncclGetVersion(&runtime_version));Dl_info loaded{};dladdr(reinterpret_cast<void*>(&ncclGetVersion),&loaded);
 printf("GDAKI_VERIFIED rank=%d device_type=%u contexts=%u nic_handler=2 nccl=%d library=%s\n",rank,unsigned(dc.ginNetDeviceTypes[0]),dc.ginContextCount,runtime_version,loaded.dli_fname?loaded.dli_fname:"unknown");fflush(stdout);
 Nixl nx(argv[5],np,rank==0?"gin_nixl_owner":"gin_nixl_reader");nx.regist(rank==0,rank==0?reinterpret_cast<void*>(source.ptr):dest,rank==0?source.size:destsize);
 if(rank==0){auto m=nx.md();uint64_t hdr[3]={uint64_t(m.size()),uint64_t(source.ptr),uint64_t(source.size)};transfer(fd,hdr,sizeof(hdr),true);transfer(fd,m.data(),m.size(),true);char x;transfer(fd,&x,1,false);}
 else {
  uint64_t hdr[3];transfer(fd,hdr,sizeof(hdr),false);if(!hdr[0]||hdr[0]>16*1024*1024){fprintf(stderr,"invalid metadata length\n");return 4;}std::vector<char> m(hdr[0]);transfer(fd,m.data(),m.size(),false);nx.attach(m,hdr[1],hdr[2]);
  cudaStream_t stream;RTCHK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));int64_t*ids=nullptr;unsigned*seed=nullptr;unsigned long long*sum=nullptr;uint64_t*stats=nullptr;RTCHK(cudaMalloc(&ids,4096*sizeof(*ids)));RTCHK(cudaMalloc(&seed,sizeof(*seed)));RTCHK(cudaMalloc(&sum,sizeof(*sum)));RTCHK(cudaMalloc(&stats,8*sizeof(*stats)));
  cudaEvent_t start,stop;RTCHK(cudaEventCreate(&start));RTCHK(cudaEventCreate(&stop));
  std::ofstream out(argv[6]);if(!out){fprintf(stderr,"cannot open output\n");return 4;}out<<"{\n\"schema\":2,\"gin_backend\":\"gdaki\",\"nic_handler\":2,\"gin_sharing\":"<<int(BENCH_GIN_SHARING)<<",\"nccl_runtime\":"<<runtime_version<<",\"gin_contexts\":"<<dc.ginContextCount<<",\"warmups\":"<<warm<<",\"samples_per_mode\":"<<samples<<",\"row_bytes\":264,\"modes\":{\n";bool firstN=true;
  for(unsigned n:{384u,4096u}){
   unsigned zero=0;RTCHK(cudaMemsetAsync(sum,0,sizeof(*sum),stream));RTCHK(cudaMemcpyAsync(seed,&zero,sizeof(zero),cudaMemcpyHostToDevice,stream));generate_ids<<<1,128,0,stream>>>(ids,n,seed);nx.read(ids,n,stats,stream);consume<<<1,128,0,stream>>>((uint8_t*)dest,n,sum);nx.retire(stream);RTCHK(cudaStreamSynchronize(stream));
   gin_read<<<1,128,0,stream>>>(dc,sw,dw,ids,n);RTCHK(cudaGetLastError());RTCHK(cudaStreamSynchronize(stream));
   cudaGraph_t gg{},ng{};cudaGraphExec_t ge{},ne{};
   RTCHK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));generate_ids<<<1,128,0,stream>>>(ids,n,seed);gin_read<<<1,128,0,stream>>>(dc,sw,dw,ids,n);consume<<<1,128,0,stream>>>((uint8_t*)dest,n,sum);RTCHK(cudaStreamEndCapture(stream,&gg));RTCHK(cudaGraphInstantiate(&ge,gg,0));
   RTCHK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));generate_ids<<<1,128,0,stream>>>(ids,n,seed);nx.read(ids,n,stats,stream);consume<<<1,128,0,stream>>>((uint8_t*)dest,n,sum);nx.retire(stream);RTCHK(cudaStreamEndCapture(stream,&ng));RTCHK(cudaGraphInstantiate(&ne,ng,0));
   std::vector<Sample> all[4];std::vector<uint8_t> got(destsize);unsigned char modeids[4]={0,1,2,3};
   for(int i=0;i<warm+samples;i++){unsigned z=(unsigned(i)*313u)%8192u;for(int q=0;q<4;q++){int mode=modeids[(q+i)%4];RTCHK(cudaMemcpyAsync(seed,&z,sizeof(z),cudaMemcpyHostToDevice,stream));RTCHK(cudaMemsetAsync(sum,0,sizeof(*sum),stream));RTCHK(cudaMemsetAsync(dest,0xa5,destsize,stream));RTCHK(cudaMemsetAsync(stats,0,8*sizeof(*stats),stream));RTCHK(cudaStreamSynchronize(stream));uint64_t t0=ns(CLOCK_MONOTONIC),c0=ns(CLOCK_PROCESS_CPUTIME_ID),h0=ns(CLOCK_THREAD_CPUTIME_ID);RTCHK(cudaEventRecord(start,stream));
    if(mode==2)RTCHK(cudaGraphLaunch(ge,stream));else if(mode==3)RTCHK(cudaGraphLaunch(ne,stream));else {generate_ids<<<1,128,0,stream>>>(ids,n,seed);if(mode==0){gin_read<<<1,128,0,stream>>>(dc,sw,dw,ids,n);}else{nx.read(ids,n,stats,stream);}consume<<<1,128,0,stream>>>((uint8_t*)dest,n,sum);if(mode==1)nx.retire(stream);}
    RTCHK(cudaGetLastError());RTCHK(cudaEventRecord(stop,stream));RTCHK(cudaStreamSynchronize(stream));uint64_t t1=ns(CLOCK_MONOTONIC),c1=ns(CLOCK_PROCESS_CPUTIME_ID),h1=ns(CLOCK_THREAD_CPUTIME_ID);float gpu_us=0;RTCHK(cudaEventElapsedTime(&gpu_us,start,stop));gpu_us*=1000; // CUDA returns milliseconds.
    RTCHK(cudaMemcpy(got.data(),dest,got.size(),cudaMemcpyDeviceToHost));unsigned long long expected=0,actual=0;for(unsigned j=0;j<n;j++){unsigned row=(j*109u+z)%8192;for(unsigned k=0;k<264;k++){expected+=value(row,k);if(got[64+size_t(j)*264+k]!=value(row,k)){fprintf(stderr,"MISMATCH n=%u mode=%d iteration=%d row=%u byte=%u\n",n,mode,i,j,k);return 5;}}}
    RTCHK(cudaMemcpy(&actual,sum,sizeof(actual),cudaMemcpyDeviceToHost));if(actual!=expected){fprintf(stderr,"consumer checksum mismatch\n");return 5;}
    // The shim uses the last 64B for NIC consistency scratch, not payload.
    constexpr size_t scratch=64+4096*264+64;
    for(size_t j=0;j<scratch;j++)if((j<64||j>=64+size_t(n)*264)&&got[j]!=0xa5){fprintf(stderr,"destination guard mismatch at %zu\n",j);return 5;}
    if(mode==1||mode==3){uint64_t h[8];RTCHK(cudaMemcpy(h,stats,sizeof(h),cudaMemcpyDeviceToHost));if(h[0]||h[4]!=n){fprintf(stderr,"NIXL completion n=%u error=%llu reads=%llu\n",n,(unsigned long long)h[0],(unsigned long long)h[4]);return 6;}}
    if(i>=warm)all[mode].push_back({t1-t0,c1-c0,h1-h0,gpu_us});
   }}
   if(!firstN)out<<",\n";firstN=false;out<<"\""<<n<<"\":{";const char*names[]={"gin_eager","nixl_eager","gin_graph","nixl_graph"};for(int k=0;k<4;k++){if(k)out<<",";out<<"\""<<names[k]<<"\":[";for(size_t j=0;j<all[k].size();j++){if(j)out<<",";auto v=all[k][j];out<<"{\"latency_ns\":"<<v.latency<<",\"process_cpu_ns\":"<<v.cpu<<",\"thread_cpu_ns\":"<<v.thread<<",\"gpu_us\":"<<v.gpu_us<<"}";}out<<"]";}out<<"}";out.flush();printf("PASS rows=%u arms=4 samples=%d\n",n,samples);fflush(stdout);
   RTCHK(cudaGraphExecDestroy(ge));RTCHK(cudaGraphDestroy(gg));RTCHK(cudaGraphExecDestroy(ne));RTCHK(cudaGraphDestroy(ng));
  }
  out<<"\n}}\n";out.close();RTCHK(cudaStreamSynchronize(stream));nx.close();RTCHK(cudaEventDestroy(start));RTCHK(cudaEventDestroy(stop));RTCHK(cudaFree(stats));RTCHK(cudaFree(sum));RTCHK(cudaFree(seed));RTCHK(cudaFree(ids));RTCHK(cudaStreamDestroy(stream));char x='D';transfer(fd,&x,1,true);
 }
 if(rank==0)nx.close();close(fd);NCCHECK(ncclDevCommDestroy(comm,&dc));NCCHECK(ncclCommWindowDeregister(comm,dw));NCCHECK(ncclCommWindowDeregister(comm,sw));NCCHECK(ncclMemFree(dest));CUCHK(cuMemUnmap(source.ptr,source.size));CUCHK(cuMemAddressFree(source.ptr,source.size));CUCHK(cuMemRelease(source.handle));NCCHECK(ncclCommFinalize(comm));NCCHECK(ncclCommDestroy(comm));printf("CLEAN_CLOSE rank=%d\n",rank);return 0;
}
