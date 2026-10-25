// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <nccl_device.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
#include <dlfcn.h>
inline void ck(cudaError_t e,const char* s){if(e)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
inline void nc(ncclResult_t e,const char* s){if(e)throw std::runtime_error(std::string(s)+": "+ncclGetErrorString(e));}
inline void dc(CUresult e,const char* s){if(e){const char* t;cuGetErrorString(e,&t);throw std::runtime_error(std::string(s)+": "+t);}}
inline void bc(cublasStatus_t e,const char* s){if(e)throw std::runtime_error(std::string(s)+": "+std::to_string(e));}
#define C(x) ck((x),#x)
#define N(x) nc((x),#x)
#define D(x) dc((x),#x)
#define B(x) bc((x),#x)
constexpr size_t guard=4096;
struct HostVmm {
    CUdeviceptr address=0;CUmemGenericAllocationHandle handle=0;size_t allocated=0;
    uint8_t* allocate(size_t bytes,int gpu,int numa) {
        CUmemAllocationProp p{};p.type=CU_MEM_ALLOCATION_TYPE_PINNED;
        p.location={CU_MEM_LOCATION_TYPE_HOST_NUMA,numa};
        p.requestedHandleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        size_t gran;D(cuMemGetAllocationGranularity(&gran,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        allocated=(bytes+gran-1)/gran*gran;D(cuMemCreate(&handle,allocated,&p,0));
        CUmemAllocationProp actual{};D(cuMemGetAllocationPropertiesFromHandle(&actual,handle));
        if(actual.location.type!=CU_MEM_LOCATION_TYPE_HOST_NUMA||actual.location.id!=numa)throw std::runtime_error("wrong NUMA backing");
        D(cuMemAddressReserve(&address,allocated,gran,0,0));D(cuMemMap(address,allocated,0,handle,0));
        CUmemAccessDesc a[2]={{{CU_MEM_LOCATION_TYPE_DEVICE,gpu},CU_MEM_ACCESS_FLAGS_PROT_READWRITE},{{CU_MEM_LOCATION_TYPE_HOST_NUMA,numa},CU_MEM_ACCESS_FLAGS_PROT_READWRITE}};
        D(cuMemSetAccess(address,allocated,a,2));return reinterpret_cast<uint8_t*>(address);
    }
    void close(){D(cuMemUnmap(address,allocated));D(cuMemAddressFree(address,allocated));D(cuMemRelease(handle));}
};
__host__ __device__ inline uint32_t pattern(size_t i,unsigned epoch){uint32_t x=uint32_t(i)^epoch*0x9e3779b9u;x^=x>>16;x*=0x7feb352d;x^=x>>15;return x;}
inline float elapsed(cudaEvent_t a,cudaEvent_t b){float t;C(cudaEventElapsedTime(&t,a,b));return t;}
inline cudaGraphNode_t eventNode(cudaGraph_t g,cudaEvent_t e,std::vector<cudaGraphNode_t> d={}){
    cudaGraphNode_t n;C(cudaGraphAddEventRecordNode(&n,g,d.data(),d.size(),e));return n;
}
inline cudaGraphNode_t kernelNode(cudaGraph_t g,void* f,int grid,int threads,void** args,std::vector<cudaGraphNode_t> d={}){
    cudaKernelNodeParams p{};p.func=f;p.gridDim=dim3(grid);p.blockDim=dim3(threads);p.kernelParams=args;
    cudaGraphNode_t n;C(cudaGraphAddKernelNode(&n,g,d.data(),d.size(),&p));return n;
}
