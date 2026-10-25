// SPDX-License-Identifier: Apache-2.0
#include "../../workdir/kernel_copy.cuh"
#include "abi.cuh"

// The public usage demonstration: no posting kernel, no per-copy descriptors,
// no request handle. CPU prepares the context and registrations only once.
__global__ void thin_example(local_copy::Copy* prepared,const uint32_t* src,
                             uint32_t* dst,uint32_t* output,int* error){
    local_copy::ThreadCopy copy(prepared);
    __shared__ int status;
    if(threadIdx.x==0)status=copy.copy_async(dst,src,128*sizeof(uint32_t));
    __syncthreads();

    uint32_t independent=independentWork(42,0,threadIdx.x,128);
    asm volatile("" : "+r"(independent) :: "memory");

    if(threadIdx.x==0)status=copy.wait();
    __syncthreads(); // Completion is now visible to every CTA consumer.
    if(!status)output[threadIdx.x]=dst[threadIdx.x]^independent;
    if(threadIdx.x==0)*error=status;
}
extern "C" int lc_thin_example(void* prepared,const void* src,void* dst,void* output,void* error){
    thin_example<<<1,128>>>(static_cast<local_copy::Copy*>(prepared),static_cast<const uint32_t*>(src),static_cast<uint32_t*>(dst),static_cast<uint32_t*>(output),static_cast<int*>(error));return cudaGetLastError();
}

__global__ void thin_warp_example(local_copy::Copy* prepared,const uint32_t* src,
                                  uint32_t* dst,uint32_t* output,int* error){
    local_copy::WarpCopy copy(prepared);
    int lane=threadIdx.x;
    copy.copy_async(dst+lane*16,src+lane*16,16*sizeof(uint32_t));
    uint32_t independent=independentWork(43,0,lane,128);
    asm volatile("" : "+r"(independent) :: "memory");
    int status=copy.wait();
    if(!status)for(int i=0;i<16;++i)output[lane*16+i]=dst[lane*16+i]^independent;
    if(lane==0)*error=status;
}
extern "C" int lc_thin_warp_example(void* prepared,const void* src,void* dst,void* output,void* error){
    thin_warp_example<<<1,32>>>(static_cast<local_copy::Copy*>(prepared),static_cast<const uint32_t*>(src),static_cast<uint32_t*>(dst),static_cast<uint32_t*>(output),static_cast<int*>(error));return cudaGetLastError();
}
