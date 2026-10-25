// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
__device__ inline unsigned smaddr(const void* p){return unsigned(__cvta_generic_to_shared(p));}
__device__ inline void initBarrier(unsigned long long* p){
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(smaddr(p)):"memory");
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
}
__device__ inline void expectBytes(unsigned long long* p,unsigned bytes){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(smaddr(p)),"r"(bytes):"memory");
}
__device__ inline void loadBulk(void* dst,const void* src,unsigned bytes,unsigned long long* bar){
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        ::"r"(smaddr(dst)),"l"(src),"r"(bytes),"r"(smaddr(bar)):"memory");
}
__device__ inline void loadGather4(void* dst,const CUtensorMap* map,int x,int y0,int y1,int y2,int y3,unsigned long long* bar){
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4,%5,%6}], [%7];"
        ::"r"(smaddr(dst)),"l"(map),"r"(x),"r"(y0),"r"(y1),"r"(y2),"r"(y3),"r"(smaddr(bar)):"memory");
}
__device__ inline void waitBarrier(unsigned long long* p,unsigned phase){
    unsigned done=0;unsigned long long start=clock64();
    do{
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
            :"=r"(done):"r"(smaddr(p)),"r"(phase):"memory");
        if(clock64()-start>2000000000ull)asm volatile("trap;");
    }while(!done);
}
__device__ inline void storeBulkAndWait(void* dst,const void* src,unsigned bytes){
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"::"l"(dst),"r"(smaddr(src)),"r"(bytes):"memory");
    asm volatile("cp.async.bulk.commit_group;":::"memory");
    // Full completion, not merely .read/source-buffer release.
    asm volatile("cp.async.bulk.wait_group 0;":::"memory");
    asm volatile("fence.proxy.async.global;":::"memory");
    __threadfence();
}
