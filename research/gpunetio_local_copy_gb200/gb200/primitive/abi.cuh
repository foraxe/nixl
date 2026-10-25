// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
struct PrimitiveStats {
    unsigned long long submit=0,compute=0,wait=0,consume=0,issued=0,retired=0,cqci=0,owner=0;
    int status=0,mcst=0;
};
constexpr size_t primitiveGuard=4096;
__host__ __device__ inline uint32_t sourceWord(size_t i,unsigned seed){uint32_t x=uint32_t(i)^seed*0x9e3779b9u;x^=x>>16;x*=0x7feb352d;x^=x>>15;return x;}
__host__ __device__ inline uint32_t independentWork(unsigned seed,int round,int thread,int work){
    uint32_t x=seed^uint32_t(round)*0x85ebca6bu^uint32_t(thread)*0xc2b2ae35u;
    for(int i=0;i<work;++i)x=(x^(x>>13))*1664525u+1013904223u;
    return x;
}
__host__ __device__ inline int sourceRow(int round,int fragment,unsigned seed){return (unsigned(round)*37+unsigned(fragment)*97+seed)&1023;}
