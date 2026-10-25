// SPDX-License-Identifier: Apache-2.0
#include "../../workdir/kernel_copy.cuh"
#include "tma_control.cuh"
#include "abi.cuh"

// Kinds: scalar legacy/batched completion, warp legacy/batched/parallel
// completion, SM control, TMA-to-HBM control. Exactly one application kernel.
template<int Kind,bool Serialized,bool Profile>
__global__ void primitive(local_copy::Copy* prepared,const uint8_t* src,uint8_t* dst,uint32_t* out,
                          size_t bytes,int fragments,int work,int rounds,unsigned seed,PrimitiveStats* stats){
    constexpr bool Warp=Kind>=2;
    constexpr int Completion=(Kind==0||Kind==2)?0:(Kind==4?2:1);
    local_copy::KernelCopy<Warp,Completion> copy(prepared);
    __shared__ int status;
    __shared__ alignas(128) uint8_t tile[Kind==6?16384:16];
    __shared__ alignas(8) unsigned long long barrier;
    if(threadIdx.x==0){status=0;if constexpr(Kind==6)initBarrier(&barrier);}
    __syncthreads();
    unsigned long long s=0,c=0,w=0,u=0;
    const size_t pitch=bytes+64,words=bytes*fragments/4;
    for(int round=0;round<rounds;++round){
        unsigned long long t0=0,t1=0,t2=0,t3=0,t4=0;
        if constexpr(Profile)if(threadIdx.x==0)t0=clock64();
        if constexpr(Kind<=1){
            if(threadIdx.x==0)for(int f=0;f<fragments;++f){int e=copy.copy_async(dst+primitiveGuard+f*bytes,src+primitiveGuard+sourceRow(round,f,seed)*pitch,bytes);if(e){status=e;break;}}
        }else if constexpr(Kind<=4){
            if(threadIdx.x<32){bool active=int(threadIdx.x)<fragments;int e=copy.copy_async(active?dst+primitiveGuard+threadIdx.x*bytes:dst,active?src+primitiveGuard+sourceRow(round,threadIdx.x,seed)*pitch:src,bytes,active);if(threadIdx.x==0)status=e;}
        }else if constexpr(Kind==5){
            for(size_t v=threadIdx.x;v<words/4;v+=blockDim.x){size_t f=(v*16)/bytes,offset=(v*16)%bytes;reinterpret_cast<uint4*>(dst+primitiveGuard)[v]=*reinterpret_cast<const uint4*>(src+primitiveGuard+sourceRow(round,f,seed)*pitch+offset);}
        }else{
            if(threadIdx.x==0){expectBytes(&barrier,bytes*fragments);for(int f=0;f<fragments;++f)loadBulk(tile+f*bytes,src+primitiveGuard+sourceRow(round,f,seed)*pitch,bytes,&barrier);}
        }
        __syncthreads();
        if constexpr(Profile)if(threadIdx.x==0)t1=clock64();
        auto wait=[&](){
            if constexpr(Kind<=1){if(threadIdx.x==0)status=copy.wait();}
            else if constexpr(Kind<=4){if(threadIdx.x<32){int e=copy.wait();if(threadIdx.x==0)status=e;}}
            else if constexpr(Kind==6){if(threadIdx.x==0){waitBarrier(&barrier,round&1);storeBulkAndWait(dst+primitiveGuard,tile,bytes*fragments);}}
            __syncthreads();
        };
        if constexpr(Serialized){wait();if constexpr(Profile)if(threadIdx.x==0)t2=clock64();}
        uint32_t independent=independentWork(seed,round,threadIdx.x,work);
        // Force the independent computation to finish before the wait/consume
        // boundary; keep its result live in the subsequent real data consumer.
        asm volatile("" : "+r"(independent) :: "memory");
        __syncthreads();
        if constexpr(Profile)if(threadIdx.x==0){if constexpr(Serialized)t3=clock64();else t2=clock64();}
        if constexpr(!Serialized){wait();if constexpr(Profile)if(threadIdx.x==0)t3=clock64();}
        if(status){if(threadIdx.x==0)stats->status=status;return;}
        for(size_t i=threadIdx.x;i<words;i+=blockDim.x)out[size_t(round)*words+i]=reinterpret_cast<const uint32_t*>(dst+primitiveGuard)[i]^independent;
        __syncthreads();
        if constexpr(Profile)if(threadIdx.x==0){t4=clock64();s+=t1-t0;u+=t4-t3;if constexpr(Serialized){w+=t2-t1;c+=t3-t2;}else{c+=t2-t1;w+=t3-t2;}}
    }
    if(threadIdx.x==0){stats->submit=s;stats->compute=c;stats->wait=w;stats->consume=u;stats->status=0;
        if constexpr(Kind<5){stats->issued=prepared->qp->sq_rsvd_index;stats->retired=prepared->retired;stats->mcst=prepared->mcst;stats->cqci=doca_gpu_dev_verbs_qp_get_cq<DOCA_GPUNETIO_VERBS_QP_SQ>(prepared->qp)->cqe_ci;stats->owner=prepared->owner;}}
}
template<int Kind> void* choose(bool serial,bool profile){
    if(profile)return serial?reinterpret_cast<void*>(primitive<Kind,true,true>):reinterpret_cast<void*>(primitive<Kind,false,true>);
    return serial?reinterpret_cast<void*>(primitive<Kind,true,false>):reinterpret_cast<void*>(primitive<Kind,false,false>);
}
extern "C" void* lc_primitive_kernel(int kind,int serial,int profile){
    switch(kind){case 0:return choose<0>(serial,profile);case 1:return choose<1>(serial,profile);case 2:return choose<2>(serial,profile);case 3:return choose<3>(serial,profile);case 4:return choose<4>(serial,profile);case 5:return choose<5>(serial,profile);case 6:return choose<6>(serial,profile);default:return nullptr;}
}

// Facade-only refusal/drain checks complement the original low-level CQ/error
// regressions. Errors are deliberately issued without abandoning admitted work.
template<bool Warp,int Completion>
__global__ void facadeCheck(local_copy::Copy* prepared,const uint8_t* src,uint8_t* dst,int mode,int* errors){
    local_copy::KernelCopy<Warp,Completion> copy(prepared);
    if constexpr(!Warp){if(threadIdx.x!=0)return;}
    int lane=threadIdx.x;
    int first=copy.wait();if(first&&lane==0)atomicAdd(errors,1);
    if(mode==0){
        int e=copy.copy_async(dst+primitiveGuard+lane*64,src+primitiveGuard+lane*64,64,!Warp||lane<31);
        int refused=copy.copy_async(dst+prepared->dst_size,src,64,!Warp||lane==0);
        int result=copy.wait();
        if(lane==0&&(e||refused!=-EINVAL||result!=-EINVAL||prepared->retired!=prepared->qp->sq_rsvd_index))atomicAdd(errors,1);
    }else{
        int count=Warp?1:32;
        for(int j=0;j<count;++j){int e=copy.copy_async(dst+primitiveGuard+(Warp?lane:j)*64,src+primitiveGuard+(Warp?lane:j)*64,64);if(e&&lane==0)atomicAdd(errors,1);}
        int refused=copy.copy_async(dst+primitiveGuard+4096,src+primitiveGuard+4096,64,!Warp||lane==0);
        int result=copy.wait();if(lane==0&&(refused!=-EAGAIN||result!=-EAGAIN||prepared->retired!=prepared->qp->sq_rsvd_index))atomicAdd(errors,1);
    }
    // Reuse after a fully drained admission failure must work normally.
    int e=copy.copy_async(dst+primitiveGuard+lane*64,src+primitiveGuard+lane*64,64,!Warp||lane<31);
    int result=copy.wait();if(lane==0&&(e||result))atomicAdd(errors,1);
}
extern "C" int lc_facade_test(void* copy,const void* src,void* dst,int kind,int mode,void* errors){
    auto* c=static_cast<local_copy::Copy*>(copy);auto* s=static_cast<const uint8_t*>(src);auto* d=static_cast<uint8_t*>(dst);auto* e=static_cast<int*>(errors);
    if(kind==0)facadeCheck<false,1><<<1,1>>>(c,s,d,mode,e);
    else if(kind==1)facadeCheck<true,1><<<1,32>>>(c,s,d,mode,e);
    else facadeCheck<true,2><<<1,32>>>(c,s,d,mode,e);
    return cudaGetLastError();
}

__global__ void warpContract(local_copy::Copy* copy,const uint8_t* src,uint8_t* dst,int mode,int* errors){
    __shared__ local_copy::Ticket ticket;
    int lane=threadIdx.x;
    if(mode==3){
        if(lane==0){auto a=copy->copy_async(dst+4096,src+4096,64);auto key=copy->rkey;copy->rkey^=0x01000000;auto b=copy->copy_async(dst+8192,src+8192,64);copy->rkey=key;ticket=copy->copy_async(dst+12288,src+12288,64);if(a.error||b.error||ticket.error)atomicAdd(errors,1);}
        __syncwarp();int e=copy->wait_warp(ticket);
        if(lane==0){auto refused=copy->copy_async(dst,src,64);if(e!=-EIO||refused.error!=-EIO||copy->recovery->stopped!=1)atomicAdd(errors,1);}return;
    }
#ifdef LOCAL_COPY_TESTING
    if(mode==4){
        // Deliberately reserve without posting on a fresh test-owned QP.
        // Its absent CQE must hit the normal deadline and reset/quarantine path.
        if(lane==0){
            auto first=copy->qp->sq_rsvd_index++;copy->owner=copy->grid();ticket={first,first,0,copy->cookie};
            auto* cq=doca_gpu_dev_verbs_qp_get_cq<DOCA_GPUNETIO_VERBS_QP_SQ>(copy->qp);
            uint64_t physical=first+cq->cqe_rsvd;uint32_t slot=physical&(cq->cqe_num-1);
            auto* cqe=reinterpret_cast<doca_gpu_dev_verbs_cqe64*>(reinterpret_cast<uint8_t*>(cq->cqe_daddr)+slot*DOCA_GPUNETIO_VERBS_CQE_SIZE);
            *reinterpret_cast<volatile uint8_t*>(&cqe->op_own)=uint8_t((MLX5_CQE_INVALID<<4)|(!bool(physical&cq->cqe_num)));
            __threadfence_system();
        }
        __syncwarp();int e=copy->wait_warp(ticket);
        if(lane==0&&(e!=-ETIMEDOUT||copy->recovery->stopped!=1))atomicAdd(errors,1);return;
    }
#endif
    bool original=copy->mcst;
    if(mode==2){if(lane==0)copy->mcst=true;__syncwarp();}
    auto t=copy->copy_async_warp(dst+4096+lane*64,src+4096+lane*64,64,lane<(mode==2?31:32));
    if(t.error){if(lane==0)atomicAdd(errors,1);return;}
    if(mode==0){
        auto wrong=t;if(lane==7)wrong.cookie++;
        if(copy->wait_warp(wrong)!=-EINVAL&&lane==0)atomicAdd(errors,1);
        wrong=t;wrong.last=copy->qp->sq_rsvd_index;
        if(copy->wait_warp(wrong)!=-EINVAL&&lane==0)atomicAdd(errors,1);
        if(lane<16&&copy->wait_warp(t)!=-EINVAL)atomicAdd(errors,1);
        __syncwarp();
    }
    int e=copy->wait_warp(t);if(e&&lane==0)atomicAdd(errors,1);
    if(copy->wait_warp(t)!=-EINVAL&&lane==0)atomicAdd(errors,1);
    if(mode==2){__syncwarp();if(lane==0)copy->mcst=original;__syncwarp();}
    for(int i=lane;i<(mode==2?31:32)*64;i+=32)if(dst[4096+i]!=src[4096+i])atomicAdd(errors,1);
}
extern "C" int lc_warp_contract(void* copy,const void* src,void* dst,int mode,void* errors){
    warpContract<<<1,32>>>(static_cast<local_copy::Copy*>(copy),static_cast<const uint8_t*>(src),static_cast<uint8_t*>(dst),mode,static_cast<int*>(errors));return cudaGetLastError();
}
