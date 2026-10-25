// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "local_copy.cuh"
namespace local_copy {
// Register-local facade, not a new allocator/session/request system.
// Prepared context has one owner: CTA0/thread0, or the whole of CTA0/warp0.
// Every successful call submits immediately. At most32 WQE credits may be live;
// -EAGAIN is explicit backpressure. wait() drains admitted work even if a later
// call was refused, then reports that error. Buffers stay live until it returns.
// A CTA consumer must synchronize after the issuing thread/warp's wait.
// Warp methods require convergent full-warp participation. With MCST enabled,
// the trailing dump consumes one credit, so at most31 lanes may issue per batch.
// Destruction is not an implicit wait; callers must drain before ending scope.
template<bool Warp, int Completion=1> class KernelCopy {
    Copy* context;
    Ticket pending{};
    bool live=false;
    int first_error=0;
public:
    __device__ explicit KernelCopy(Copy* prepared):context(prepared){}
    __device__ int copy_async(void* dst,const void* src,size_t bytes,bool active=true){
        // Full participation need not mean simultaneous arrival under ITS.
        // Rendezvous before the low-level API's full-active-mask admission.
        if constexpr(Warp) __syncwarp();
        else if(!active){if(!first_error)first_error=-EINVAL;return -EINVAL;}
        Ticket next;
        if constexpr(Warp) next=context->copy_async_warp(dst,src,bytes,active);
        else next=context->copy_async(dst,src,bytes);
        if(next.error){if(!first_error)first_error=next.error;return next.error;}
        pending=next;live=true;return 0;
    }
    __device__ int wait(){
        if constexpr(Warp) __syncwarp();
        int status=0;
        if(live){
            if constexpr(Warp&&Completion==2) status=context->wait_warp(pending);
            else{
                if(!Warp||threadIdx.x==0){
                    if constexpr(Completion==0) status=context->wait_legacy(pending);
                    else status=context->wait(pending);
                }
                if constexpr(Warp){status=__shfl_sync(0xffffffffu,status,0);__syncwarp();}
            }
        }
        live=false;pending={};int result=status?status:first_error;first_error=0;return result;
    }
};
using ThreadCopy=KernelCopy<false>;
using WarpCopy=KernelCopy<true,2>;
} // namespace local_copy
