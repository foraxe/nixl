// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cerrno>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <doca_gpunetio_dev_verbs_onesided.cuh>

namespace local_copy {

// Host-mapped recovery state. Only the failure path requires host intervention.
struct alignas(64) Recovery {
    int fault = 0;
    int stopped = 0;
};

struct Ticket {
    uint64_t first = 0;
    uint64_t last = 0;
    int error = 0;
    uint64_t cookie = 0;
};

// One submission thread (CTA0/thread0) per context; wait before handing it off.
struct Copy {
    doca_gpu_dev_verbs_qp *qp;
    Recovery *recovery;
    uintptr_t src_base, dst_base;
    uint64_t src_size, dst_size;
    uint32_t rkey, lkey;
    doca_gpu_dev_verbs_addr dump;
    bool mcst;
    uint64_t cookie;
    uint64_t owner = 0;
    uint64_t retired = 0;
    int first_fault = 0;

    __device__ static uint64_t grid() {
        uint64_t id;
        asm volatile("mov.u64 %0, %%gridid;" : "=l"(id));
        return id + 1;
    }
    __device__ static bool caller() {
        return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
               threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
    }
    __device__ static bool range(uintptr_t p, uintptr_t base, uint64_t extent, size_t n) {
        return p >= base && p - base <= extent && n <= extent - (p - base);
    }

    __device__ Ticket copy_async(void *dst, const void *src, size_t bytes) {
        if (!caller()) return {0, 0, -EPERM};
        if (!range(reinterpret_cast<uintptr_t>(src), src_base, src_size, bytes) ||
            !range(reinterpret_cast<uintptr_t>(dst), dst_base, dst_size, bytes) ||
            bytes == 0 || bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)
            return {0, 0, -EINVAL};
        auto fail = cuda::atomic_ref<int, cuda::thread_scope_device>(first_fault);
        if (fail.load(cuda::memory_order_acquire)) return {0, 0, -EIO};
        const uint64_t me = grid();
        auto lock = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(owner);
        uint64_t previous = 0;
        if (!lock.compare_exchange_strong(previous, me, cuda::memory_order_acq_rel) && previous != me)
            return {0, 0, -EBUSY};
        const uint64_t first = qp->sq_rsvd_index;
        const uint64_t cost = mcst ? 2 : 1;
        // Admission before SDK reservation: neither SQ nor CQ can be overrun.
        if (first + cost - retired > 32) {
            if (first == retired) lock.store(0, cuda::memory_order_release);
            return {0, 0, -EAGAIN};
        }
        doca_gpu_dev_verbs_ticket_t last;
        const doca_gpu_dev_verbs_addr remote{reinterpret_cast<uintptr_t>(src), rkey};
        const doca_gpu_dev_verbs_addr local{reinterpret_cast<uintptr_t>(dst), lkey};
        if (mcst)
            doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB,
                DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD, DOCA_GPUNETIO_VERBS_MCST_ENABLED>(
                    qp, remote, local, bytes, dump, &last);
        else
            doca_gpu_dev_verbs_get<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
                    qp, remote, local, bytes, dump, &last);
        return {first, last, 0, cookie};
    }

    __device__ int recover(int cause) {
        auto fail = cuda::atomic_ref<int, cuda::thread_scope_device>(first_fault);
        int expected = 0;
        if (cause && fail.compare_exchange_strong(expected, cause, cuda::memory_order_acq_rel))
            cuda::atomic_ref<int, cuda::thread_scope_system>(recovery->fault)
                .store(cause, cuda::memory_order_release);
        auto stopped = cuda::atomic_ref<int, cuda::thread_scope_system>(recovery->stopped);
        // Never report reusable buffers if the NIC cannot be stopped.
        while (stopped.load(cuda::memory_order_acquire) != 1) __nanosleep(1000);
        return fail.load(cuda::memory_order_acquire);
    }

    __device__ static uint64_t broadcast(uint64_t value) {
        return uint64_t(__shfl_sync(0xffffffffu, uint32_t(value), 0)) |
               (uint64_t(__shfl_sync(0xffffffffu, uint32_t(value >> 32), 0)) << 32);
    }

    // Warp0 cooperates: each active lane supplies its own address/length.
    // No pointer table or per-copy host descriptor; lane0 waits the common ticket.
    __device__ Ticket copy_async_warp(void *dst, const void *src, size_t bytes, bool active = true) {
        if (__activemask() != 0xffffffffu) return {0, 0, -EINVAL};
        if (blockIdx.x || blockIdx.y || blockIdx.z || blockDim.y != 1 || blockDim.z != 1 ||
            threadIdx.x >= 32) return {0, 0, -EPERM};
        const int lane = threadIdx.x;
        const uint64_t context = reinterpret_cast<uintptr_t>(this);
        const uint64_t leader = broadcast(context);
        bool invalid = context != leader || (active &&
            (!range(reinterpret_cast<uintptr_t>(src), src_base, src_size, bytes) ||
             !range(reinterpret_cast<uintptr_t>(dst), dst_base, dst_size, bytes) ||
             bytes == 0 || bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE));
        if (__any_sync(0xffffffffu, invalid)) return {0, 0, -EINVAL};
        const uint32_t members = __ballot_sync(0xffffffffu, active);
        const uint32_t count = __popc(members);
        if (!count) return {0, 0, -EINVAL};
        int status = 0;
        uint64_t first = 0;
        const uint32_t cost = count + (mcst ? 1 : 0);
        if (lane == 0) {
            if (cuda::atomic_ref<int, cuda::thread_scope_device>(first_fault)
                    .load(cuda::memory_order_acquire)) status = -EIO;
            else {
                auto lock = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(owner);
                uint64_t previous = 0;
                if (!lock.compare_exchange_strong(previous, grid(), cuda::memory_order_acq_rel) &&
                    previous != grid()) status = -EBUSY;
                else {
                    first = qp->sq_rsvd_index;
                    if (first + cost - retired > 32) {
                        status = -EAGAIN;
                        if (first == retired) lock.store(0, cuda::memory_order_release);
                    } else {
                        first = doca_gpu_dev_verbs_reserve_wq_slots<
                            DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, cost);
                    }
                }
            }
        }
        status = __shfl_sync(0xffffffffu, status, 0);
        if (status) return {0, 0, status};
        first = broadcast(first);
        const uint64_t last = first + cost - 1;
        if (active) {
            const uint64_t index = first + __popc(members & ((1u << lane) - 1));
            doca_gpu_dev_verbs_wqe_prepare_read(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, index),
                index, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                reinterpret_cast<uintptr_t>(src), rkey,
                reinterpret_cast<uintptr_t>(dst), lkey, bytes);
        }
        __syncwarp();
        if (lane == 0) {
            if (mcst)
                doca_gpu_dev_verbs_wqe_prepare_dump(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, last),
                    last, DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, dump.addr, dump.key, 1);
            doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
                qp, first, last);
            doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
                    qp, last + 1);
        }
        __syncwarp();
        return {first, last, 0, cookie};
    }

    // Retained for matched measurement of the original completion path.
    __device__ int wait_legacy(Ticket ticket) {
        if (ticket.error) return ticket.error;
        if (ticket.cookie != cookie || ticket.first > ticket.last) return -EINVAL;
        auto lock = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(owner);
        if (!caller() || lock.load(cuda::memory_order_acquire) != grid()) return -EPERM;
        if (cuda::atomic_ref<int, cuda::thread_scope_device>(first_fault)
                .load(cuda::memory_order_acquire)) return recover(0);
        if (ticket.first < retired || ticket.last >= qp->sq_rsvd_index) return -EINVAL;
        // Inspect every CQE: polling just the last CQE could skip an earlier error.
        const uint64_t deadline = clock64() + 2000000000ULL;
        for (uint64_t i = retired; i <= ticket.last; ++i) {
            int status;
            do {
                status = doca_gpu_dev_verbs_poll_one_cq_at<
                    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, i);
            } while (status == EBUSY && clock64() < deadline);
            if (status != 0) {
                return recover(status == EBUSY ? -ETIMEDOUT : status);
            }
            retired = i + 1;
        }
        if (retired == qp->sq_rsvd_index)
            lock.store(0, cuda::memory_order_release);
        return 0;
    }

    __device__ int wait(Ticket ticket) {
        if (ticket.error) return ticket.error;
        if (ticket.cookie != cookie || ticket.first > ticket.last) return -EINVAL;
        auto lock = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(owner);
        if (!caller() || lock.load(cuda::memory_order_acquire) != grid()) return -EPERM;
        if (cuda::atomic_ref<int, cuda::thread_scope_device>(first_fault)
                .load(cuda::memory_order_acquire)) return recover(0);
        if (ticket.first < retired || ticket.last >= qp->sq_rsvd_index) return -EINVAL;
        auto* cq = doca_gpu_dev_verbs_qp_get_cq<DOCA_GPUNETIO_VERBS_QP_SQ>(qp);
        // Exclusive owner; <=32 admitted entries fit the runtime's 128-entry CQ.
        // Observe every CQE, including intermediate errors, without publishing
        // partial progress or repeating the data-acquire fence for each entry.
        const uint64_t deadline = clock64() + 2000000000ULL;
        for (uint64_t i = retired; i <= ticket.last; ++i) {
            int status;
            do {
                status = doca_priv_gpu_dev_verbs_poll_one_cq_device_at<
                    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(cq, i);
            } while (status == EBUSY && clock64() < deadline);
            if (status != 0) return recover(status == EBUSY ? -ETIMEDOUT : status);
        }
        doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
        doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            &cq->cqe_ci, ticket.last + 1);
        retired = ticket.last + 1;
        if (retired == qp->sq_rsvd_index) lock.store(0, cuda::memory_order_release);
        return 0;
    }

    // Warp0 cooperatively inspects the same bounded batch, then lane0 acquires
    // payload visibility and publishes retirement. No CQE/error is skipped.
    __device__ int wait_warp(Ticket ticket) {
        if (__activemask() != 0xffffffffu) return -EINVAL;
        if (blockIdx.x || blockIdx.y || blockIdx.z || blockDim.y != 1 || blockDim.z != 1 || threadIdx.x >= 32)
            return -EPERM;
        const int lane = threadIdx.x;
        bool different = reinterpret_cast<uintptr_t>(this) != broadcast(reinterpret_cast<uintptr_t>(this));
        const uint64_t leader_first=broadcast(ticket.first), leader_last=broadcast(ticket.last),
                       leader_cookie=broadcast(ticket.cookie);
        const int leader_error=__shfl_sync(0xffffffffu, ticket.error, 0);
        different |= ticket.first != leader_first || ticket.last != leader_last ||
                     ticket.cookie != leader_cookie || ticket.error != leader_error;
        if (__any_sync(0xffffffffu, different)) return -EINVAL;
        if (ticket.error) return ticket.error;
        int status = 0;
        if (lane == 0) {
            if (ticket.cookie != cookie || ticket.first > ticket.last || ticket.first < retired ||
                ticket.last >= qp->sq_rsvd_index || ticket.last - retired >= 32) status = -EINVAL;
            else if (cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(owner)
                         .load(cuda::memory_order_acquire) != grid()) status = -EPERM;
            else if (cuda::atomic_ref<int, cuda::thread_scope_device>(first_fault)
                         .load(cuda::memory_order_acquire)) status = recover(0);
        }
        status = __shfl_sync(0xffffffffu, status, 0);
        if (status) return status;
        const uint64_t first = broadcast(retired), index = first + lane;
        auto* cq = doca_gpu_dev_verbs_qp_get_cq<DOCA_GPUNETIO_VERBS_QP_SQ>(qp);
        const uint64_t deadline = clock64() + 2000000000ULL;
        do {
            int own = index <= ticket.last ? doca_priv_gpu_dev_verbs_poll_one_cq_device_at<
                DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(cq, index) : 0;
            if (own == EBUSY && clock64() >= deadline) own = -ETIMEDOUT;
            unsigned failed = __ballot_sync(0xffffffffu, own < 0);
            if (failed) {
                int cause = __shfl_sync(0xffffffffu, own, __ffs(failed)-1);
                if (lane == 0) status = recover(cause);
                return __shfl_sync(0xffffffffu, status, 0);
            }
            if (!__any_sync(0xffffffffu, own == EBUSY)) break;
        } while (true);
        if (lane == 0) {
            doca_gpu_dev_verbs_fence_acquire_nvidia_nic();
            doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
                &cq->cqe_ci, ticket.last + 1);
            retired = ticket.last + 1;
            if (retired == qp->sq_rsvd_index)
                cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(owner).store(0, cuda::memory_order_release);
        }
        __syncwarp();
        return 0;
    }
};
} // namespace local_copy
