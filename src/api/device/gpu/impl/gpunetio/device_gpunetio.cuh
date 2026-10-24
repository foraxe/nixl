/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH

#include <gpu/device_types.cuh>
#include <gpu/impl/gpunetio/device_gpunetio_compat.cuh>

#include <cstdint>
#include <cuda/atomic>

namespace nixl::gpu::impl::gpunetio {

static_assert(sizeof(nixl_status_t) <= sizeof(int32_t));
static_assert(sizeof(GpunetioStatusPayload) <= 60);
static_assert(xfer_status_payload_size >= 60);

#if NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE && defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
#if defined(NIXL_GPUNETIO_OWNER_GPU_SHARING_CONTROL)
constexpr enum doca_gpu_dev_verbs_resource_sharing_mode owner_resource_sharing_mode =
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
#else
constexpr enum doca_gpu_dev_verbs_resource_sharing_mode owner_resource_sharing_mode =
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
#endif
#endif

__device__ inline GpunetioStatusPayload *
statusPayload(xferStatusH *xfer_status) {
    return reinterpret_cast<GpunetioStatusPayload *>(xfer_status->storage);
}

__device__ inline const GpunetioStatusPayload *
statusPayload(const xferStatusH *xfer_status) {
    return reinterpret_cast<const GpunetioStatusPayload *>(xfer_status->storage);
}

template<typename T> using device_atomic_ref = cuda::atomic_ref<T, cuda::thread_scope_device>;

template<typename T>
__device__ inline device_atomic_ref<T>
deviceAtomic(T &value) {
    return device_atomic_ref<T>(value);
}

__device__ inline void
failLane(GpunetioStatusPayload *payload) {
    payload->cached_status = NIXL_ERR_BACKEND;
    deviceAtomic(payload->lane->failed).store(1U, cuda::memory_order_relaxed);
    deviceAtomic(payload->lane->phase)
        .store(gpunetio_lane_phase_failed, cuda::memory_order_release);
}

__device__ inline bool
isOwnerActor() {
    return blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 &&
        threadIdx.y == 0 && threadIdx.z == 0;
}

__device__ inline uint64_t
currentGridId() {
    uint64_t grid_id;
    asm volatile("mov.u64 %0, %gridid;" : "=l"(grid_id));
    return grid_id;
}

__device__ inline bool
claimOwnerToken(GpunetioDeviceLane *lane, uint64_t owner_token) {
    if (owner_token == 0 || !isOwnerActor()) {
        return false;
    }
    const uint64_t grid_id = currentGridId();
    auto claim = deviceAtomic(lane->owner_claim);
    const uint32_t observed = claim.load(cuda::memory_order_acquire);
    if (observed == 0) {
        uint32_t empty = 0;
        if (!claim.compare_exchange_strong(
                empty, 1, cuda::memory_order_acq_rel, cuda::memory_order_acquire)) {
            return false;
        }
        if (lane->owner_expected_token == 0 || lane->owner_expected_token != owner_token) {
            claim.store(0, cuda::memory_order_release);
            return false;
        }
        lane->owner_token = owner_token;
        lane->owner_grid_id = grid_id;
        claim.store(2, cuda::memory_order_release);
        return true;
    }
    if (observed != 2) {
        return false;
    }
    return lane->owner_token == owner_token && lane->owner_grid_id == grid_id;
}

__device__ inline bool
matchesOwnerLease(GpunetioDeviceLane *lane, uint64_t owner_token) {
    return owner_token != 0 && isOwnerActor() &&
        deviceAtomic(lane->owner_claim).load(cuda::memory_order_acquire) == 2 &&
        lane->owner_token == owner_token && lane->owner_grid_id == currentGridId();
}

// The dispatcher calls this only after it has established that the status
// carries the GPUNETIO_DIRECT mode tag.  put() intentionally does not read an
// untagged caller buffer: a caller may pass an uninitialized xferStatusH.
__device__ inline bool
isStatusInProgress(const xferStatusH &xfer_status) {
    const auto *payload = statusPayload(&xfer_status);
    return payload->initialized != 0 && payload->cached_status == NIXL_IN_PROG;
}

__device__ inline bool
rangeIsValid(uint64_t base, uint64_t length, size_t offset, size_t bytes, uint64_t *address) {
    if (offset > length || bytes > length - offset) {
        return false;
    }
    const uint64_t offset64 = static_cast<uint64_t>(offset);
    const uint64_t bytes64 = static_cast<uint64_t>(bytes);
    if (static_cast<size_t>(offset64) != offset || static_cast<size_t>(bytes64) != bytes) {
        return false;
    }
    if (base > ~uint64_t{0} - offset64) {
        return false;
    }
    *address = base + offset64;
    return *address <= ~uint64_t{0} - bytes64;
}

template<level_t level>
__device__ nixl_status_t
putViews(const GpunetioDeviceView *src_view,
         size_t src_index,
         size_t src_offset,
         const GpunetioDeviceView *dst_view,
         size_t dst_index,
         size_t dst_offset,
         size_t bytes,
         unsigned channel_id,
         uint64_t flags,
         xferStatusH *xfer_status) {
#if !NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    (void)src_view;
    (void)src_index;
    (void)src_offset;
    (void)dst_view;
    (void)dst_index;
    (void)dst_offset;
    (void)bytes;
    (void)channel_id;
    (void)flags;
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    if constexpr (level != level_t::THREAD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (channel_id != 0 || flags != 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (xfer_status == nullptr || bytes == 0 ||
        bytes > static_cast<size_t>(DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (src_view == nullptr || dst_view == nullptr ||
        src_view->abi_version != gpunetio_device_view_abi_version ||
        dst_view->abi_version != gpunetio_device_view_abi_version ||
        src_view->role != gpunetio_device_view_role_local ||
        dst_view->role != gpunetio_device_view_role_remote || src_view->context == nullptr ||
        src_view->context != dst_view->context || src_view->context_cookie == 0 ||
        src_view->context_cookie != dst_view->context_cookie ||
        src_view->execution_gpu != dst_view->execution_gpu || src_view->context->lane == nullptr ||
        src_view->elems == nullptr || dst_view->elems == nullptr || src_index >= src_view->count ||
        dst_index >= dst_view->count) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto &src = src_view->elems[src_index];
    const auto &dst = dst_view->elems[dst_index];
    uint64_t local_address;
    uint64_t remote_address;
    if (src.valid == 0 || dst.valid == 0 || src.peer_slot != gpunetio_local_peer_slot ||
        dst.peer_slot != gpunetio_remote_peer_slot ||
        !rangeIsValid(src.base, src.length, src_offset, bytes, &local_address) ||
        !rangeIsValid(dst.base, dst.length, dst_offset, bytes, &remote_address)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *lane = src_view->context->lane;
    if (lane->owner_enabled != 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    auto phase = deviceAtomic(lane->phase);
    uint32_t observed_phase = gpunetio_lane_phase_idle;
    if (!phase.compare_exchange_strong(observed_phase,
                                       gpunetio_lane_phase_posting,
                                       cuda::memory_order_acq_rel,
                                       cuda::memory_order_acquire)) {
        return observed_phase == gpunetio_lane_phase_failed ||
                deviceAtomic(lane->failed).load(cuda::memory_order_relaxed) != 0 ?
            NIXL_ERR_BACKEND :
            NIXL_ERR_NOT_ALLOWED;
    }
    if (deviceAtomic(lane->failed).load(cuda::memory_order_relaxed) != 0 || lane->qp == nullptr ||
        !compat::usesGpuSmDoorbell(lane->qp)) {
        deviceAtomic(lane->failed).store(1U, cuda::memory_order_relaxed);
        phase.store(gpunetio_lane_phase_failed, cuda::memory_order_release);
        return NIXL_ERR_BACKEND;
    }

    auto generation_ref = deviceAtomic(lane->generation);
    const uint64_t previous_generation = generation_ref.load(cuda::memory_order_relaxed);
    if (previous_generation == ~uint64_t{0}) {
        phase.store(gpunetio_lane_phase_idle, cuda::memory_order_release);
        return NIXL_ERR_BACKEND;
    }
    const uint64_t generation = generation_ref.fetch_add(1U, cuda::memory_order_relaxed) + 1U;

    uint64_t ticket;
    compat::releaseFence();
    compat::put(lane->qp, remote_address, dst.key, local_address, src.key, bytes, &ticket);
    deviceAtomic(lane->active_ticket).store(ticket, cuda::memory_order_relaxed);

    // All state becomes visible before POSTED.  The common dispatcher writes
    // the final four-byte mode tag only after this function returns IN_PROG.
    auto *payload = statusPayload(xfer_status);
    payload->lane = lane;
    payload->generation = generation;
    payload->ticket = ticket;
    payload->cached_status = NIXL_IN_PROG;
    payload->initialized = 1;
    phase.store(gpunetio_lane_phase_posted, cuda::memory_order_release);
    return NIXL_IN_PROG;
#endif
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t bytes,
    unsigned channel_id,
    uint64_t flags,
    xferStatusH *xfer_status) {
    return putViews<level>(static_cast<const GpunetioDeviceView *>(src.mvh),
                           src.index,
                           src.offset,
                           static_cast<const GpunetioDeviceView *>(dst.mvh),
                           dst.index,
                           dst.offset,
                           bytes,
                           channel_id,
                           flags,
                           xfer_status);
}

template<level_t level>
__device__ nixl_status_t
putDeferred(const memViewElem &src,
            const memViewElem &dst,
            size_t bytes,
            uint64_t owner_token,
            xferStatusH *xfer_status) {
#if !NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE || !defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
    (void)src;
    (void)dst;
    (void)bytes;
    (void)owner_token;
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    if constexpr (level != level_t::THREAD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (xfer_status == nullptr || bytes == 0 ||
        bytes > static_cast<size_t>(DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto *src_view = static_cast<const GpunetioDeviceView *>(src.mvh);
    const auto *dst_view = static_cast<const GpunetioDeviceView *>(dst.mvh);
    if (src_view == nullptr || dst_view == nullptr ||
        src_view->abi_version != gpunetio_device_view_abi_version ||
        dst_view->abi_version != gpunetio_device_view_abi_version ||
        src_view->role != gpunetio_device_view_role_local ||
        dst_view->role != gpunetio_device_view_role_remote || src_view->context == nullptr ||
        src_view->context != dst_view->context || src_view->context_cookie == 0 ||
        src_view->context_cookie != dst_view->context_cookie ||
        src_view->execution_gpu != dst_view->execution_gpu || src_view->elems == nullptr ||
        dst_view->elems == nullptr || src.index >= src_view->count ||
        dst.index >= dst_view->count) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const auto &local = src_view->elems[src.index];
    const auto &remote = dst_view->elems[dst.index];
    uint64_t local_address;
    uint64_t remote_address;
    if (local.valid == 0 || remote.valid == 0 || local.peer_slot != gpunetio_local_peer_slot ||
        remote.peer_slot != gpunetio_remote_peer_slot ||
        !rangeIsValid(local.base, local.length, src.offset, bytes, &local_address) ||
        !rangeIsValid(remote.base, remote.length, dst.offset, bytes, &remote_address)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto *lane = src_view->context->lane;
    if (lane == nullptr || lane->owner_enabled == 0 || lane->qp == nullptr ||
        !claimOwnerToken(lane, owner_token)) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (lane->failed != 0) {
        return NIXL_ERR_BACKEND;
    }
    auto *qp = lane->qp;
    auto *cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
    if (lane->owner_wqe_pending >= qp->sq_wqe_num || lane->owner_cqe_pending >= cq->cqe_num ||
        lane->owner_wqe_pending >= 32 || lane->owner_generation == ~uint64_t{0}) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    const uint64_t ticket = doca_gpu_dev_verbs_reserve_wq_slots<owner_resource_sharing_mode,
                                                                DOCA_GPUNETIO_VERBS_QP_SQ,
                                                                DOCA_GPUNETIO_VERBS_CQ_64B>(
        qp, 1, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK);
    auto *wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
    doca_gpu_dev_verbs_wqe_prepare_write(qp,
                                         wqe,
                                         static_cast<uint16_t>(ticket),
                                         DOCA_GPUNETIO_MLX5_OPCODE_RDMA_WRITE,
                                         DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE,
                                         0,
                                         remote_address,
                                         remote.key,
                                         local_address,
                                         local.key,
                                         static_cast<uint32_t>(bytes));
    if (lane->owner_staged_count == 0) {
        lane->owner_staged_first = ticket;
    }
    lane->owner_staged_last = ticket;
    ++lane->owner_staged_count;
    ++lane->owner_wqe_pending;
    ++lane->owner_cqe_pending;
    const uint64_t generation = ++lane->owner_generation;
    auto *payload = statusPayload(xfer_status);
    payload->lane = lane;
    payload->generation = generation;
    payload->ticket = ticket;
    payload->owner_token = owner_token;
    payload->cached_status = NIXL_IN_PROG;
    payload->initialized = 1;
    return NIXL_IN_PROG;
#endif
}

__device__ nixl_status_t
commit(const memViewElem &owner_view, uint64_t owner_token) {
#if !NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE || !defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
    (void)owner_view;
    (void)owner_token;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    const auto *view = static_cast<const GpunetioDeviceView *>(owner_view.mvh);
    if (view == nullptr || view->abi_version != gpunetio_device_view_abi_version ||
        view->context_cookie == 0 ||
        (view->role != gpunetio_device_view_role_local &&
         view->role != gpunetio_device_view_role_remote) ||
        owner_view.index >= view->count || view->elems == nullptr || view->context == nullptr ||
        view->context->lane == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *lane = view->context->lane;
    if (lane->owner_enabled == 0 || !matchesOwnerLease(lane, owner_token)) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (lane->failed != 0) {
        return NIXL_ERR_BACKEND;
    }
    if (lane->owner_staged_count == 0) {
        return NIXL_SUCCESS;
    }
    compat::markReadyAndSubmitOwner<owner_resource_sharing_mode>(
        lane->qp, lane->owner_staged_first, lane->owner_staged_last);
    lane->owner_committed_generation = lane->owner_generation;
    lane->owner_staged_count = 0;
    return NIXL_SUCCESS;
#endif
}

__device__ nixl_status_t
releaseOwner(const memViewElem &owner_view, uint64_t owner_token) {
#if !NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE || !defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
    (void)owner_view;
    (void)owner_token;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    const auto *view = static_cast<const GpunetioDeviceView *>(owner_view.mvh);
    if (view == nullptr || view->abi_version != gpunetio_device_view_abi_version ||
        view->context_cookie == 0 ||
        (view->role != gpunetio_device_view_role_local &&
         view->role != gpunetio_device_view_role_remote) ||
        owner_view.index >= view->count || view->elems == nullptr || view->context == nullptr ||
        view->context->lane == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *lane = view->context->lane;
    if (lane->owner_enabled == 0 || !matchesOwnerLease(lane, owner_token)) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (lane->failed != 0) {
        return NIXL_ERR_BACKEND;
    }
    if (lane->owner_staged_count != 0 || lane->owner_wqe_pending != 0 ||
        lane->owner_cqe_pending != 0 ||
        lane->owner_committed_generation != lane->owner_generation ||
        lane->owner_retired_generation != lane->owner_generation) {
        return NIXL_ERR_NOT_ALLOWED;
    }

    lane->owner_expected_token = owner_token == ~uint64_t{0} ? 0 : owner_token + 1;
    // This release store publishes the next lease sequence and is the old
    // lease's final lane access. A later lease may use this grid or another.
    deviceAtomic(lane->owner_claim).store(0, cuda::memory_order_release);
    return NIXL_SUCCESS;
#endif
}

template<level_t level>
__device__ nixl_status_t
ownerGetXferStatus(xferStatusH &xfer_status, uint64_t owner_token) {
#if !NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE || !defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
    (void)xfer_status;
    (void)owner_token;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    if constexpr (level != level_t::THREAD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    auto *payload = statusPayload(&xfer_status);
    if (payload->initialized == 0 || payload->lane == nullptr || owner_token == 0 ||
        payload->owner_token != owner_token) {
        return NIXL_ERR_INVALID_PARAM;
    }
    auto *lane = payload->lane;
    if (lane->owner_enabled == 0 || !matchesOwnerLease(lane, owner_token)) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (payload->cached_status != NIXL_IN_PROG) {
        return static_cast<nixl_status_t>(payload->cached_status);
    }
    if (lane->failed != 0) {
        payload->cached_status = NIXL_ERR_BACKEND;
        return NIXL_ERR_BACKEND;
    }
    if (payload->generation > lane->owner_committed_generation) {
        return NIXL_IN_PROG;
    }
    if (payload->generation <= lane->owner_retired_generation) {
        payload->cached_status = NIXL_SUCCESS;
        return NIXL_SUCCESS;
    }
    const int completion =
        compat::pollOneOwner<owner_resource_sharing_mode>(lane->qp, payload->ticket);
    if (completion == EBUSY) {
        return NIXL_IN_PROG;
    }
    if (completion != 0) {
        lane->failed = 1;
        payload->cached_status = NIXL_ERR_BACKEND;
        return NIXL_ERR_BACKEND;
    }
    const uint64_t retired = payload->generation - lane->owner_retired_generation;
    lane->owner_retired_generation = payload->generation;
    lane->owner_wqe_pending -= static_cast<uint32_t>(retired);
    lane->owner_cqe_pending -= static_cast<uint32_t>(retired);
    payload->cached_status = NIXL_SUCCESS;
    return NIXL_SUCCESS;
#endif
}

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
#if !NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE
    (void)xfer_status;
    return NIXL_ERR_NOT_SUPPORTED;
#else
    if constexpr (level != level_t::THREAD) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    auto *payload = statusPayload(&xfer_status);
    if (payload->initialized == 0 || payload->lane == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (payload->lane->owner_enabled != 0) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    if (payload->cached_status != NIXL_IN_PROG) {
        return static_cast<nixl_status_t>(payload->cached_status);
    }

    auto *lane = payload->lane;
    if (deviceAtomic(lane->phase).load(cuda::memory_order_acquire) != gpunetio_lane_phase_posted ||
        deviceAtomic(lane->generation).load(cuda::memory_order_relaxed) != payload->generation ||
        deviceAtomic(lane->active_ticket).load(cuda::memory_order_relaxed) != payload->ticket) {
        failLane(payload);
        return NIXL_ERR_BACKEND;
    }

    const int completion = compat::pollOne(lane->qp, payload->ticket);
    if (completion == EBUSY) {
        return NIXL_IN_PROG;
    }
    if (completion != 0) {
        failLane(payload);
        return NIXL_ERR_BACKEND;
    }

    payload->cached_status = NIXL_SUCCESS;
    deviceAtomic(lane->last_success_generation)
        .store(payload->generation, cuda::memory_order_relaxed);
    // IDLE is the last release: a new submitter may now replace active_ticket.
    deviceAtomic(lane->phase).store(gpunetio_lane_phase_idle, cuda::memory_order_release);
    return NIXL_SUCCESS;
#endif
}

__device__ inline void *
getPtr(void *, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::impl::gpunetio

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_GPUNETIO_DEVICE_GPUNETIO_CUH
