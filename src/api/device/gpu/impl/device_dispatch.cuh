/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#ifndef NIXL_SRC_API_DEVICE_GPU_IMPL_DEVICE_DISPATCH_CUH
#define NIXL_SRC_API_DEVICE_GPU_IMPL_DEVICE_DISPATCH_CUH

#include <gpu/device_types.cuh>

#if !defined(NIXL_DISABLE_UCX_DEVICE_API) && \
    (defined(HAVE_UCX_GPU_DEVICE_API) || __has_include(<ucp/api/device/ucp_device_impl.h>))
#include <gpu/impl/ucx/device_ucx.cuh>
#else
namespace nixl::gpu::impl::ucx {

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &, const memViewElem &, size_t, unsigned, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
putDeferred(const memViewElem &, const memViewElem &, size_t, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline nixl_status_t
commit(const memViewElem &, uint64_t) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline nixl_status_t
releaseOwner(const memViewElem &, uint64_t) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
ownerGetXferStatus(xferStatusH &, uint64_t) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t, const memViewElem &, unsigned, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline void *
getPtr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::impl::ucx
#endif

#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API)
#include <gpu/impl/gpunetio/device_gpunetio.cuh>
#else
namespace nixl::gpu::impl::gpunetio {

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &, const memViewElem &, size_t, unsigned, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
putDeferred(const memViewElem &, const memViewElem &, size_t, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline nixl_status_t
commit(const memViewElem &, uint64_t) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline nixl_status_t
releaseOwner(const memViewElem &, uint64_t) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
ownerGetXferStatus(xferStatusH &, uint64_t) {
    return NIXL_ERR_NOT_SUPPORTED;
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t, const memViewElem &, unsigned, uint64_t, xferStatusH *) {
    return NIXL_ERR_NOT_SUPPORTED;
}

__device__ inline void *
getPtr(nixlMemViewH, size_t) {
    return nullptr;
}

} // namespace nixl::gpu::impl::gpunetio
#endif

namespace nixl::gpu::impl {

__device__ inline uint32_t
statusModeTag(const xferStatusH &xfer_status) noexcept {
    return *reinterpret_cast<const uint32_t *>(xfer_status.storage + xfer_status_mode_offset);
}

__device__ inline void
setStatusModeTag(xferStatusH &xfer_status, nixl_device_exec_mode_t mode) noexcept {
    *reinterpret_cast<uint32_t *>(xfer_status.storage + xfer_status_mode_offset) =
        static_cast<uint32_t>(mode);
}

__device__ inline bool
isKnownMode(nixl_device_exec_mode_t mode) noexcept {
    return mode == nixl_device_exec_mode_t::UCX_DIRECT ||
        mode == nixl_device_exec_mode_t::GPUNETIO_DIRECT;
}

__device__ inline const nixlDeviceMemViewWrapper *
viewWrapper(nixlMemViewH mvh) noexcept {
    return reinterpret_cast<const nixlDeviceMemViewWrapper *>(mvh);
}

__device__ inline bool
unwrapViews(const memViewElem &src,
            const memViewElem &dst,
            const nixlDeviceMemViewWrapper *&src_wrapper,
            const nixlDeviceMemViewWrapper *&dst_wrapper,
            nixl_device_exec_mode_t &mode) noexcept {
    if (src.mvh == nullptr || dst.mvh == nullptr) {
        return false;
    }

    src_wrapper = viewWrapper(src.mvh);
    dst_wrapper = viewWrapper(dst.mvh);
    if (!isKnownMode(src_wrapper->execution_mode) ||
        src_wrapper->execution_mode != dst_wrapper->execution_mode ||
        src_wrapper->backend_memview == nullptr || dst_wrapper->backend_memview == nullptr) {
        return false;
    }

    mode = src_wrapper->execution_mode;
    return true;
}

__device__ inline bool
prepareStatus(const xferStatusH *xfer_status, nixl_device_exec_mode_t mode) noexcept {
    if (xfer_status == nullptr) {
        return false;
    }
    const uint32_t current = statusModeTag(*xfer_status);
    return current == 0 || current == static_cast<uint32_t>(mode);
}

template<level_t level>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    switch (statusModeTag(xfer_status)) {
    case static_cast<uint32_t>(nixl_device_exec_mode_t::UCX_DIRECT):
        return ucx::getXferStatus<level>(xfer_status);
    case static_cast<uint32_t>(nixl_device_exec_mode_t::GPUNETIO_DIRECT):
        return gpunetio::getXferStatus<level>(xfer_status);
    default:
        return NIXL_ERR_INVALID_PARAM;
    }
}

template<level_t level>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id,
    uint64_t flags,
    xferStatusH *xfer_status) {
    const nixlDeviceMemViewWrapper *src_wrapper = nullptr;
    const nixlDeviceMemViewWrapper *dst_wrapper = nullptr;
    nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::NONE;
    if (!unwrapViews(src, dst, src_wrapper, dst_wrapper, mode) ||
        !prepareStatus(xfer_status, mode)) {
        return NIXL_ERR_INVALID_PARAM;
    }

#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API)
    if (mode == nixl_device_exec_mode_t::GPUNETIO_DIRECT &&
        gpunetio::isStatusInProgress(*xfer_status)) {
        return NIXL_ERR_REPOST_ACTIVE;
    }
#endif

    const memViewElem backend_src{src_wrapper->backend_memview, src.index, src.offset};
    const memViewElem backend_dst{dst_wrapper->backend_memview, dst.index, dst.offset};
    nixl_status_t status = NIXL_ERR_NOT_SUPPORTED;
    if (mode == nixl_device_exec_mode_t::UCX_DIRECT) {
        status = ucx::put<level>(backend_src, backend_dst, size, channel_id, flags, xfer_status);
    } else if (mode == nixl_device_exec_mode_t::GPUNETIO_DIRECT) {
        status =
            gpunetio::put<level>(backend_src, backend_dst, size, channel_id, flags, xfer_status);
    }

    if (status == NIXL_IN_PROG) {
        setStatusModeTag(*xfer_status, mode);
    }
    return status;
}

template<level_t level>
__device__ nixl_status_t
putDeferred(const memViewElem &src,
            const memViewElem &dst,
            size_t bytes,
            uint64_t owner_token,
            xferStatusH *xfer_status) {
    const nixlDeviceMemViewWrapper *src_wrapper = nullptr;
    const nixlDeviceMemViewWrapper *dst_wrapper = nullptr;
    nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::NONE;
    if (!unwrapViews(src, dst, src_wrapper, dst_wrapper, mode) ||
        !prepareStatus(xfer_status, mode)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (mode != nixl_device_exec_mode_t::GPUNETIO_DIRECT) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API)
    if (gpunetio::isStatusInProgress(*xfer_status)) {
        return NIXL_ERR_REPOST_ACTIVE;
    }
#endif
    const memViewElem backend_src{src_wrapper->backend_memview, src.index, src.offset};
    const memViewElem backend_dst{dst_wrapper->backend_memview, dst.index, dst.offset};
    const nixl_status_t status =
        gpunetio::putDeferred<level>(backend_src, backend_dst, bytes, owner_token, xfer_status);
    if (status == NIXL_IN_PROG) {
        setStatusModeTag(*xfer_status, mode);
    }
    return status;
}

__device__ inline nixl_status_t
putDeferredWarp(const memViewElem &src,
                const memViewElem &dst,
                size_t bytes,
                uint64_t owner_token,
                xferStatusH *statuses,
                bool active) {
#if defined(NIXL_ENABLE_GPUNETIO_DEVICE_API) && NIXL_GPUNETIO_DEVICE_API_HAS_SDK_PROFILE && \
    defined(NIXL_GPUNETIO_DEVICE_API_PUBLIC)
    constexpr uint32_t mask = 0xffffffffU;
    if (blockIdx.x || blockIdx.y || blockIdx.z || blockDim.y != 1 || blockDim.z != 1 ||
        threadIdx.x >= 32) {
        return NIXL_ERR_NOT_ALLOWED;
    }
    if (__activemask() != mask) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const uint64_t token = __shfl_sync(mask, static_cast<unsigned long long>(owner_token), 0);
    const auto status_base = reinterpret_cast<unsigned long long>(statuses);
    const auto common_status_base = __shfl_sync(mask, status_base, 0);
    // CUDA 13.0 can miscompile isspacep intrinsics after merging typed pointers
    // from different address spaces. Query the generic address directly.
    uint32_t owner_visible;
    asm("{ .reg .pred g, s; isspacep.global g, %1; isspacep.shared s, %1; "
        "or.pred g, g, s; selp.u32 %0, 1, 0, g; }"
        : "=r"(owner_visible)
        : "l"(status_base));
    int error = NIXL_SUCCESS;
    if (owner_token == 0 || owner_token != token || statuses == nullptr ||
        status_base != common_status_base || !owner_visible) {
        error = NIXL_ERR_INVALID_PARAM;
    }
    const nixlDeviceMemViewWrapper *s = nullptr, *d = nullptr;
    nixl_device_exec_mode_t mode = nixl_device_exec_mode_t::NONE;
    memViewElem backend_src{}, backend_dst{};
    if (active && error == NIXL_SUCCESS) {
        if (!unwrapViews(src, dst, s, d, mode) || !prepareStatus(&statuses[threadIdx.x], mode)) {
            error = NIXL_ERR_INVALID_PARAM;
        } else if (mode != nixl_device_exec_mode_t::GPUNETIO_DIRECT) {
            error = NIXL_ERR_NOT_SUPPORTED;
        } else if (statusModeTag(statuses[threadIdx.x]) == static_cast<uint32_t>(mode) &&
                   gpunetio::isStatusInProgress(statuses[threadIdx.x])) {
            error = NIXL_ERR_REPOST_ACTIVE;
        } else {
            backend_src = {s->backend_memview, src.index, src.offset};
            backend_dst = {d->backend_memview, dst.index, dst.offset};
        }
    }
    const uint32_t invalid = __ballot_sync(mask, error != NIXL_SUCCESS);
    if (invalid) {
        return static_cast<nixl_status_t>(__shfl_sync(mask, error, __ffs(invalid) - 1));
    }
    const nixl_status_t result =
        gpunetio::putDeferredWarp(backend_src, backend_dst, bytes, owner_token, statuses, active);
    if (result == NIXL_IN_PROG && active) {
        setStatusModeTag(statuses[threadIdx.x], nixl_device_exec_mode_t::GPUNETIO_DIRECT);
    }
    __syncwarp(mask);
    return result;
#else
    (void)src;
    (void)dst;
    (void)bytes;
    (void)owner_token;
    (void)statuses;
    (void)active;
    return NIXL_ERR_NOT_SUPPORTED;
#endif
}

__device__ inline nixl_status_t
commit(const memViewElem &owner_view, uint64_t owner_token) {
    if (owner_view.mvh == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const nixlDeviceMemViewWrapper *wrapper = viewWrapper(owner_view.mvh);
    if (!isKnownMode(wrapper->execution_mode) || wrapper->backend_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (wrapper->execution_mode != nixl_device_exec_mode_t::GPUNETIO_DIRECT) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const memViewElem backend_view{wrapper->backend_memview, owner_view.index, owner_view.offset};
    return gpunetio::commit(backend_view, owner_token);
}

__device__ inline nixl_status_t
releaseOwner(const memViewElem &owner_view, uint64_t owner_token) {
    if (owner_view.mvh == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    const nixlDeviceMemViewWrapper *wrapper = viewWrapper(owner_view.mvh);
    if (!isKnownMode(wrapper->execution_mode) || wrapper->backend_memview == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    if (wrapper->execution_mode != nixl_device_exec_mode_t::GPUNETIO_DIRECT) {
        return NIXL_ERR_NOT_SUPPORTED;
    }
    const memViewElem backend_view{wrapper->backend_memview, owner_view.index, owner_view.offset};
    return gpunetio::releaseOwner(backend_view, owner_token);
}

template<level_t level>
__device__ nixl_status_t
ownerGetXferStatus(xferStatusH &xfer_status, uint64_t owner_token) {
    if (statusModeTag(xfer_status) !=
        static_cast<uint32_t>(nixl_device_exec_mode_t::GPUNETIO_DIRECT)) {
        return NIXL_ERR_INVALID_PARAM;
    }
    return gpunetio::ownerGetXferStatus<level>(xfer_status, owner_token);
}

template<level_t level>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id,
          uint64_t flags,
          xferStatusH *xfer_status) {
    if (counter.mvh == nullptr || xfer_status == nullptr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    const nixlDeviceMemViewWrapper *wrapper = viewWrapper(counter.mvh);
    if (!isKnownMode(wrapper->execution_mode) || wrapper->backend_memview == nullptr ||
        !prepareStatus(xfer_status, wrapper->execution_mode)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    if (wrapper->execution_mode == nixl_device_exec_mode_t::GPUNETIO_DIRECT) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    const memViewElem backend_counter{wrapper->backend_memview, counter.index, counter.offset};
    const nixl_status_t status =
        ucx::atomicAdd<level>(value, backend_counter, channel_id, flags, xfer_status);
    if (status == NIXL_IN_PROG) {
        setStatusModeTag(*xfer_status, wrapper->execution_mode);
    }
    return status;
}

__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    if (mvh == nullptr) {
        return nullptr;
    }

    const nixlDeviceMemViewWrapper *wrapper = viewWrapper(mvh);
    if (!isKnownMode(wrapper->execution_mode) || wrapper->backend_memview == nullptr) {
        return nullptr;
    }
    if (wrapper->execution_mode == nixl_device_exec_mode_t::UCX_DIRECT) {
        return ucx::getPtr(wrapper->backend_memview, index);
    }
    return gpunetio::getPtr(wrapper->backend_memview, index);
}

} // namespace nixl::gpu::impl

#endif // NIXL_SRC_API_DEVICE_GPU_IMPL_DEVICE_DISPATCH_CUH
