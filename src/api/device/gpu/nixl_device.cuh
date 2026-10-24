/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Generic entry point for the GPU Device API.

#ifndef NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
#define NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH

#include <gpu/impl/device_dispatch.cuh>

namespace nixl::gpu {

template<level_t level = level_t::THREAD>
__device__ nixl_status_t
getXferStatus(xferStatusH &xfer_status) {
    return impl::getXferStatus<level>(xfer_status);
}

template<level_t level = level_t::THREAD>
__device__ nixl_status_t
put(const memViewElem &src,
    const memViewElem &dst,
    size_t size,
    unsigned channel_id = 0,
    uint64_t flags = 0,
    xferStatusH *xfer_status = nullptr) {
    return impl::put<level>(src, dst, size, channel_id, flags, xfer_status);
}

/**
 * @brief Stage one bounded GPUNETIO owner PUT without ringing the doorbell.
 *
 * Available only for a GPUNETIO engine configured with native_device_owner=true.
 * A per-engine token sequence starts at 1 and increases by one after each
 * released lease. The active token binds the lane to one CUDA grid; pass it
 * for stage, commit, poll, and release. Closed tokens cannot be reused.
 * The caller must keep the stream and views alive, synchronize before release,
 * and retire accepted work; NIXL does not track caller stream lifetime.
 */
template<level_t level = level_t::THREAD>
__device__ nixl_status_t
putDeferred(const memViewElem &src,
            const memViewElem &dst,
            size_t size,
            uint64_t owner_token,
            xferStatusH *xfer_status) {
    return impl::putDeferred<level>(src, dst, size, owner_token, xfer_status);
}

/** @brief Commit all deferred PUTs staged by the owner token on this view. */
__device__ inline nixl_status_t
commit(const memViewElem &owner_view, uint64_t owner_token) {
    return impl::commit(owner_view, owner_token);
}

/**
 * @brief Release a fully drained owner lease; this must be the actor's final
 * NIXL device API access for this grid.
 */
__device__ inline nixl_status_t
releaseOwner(const memViewElem &owner_view, uint64_t owner_token) {
    return impl::releaseOwner(owner_view, owner_token);
}

/** @brief Poll a deferred owner PUT using the same explicit owner token. */
template<level_t level = level_t::THREAD>
__device__ nixl_status_t
ownerGetXferStatus(xferStatusH &xfer_status, uint64_t owner_token) {
    return impl::ownerGetXferStatus<level>(xfer_status, owner_token);
}

template<level_t level = level_t::THREAD>
__device__ nixl_status_t
atomicAdd(uint64_t value,
          const memViewElem &counter,
          unsigned channel_id = 0,
          uint64_t flags = 0,
          xferStatusH *xfer_status = nullptr) {
    return impl::atomicAdd<level>(value, counter, channel_id, flags, xfer_status);
}

__device__ inline void *
getPtr(nixlMemViewH mvh, size_t index) {
    return impl::getPtr(mvh, index);
}

} // namespace nixl::gpu

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuGetXferStatus(nixlGpuXferStatusH &xfer_status) {
    return nixl::gpu::getXferStatus<level>(xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlPut(const nixlMemViewElem &src,
        const nixlMemViewElem &dst,
        size_t size,
        unsigned channel_id = 0,
        uint64_t flags = 0,
        nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::put<level>(src, dst, size, channel_id, flags, xfer_status);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlPutDeferred(const nixlMemViewElem &src,
                const nixlMemViewElem &dst,
                size_t size,
                uint64_t owner_token,
                nixlGpuXferStatusH *xfer_status) {
    return nixl::gpu::putDeferred<level>(src, dst, size, owner_token, xfer_status);
}

__device__ inline nixl_status_t
nixlGpuCommit(const nixlMemViewElem &owner_view, uint64_t owner_token) {
    return nixl::gpu::commit(owner_view, owner_token);
}

__device__ inline nixl_status_t
nixlGpuReleaseOwner(const nixlMemViewElem &owner_view, uint64_t owner_token) {
    return nixl::gpu::releaseOwner(owner_view, owner_token);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlGpuOwnerGetXferStatus(nixlGpuXferStatusH &xfer_status, uint64_t owner_token) {
    return nixl::gpu::ownerGetXferStatus<level>(xfer_status, owner_token);
}

template<nixl_gpu_level_t level = nixl_gpu_level_t::THREAD>
__device__ nixl_status_t
nixlAtomicAdd(uint64_t value,
              const nixlMemViewElem &counter,
              unsigned channel_id = 0,
              uint64_t flags = 0,
              nixlGpuXferStatusH *xfer_status = nullptr) {
    return nixl::gpu::atomicAdd<level>(value, counter, channel_id, flags, xfer_status);
}

__device__ inline void *
nixlGetPtr(nixlMemViewH mvh, size_t index) {
    return nixl::gpu::getPtr(mvh, index);
}

#endif // NIXL_SRC_API_DEVICE_GPU_NIXL_DEVICE_CUH
