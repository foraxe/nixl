// SPDX-License-Identifier: Apache-2.0
#include "local_copy.cuh"

// A consuming application kernel, not a separate posting/progress kernel.
__global__ void consume(local_copy::Copy *copy, const uint8_t *src, uint8_t *dst,
                        size_t bytes, int fragments, int window, int rounds,
                        int epoch, int mode, uint64_t *result) {
    __shared__ int status;
    __shared__ local_copy::Ticket tickets[32];
    if (threadIdx.x == 0) status = 0;
    __syncthreads();
    if (mode >= 57 && mode <= 59) {
        for (size_t i = threadIdx.x; i < copy->dst_size; i += blockDim.x) dst[i] = 0xcd;
        __syncthreads();
        uint64_t before = copy->qp->sq_rsvd_index;
        if (mode == 59 && threadIdx.x == 0) tickets[0] = copy->copy_async(dst + 128, src + 128, 1);
        __syncthreads();
        if (threadIdx.x < 32) {
            int lane = threadIdx.x;
            if (mode == 59) {
                auto refused = copy->copy_async_warp(dst + 256 + lane, src + 256 + lane, 1);
                if (lane == 0 && (refused.error != -EAGAIN ||
                    copy->qp->sq_rsvd_index != before + (copy->mcst ? 2 : 1))) result[0] = 1;
            }
            const bool active = mode == 57 ? (lane == 2 || lane == 7 || lane == 31) :
                                mode == 58 ? false : lane < (copy->mcst ? 29 : 31);
            auto t = copy->copy_async_warp(
                mode == 57 ? dst + 128 + lane * 64 : dst + 256 + lane,
                mode == 57 ? src + 128 + lane * 64 : src + 256 + lane,
                mode == 57 ? 64 : 1, active);
            if (lane == 0) {
                if (mode == 58) result[0] = t.error != -EINVAL || before != copy->qp->sq_rsvd_index;
                else status = copy->wait(t);
            }
        }
        __syncthreads();
        if (mode != 58 && status == 0)
            for (size_t i = threadIdx.x; i < copy->dst_size; i += blockDim.x) {
                bool payload = mode == 57 ? (i >= 128 && i < 128 + 32*64 &&
                    (((i-128)/64) == 2 || ((i-128)/64) == 7 || ((i-128)/64) == 31)) :
                    (i == 128 || (i >= 256 && i < 256 + (copy->mcst ? 29 : 31)));
                uint8_t expected = payload ? (i * 17 + epoch * 31) % 251 : 0xcd;
                if (dst[i] != expected) atomicAdd(reinterpret_cast<unsigned long long *>(result), 1ULL);
            }
        if (threadIdx.x == 0) {
            result[1] = static_cast<int64_t>(status);
            result[2] = copy->qp->sq_rsvd_index; result[3] = copy->retired;
        }
        return;
    }
    if (mode == 11) {
        if (threadIdx.x == 0) result[0] = copy->recover(-EIO);
        return;
    }
    if (mode == 10) {
        if (threadIdx.x == 0) {
            auto a = copy->copy_async(dst + 128, src + 128, 64);
            const auto key = copy->rkey;
            copy->rkey ^= 0x01000000;
            auto b = copy->copy_async(dst + 256, src + 256, 64);
            copy->rkey = key;
            auto c = copy->copy_async(dst + 384, src + 384, 64);
            const int failure = copy->wait(c);
            const int again = copy->wait(c);
            auto before = copy->qp->sq_rsvd_index;
            auto refused = copy->copy_async(dst, src, 1);
            result[0] = a.error || b.error || c.error || failure != -EIO || again != failure ||
                refused.error != -EIO || before != copy->qp->sq_rsvd_index;
            result[1] = failure;
            result[2] = copy->recovery->stopped;
            for (size_t i = 0; i < copy->dst_size; ++i) dst[i] = 0xa7;
            __threadfence_system();
            const auto until = clock64() + 100000000ULL;
            while (clock64() < until)
                for (size_t i = 0; i < copy->dst_size; ++i)
                    if (reinterpret_cast<volatile uint8_t *>(dst)[i] != 0xa7) result[0] = 1;
        }
        return;
    }
    if (mode == 5) {
        if (threadIdx.x == 0) {
            const int capacity = copy->mcst ? 16 : 32;
            for (int i = 0; i < capacity; ++i) {
                tickets[i] = copy->copy_async(dst + 128 + i, src + 128 + i, 1);
                if (tickets[i].error) result[0] = 1;
            }
            auto before = copy->qp->sq_rsvd_index;
            auto rejected = copy->copy_async(dst, src, 1);
            if (rejected.error != -EAGAIN || before != copy->qp->sq_rsvd_index) result[0] = 1;
            result[1] = copy->wait(tickets[capacity - 1]);
            result[2] = capacity;
            auto stale = tickets[capacity - 1];
            auto next = copy->copy_async(dst, src, 1);
            if (copy->wait(stale) != -EINVAL) result[0] = 1;
            auto foreign = next; ++foreign.cookie;
            if (copy->wait(foreign) != -EINVAL) result[0] = 1;
            if (copy->wait(next)) result[0] = 1;
        }
        return;
    }
    if (mode == 6) {
        if (threadIdx.x == 1) {
            const auto t = copy->copy_async(dst, src, 1);
            result[0] = t.error == -EPERM ? 0 : 1;
        }
        return;
    }
    if (mode == 30) {
        if (threadIdx.x == 0) {
            auto t = copy->copy_async(dst, src, min(bytes, size_t(1048576)));
            result[0] = t.error != 0;
        }
        return;
    }
    if (mode >= 53 && mode <= 56) {
        if (mode == 53 && threadIdx.x == 0) copy->mcst = true;
        __syncthreads();
        uint64_t before = copy->qp->sq_rsvd_index;
        if (mode == 55) {
            if (threadIdx.x == 0) {
                auto t = copy->copy_async_warp(dst, src, 1);
                result[0] = t.error != -EINVAL || before != copy->qp->sq_rsvd_index;
            }
        } else if (threadIdx.x < 32) {
            local_copy::Copy clone = *copy;
            auto *chosen = mode == 56 && threadIdx.x == 7 ? &clone : copy;
            auto t = chosen->copy_async_warp(dst + 128 + threadIdx.x,
                mode == 54 && threadIdx.x == 7 ? src + copy->src_size : src + 128 + threadIdx.x, 1);
            if (threadIdx.x == 0)
                result[0] = t.error != (mode == 53 ? -EAGAIN : -EINVAL) ||
                    before != copy->qp->sq_rsvd_index;
        }
        return;
    }
    const bool warp_path = mode >= 50 && mode <= 52;
    if ((mode == 7 || mode == 51) && threadIdx.x == 0) copy->mcst = true;
    if (mode == 52 && threadIdx.x == 0) copy->rkey ^= 0x01000000;
    __syncthreads();
    if (mode > 0 && mode != 7 && !warp_path) {
        if (threadIdx.x == 0) {
            auto before = copy->qp->sq_rsvd_index;
            auto t = mode == 1 ? copy->copy_async(dst, src + copy->src_size, 1) :
                     mode == 2 ? copy->copy_async(dst + copy->dst_size, src, 1) :
                     mode == 3 ? copy->copy_async(dst, src, 0) :
                                 copy->copy_async(dst, src, SIZE_MAX);
            result[0] = t.error == -EINVAL && copy->qp->sq_rsvd_index == before ? 0 : 1;
        }
        return;
    }
    uint64_t independent = threadIdx.x + 1;
    for (int r = 0; r < rounds; ++r) {
        if (mode >= 0)
            for (size_t i = threadIdx.x; i < copy->dst_size; i += blockDim.x) dst[i] = 0xcd;
        __syncthreads();
        for (int start = 0; start < fragments && status == 0; start += window) {
            const int count = min(window, fragments - start);
            if (warp_path && threadIdx.x < 32 && status == 0) {
                const int j = start + threadIdx.x;
                const bool active = threadIdx.x < count;
                const int source = (j * 13) % fragments;
                auto t = copy->copy_async_warp(active ? dst + 128 + j * (bytes + 32) : dst,
                    active ? src + 128 + source * (bytes + 17) : src, bytes, active);
                if (threadIdx.x == 0) { tickets[0] = t; status = t.error; }
            } else if (!warp_path && threadIdx.x == 0 && mode != -1) {
                for (int w = 0; w < count; ++w) {
                    const int j = start + w;
                    const int source = (j * 13) % fragments;
                    tickets[w] = copy->copy_async(dst + 128 + j * (bytes + 32),
                        src + 128 + source * (bytes + 17), bytes);
                    if (tickets[w].error) { status = tickets[w].error; break; }
                }
            }
            // Useful independent arithmetic is between issue and wait.
            for (int k = 0; k < 64; ++k) independent = independent * 1664525 + 1013904223;
            if (threadIdx.x == 0 && status == 0 && mode != -1)
                status = copy->wait(tickets[warp_path ? 0 : count - 1]);
            __syncthreads();
        }
        if (status != 0) break;
        for (size_t i = threadIdx.x; i < copy->dst_size; i += blockDim.x) {
            uint8_t expected = 0xcd;
            if (i >= 128) {
                const size_t j = (i - 128) / (bytes + 32);
                const size_t off = (i - 128) % (bytes + 32);
                if (j < static_cast<size_t>(fragments) && off < bytes) {
                    const size_t source = 128 + ((j * 13) % fragments) * (bytes + 17) + off;
                    expected = (source * 17 + epoch * 31) % 251;
                }
            }
            if (dst[i] != expected) atomicAdd(reinterpret_cast<unsigned long long *>(result), 1ULL);
        }
        __syncthreads();
    }
    if (mode == 52 && threadIdx.x < 32) {
        auto t = copy->copy_async_warp(dst, src, 1, threadIdx.x == 0);
        if (threadIdx.x == 0) { result[9] = static_cast<int64_t>(t.error); result[10] = copy->recovery->stopped; }
    }
    if (threadIdx.x == 0) {
        result[1] = static_cast<int64_t>(status);
        result[2] = copy->qp->sq_rsvd_index;
        result[3] = copy->retired;
        result[4] = copy->mcst;
    }
    atomicAdd(reinterpret_cast<unsigned long long *>(result + 5), independent);
}

extern "C" int lc_probe(void *device, void *src, void *dst, size_t bytes, int fragments,
                         int window, int rounds, int epoch, int mode, void *result, void *stream) {
    consume<<<1, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<local_copy::Copy *>(device), static_cast<uint8_t *>(src),
        static_cast<uint8_t *>(dst), bytes, fragments, window, rounds, epoch, mode,
        static_cast<uint64_t *>(result));
    return cudaGetLastError();
}

__global__ void hold_owner(local_copy::Copy *copy, void *dst, const void *src, uint64_t *r) {
    auto ready = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(r[0]);
    auto release = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(r[1]);
    const auto t = copy->copy_async(dst, src, 1);
    r[2] = static_cast<int64_t>(t.error);
    r[6] = local_copy::Copy::grid();
    r[8] = copy->owner;
    ready.store(1, cuda::memory_order_release);
    const auto deadline = clock64() + 2000000000ULL;
    while (!release.load(cuda::memory_order_acquire) && clock64() < deadline) __nanosleep(100);
    r[10] = release.load(cuda::memory_order_acquire);
    r[3] = static_cast<int64_t>(copy->wait(t));
}

__global__ void other_grid(local_copy::Copy *copy, void *dst, const void *src, uint64_t *r) {
    auto ready = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(r[0]);
    auto release = cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(r[1]);
    const auto deadline = clock64() + 2000000000ULL;
    while (!ready.load(cuda::memory_order_acquire) && clock64() < deadline) __nanosleep(100);
    r[7] = local_copy::Copy::grid();
    r[9] = copy->owner;
    r[11] = ready.load(cuda::memory_order_acquire);
    const auto t = copy->copy_async(dst, src, 1);
    r[4] = static_cast<int64_t>(t.error);
    release.store(1, cuda::memory_order_release);
}

extern "C" int lc_owner_probe(void *device, void *src, void *dst, void *result) {
    // Lazy function loading can synchronize the context before the second launch.
    cudaFuncAttributes attrs{};
    auto status = cudaFuncGetAttributes(&attrs, hold_owner);
    if (status != cudaSuccess) return status;
    status = cudaFuncGetAttributes(&attrs, other_grid);
    if (status != cudaSuccess) return status;
    cudaStream_t a = nullptr, b = nullptr;
    status = cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
    if (status != cudaSuccess) return status;
    status = cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
    if (status != cudaSuccess) { cudaStreamDestroy(a); return status; }
    auto *copy = static_cast<local_copy::Copy *>(device);
    auto *r = static_cast<uint64_t *>(result);
    hold_owner<<<1, 1, 0, a>>>(copy, dst, src, r);
    other_grid<<<1, 1, 0, b>>>(copy, dst, src, r);
    status = cudaStreamSynchronize(a);
    const auto second = cudaStreamSynchronize(b);
    cudaStreamDestroy(a); cudaStreamDestroy(b);
    return status == cudaSuccess ? second : status;
}
