// SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
// SPDX-License-Identifier: Apache-2.0
// Two-process public-API GPUNETIO native-owner qualification harness.
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>
#include <nixl.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/inotify.h>
#include <unistd.h>

namespace {
constexpr uint32_t kWqes = 64, kWindows[] = {1, 8, 32}, kWarmup = 200, kMeasured = 1000;
constexpr uint64_t kPayload = 4096, kStride = 4352, kGuard = 256;
constexpr uint64_t kMarker = kGuard + kWqes * kStride;
constexpr uint64_t kArena = kMarker + sizeof(uint64_t) + kGuard;
constexpr uint64_t kExpectedChecked = kArena - sizeof(uint64_t);
constexpr uint64_t kSeed = 0x91e10da5c79e7b1dull;
constexpr uint64_t kTimeoutCycles = 5000000000ull;
constexpr uint8_t kHole = 0xa5;

struct Result {
    uint64_t elapsed_ns;
    uint32_t precommit_checked;
    int status;
    int wrong_owner;
    int release;
    int capacity;
    int precommit_status;
};

struct FaultResult {
    int accepted;
    int commit;
    int completion;
    int later_post;
    int release;
};

struct Gate {
    uint32_t entered;
    uint32_t tested;
    int status;
};

struct Validation {
    uint64_t checked;
    uint64_t first_bad;
    uint64_t marker_expected;
    uint64_t marker_observed;
    uint32_t epoch;
    uint32_t expected_byte;
    uint32_t actual_byte;
    int status;
};

__host__ __device__ uint8_t
byteFor(uint64_t seed, uint64_t pos) {
    uint64_t x = seed + pos + 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return static_cast<uint8_t>((x ^ (x >> 31)) >> 8);
}

__device__ uint64_t
timerNs() {
    uint64_t n;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(n));
    return n;
}

// One fresh CTA/grid per batch; source generation, data PUTs, marker PUT, CQ
// retirement, and release are ordered in this caller-owned stream.
__global__ void
ownerBatch(nixlMemViewH local,
           nixlMemViewH remote,
           uint8_t *source,
           uint32_t epoch,
           uint32_t window,
           uint64_t token,
           Result *out,
           Gate *gate,
           bool probe,
           bool batch_commit,
           bool check_precommit,
           int prep_mode) {
    __shared__ __align__(16) unsigned char shared_bytes[32 * sizeof(nixlGpuXferStatusH)];
    __shared__ int group_status;
    if (blockIdx.x || threadIdx.x >= 256) {
        return;
    }
    for (uint32_t i = threadIdx.x; i < kWqes * kStride; i += blockDim.x) {
        const uint32_t item = i / kStride, offset = i % kStride;
        source[kGuard + i] =
            offset < kPayload ? byteFor(kSeed ^ (uint64_t(epoch) << 32) ^ item, offset) : kHole;
    }
    for (uint32_t i = threadIdx.x; i < sizeof(shared_bytes); i += blockDim.x) {
        shared_bytes[i] = 0;
    }
    if (threadIdx.x == 0) {
        *reinterpret_cast<uint64_t *>(source + kMarker) = (uint64_t(epoch) << 32) | 0x4e49584cull;
    }
    __syncthreads();
    if (threadIdx.x >= 32) {
        return;
    }
    const uint32_t lane = threadIdx.x;
    alignas(16) unsigned char local_bytes[32 * sizeof(nixlGpuXferStatusH)];
    if (lane == 0) {
        if (prep_mode == 0) {
            for (uint32_t i = 0; i < sizeof(local_bytes); ++i) {
                local_bytes[i] = 0;
            }
        }
        out->elapsed_ns = 0;
        out->precommit_checked = 0;
        out->precommit_status = NIXL_SUCCESS;
        out->status = NIXL_SUCCESS;
        out->wrong_owner = NIXL_SUCCESS;
        out->release = NIXL_IN_PROG;
        out->capacity = NIXL_SUCCESS;
    }
    auto *status =
        reinterpret_cast<nixlGpuXferStatusH *>(prep_mode == 0 ? local_bytes : shared_bytes);
    __syncwarp();
    const uint64_t begin = lane == 0 ? timerNs() : 0;
    for (uint32_t base = 0; base < kWqes; base += window) {
        if (prep_mode == 2) {
            const auto rc =
                nixlPutDeferredWarp({local, 0, kGuard + uint64_t(base + lane) * kStride},
                                    {remote, 0, kGuard + uint64_t(base + lane) * kStride},
                                    kPayload,
                                    token,
                                    status,
                                    lane < window);
            if (lane == 0) {
                group_status = rc;
            }
        } else if (lane == 0) {
            group_status = NIXL_IN_PROG;
            for (uint32_t j = 0; j < window; ++j) {
                auto rc = nixlPutDeferred<nixl_gpu_level_t::THREAD>(
                    {local, 0, kGuard + uint64_t(base + j) * kStride},
                    {remote, 0, kGuard + uint64_t(base + j) * kStride},
                    kPayload,
                    token,
                    &status[j]);
                if (rc != NIXL_IN_PROG) {
                    group_status = rc;
                    break;
                }
                if (!batch_commit) {
                    rc = nixlGpuCommit({local, 0, 0}, token);
                    if (rc != NIXL_SUCCESS) {
                        group_status = rc;
                        break;
                    }
                }
            }
        }
        __syncwarp();
        if (group_status != NIXL_IN_PROG) {
            if (lane == 0) {
                out->status = group_status;
            }
            return;
        }
        if (lane == 0) {
            if (batch_commit) {
                if (check_precommit) {
                    for (uint32_t j = 0; j < window; ++j) {
                        const auto rc =
                            nixlGpuOwnerGetXferStatus<nixl_gpu_level_t::THREAD>(status[j], token);
                        ++out->precommit_checked;
                        if (rc != NIXL_IN_PROG) {
                            out->precommit_status = rc;
                            out->status = rc == NIXL_SUCCESS ? NIXL_ERR_BACKEND : rc;
                            break;
                        }
                    }
                    if (out->status == NIXL_SUCCESS) {
                        out->precommit_status = NIXL_IN_PROG;
                    }
                }
                if (out->status == NIXL_SUCCESS) {
                    out->status = nixlGpuCommit({local, 0, 0}, token);
                }
            }
            if (out->status == NIXL_SUCCESS && base == 0 && window == 32) {
                nixlGpuXferStatusH rejected{};
                out->capacity = nixlPutDeferred<nixl_gpu_level_t::THREAD>(
                    {local, 0, kGuard}, {remote, 0, kGuard}, kPayload, token, &rejected);
            }
            if (out->status == NIXL_SUCCESS && base == 0 && probe) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> entered(gate->entered);
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tested(gate->tested);
                entered.store(1, cuda::memory_order_release);
                const uint64_t deadline = clock64() + kTimeoutCycles;
                while (!tested.load(cuda::memory_order_acquire) && clock64() < deadline) {
                    __nanosleep(64);
                }
                out->wrong_owner = gate->status;
                if (!tested.load(cuda::memory_order_acquire)) {
                    out->status = NIXL_ERR_BACKEND;
                }
            }
            for (uint32_t j = 0; j < window && out->status == NIXL_SUCCESS; ++j) {
                const uint64_t deadline = clock64() + kTimeoutCycles;
                nixl_status_t rc;
                do {
                    rc = nixlGpuOwnerGetXferStatus<nixl_gpu_level_t::THREAD>(status[j], token);
                } while (rc == NIXL_IN_PROG && clock64() < deadline);
                if (rc != NIXL_SUCCESS) {
                    out->status = rc == NIXL_IN_PROG ? NIXL_ERR_BACKEND : rc;
                }
            }
            group_status = out->status;
        }
        __syncwarp();
        if (group_status != NIXL_SUCCESS) {
            return;
        }
    }
    if (lane != 0) {
        return;
    }
    nixlGpuXferStatusH ready{};
    out->status = nixlPutDeferred<nixl_gpu_level_t::THREAD>(
        {local, 0, kMarker}, {remote, 0, kMarker}, sizeof(uint64_t), token, &ready);
    if (out->status != NIXL_IN_PROG) {
        return;
    }
    out->status = nixlGpuCommit({local, 0, kMarker}, token);
    if (out->status != NIXL_SUCCESS) {
        return;
    }
    const uint64_t deadline = clock64() + kTimeoutCycles;
    nixl_status_t rc;
    do {
        rc = nixlGpuOwnerGetXferStatus<nixl_gpu_level_t::THREAD>(ready, token);
    } while (rc == NIXL_IN_PROG && clock64() < deadline);
    if (rc != NIXL_SUCCESS) {
        out->status = rc == NIXL_IN_PROG ? NIXL_ERR_BACKEND : rc;
        return;
    }
    out->release = nixlGpuReleaseOwner({local, 0, kMarker}, token);
    out->elapsed_ns = timerNs() - begin;
}
__global__ void
wrongGrid(nixlMemViewH local, nixlMemViewH remote, uint64_t token, Gate *gate) {
    if (blockIdx.x || threadIdx.x) {
        return;
    }
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> entered(gate->entered);
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> tested(gate->tested);
    const uint64_t deadline = clock64() + kTimeoutCycles;
    while (!entered.load(cuda::memory_order_acquire) && clock64() < deadline) {
        __nanosleep(64);
    }
    nixlGpuXferStatusH status{};
    gate->status = entered.load(cuda::memory_order_acquire) ?
        nixlPutDeferred<nixl_gpu_level_t::THREAD>(
            {local, 0, 0}, {remote, 0, 0}, 8, token, &status) :
        NIXL_ERR_BACKEND;
    tested.store(1, cuda::memory_order_release);
}

__global__ void
staleToken(nixlMemViewH local, nixlMemViewH remote, uint64_t old_token, Gate *gate) {
    if (blockIdx.x || threadIdx.x) {
        return;
    }
    nixlGpuXferStatusH status{};
    gate->status = nixlPutDeferred<nixl_gpu_level_t::THREAD>(
        {local, 0, 0}, {remote, 0, 0}, 8, old_token, &status);
}

__global__ void
wrongGridWarp(nixlMemViewH local, nixlMemViewH remote, uint64_t token, Gate *gate) {
    __shared__ __align__(16) unsigned char bytes[32 * sizeof(nixlGpuXferStatusH)];
    __shared__ uint32_t entered;
    for (uint32_t i = threadIdx.x; i < sizeof(bytes); i += 32) {
        bytes[i] = 0;
    }
    if (threadIdx.x == 0) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> flag(gate->entered);
        const uint64_t deadline = clock64() + kTimeoutCycles;
        while (!flag.load(cuda::memory_order_acquire) && clock64() < deadline) {
            __nanosleep(64);
        }
        entered = flag.load(cuda::memory_order_acquire);
    }
    __syncwarp();
    if (!entered) {
        if (threadIdx.x == 0) {
            gate->status = NIXL_ERR_BACKEND;
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(gate->tested)
                .store(1, cuda::memory_order_release);
        }
        return;
    }
    auto *statuses = reinterpret_cast<nixlGpuXferStatusH *>(bytes);
    const auto rc = nixlPutDeferredWarp({local, 0, 0}, {remote, 0, 0}, 8, token, statuses, true);
    bool clear = true;
    for (uint32_t i = threadIdx.x; i < sizeof(bytes); i += 32) {
        clear &= bytes[i] == 0;
    }
    const bool rejected = __all_sync(0xffffffffU, rc == NIXL_ERR_NOT_ALLOWED && clear);
    if (threadIdx.x == 0) {
        gate->status = rejected ? NIXL_ERR_NOT_ALLOWED : NIXL_ERR_BACKEND;
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device>(gate->tested)
            .store(1, cuda::memory_order_release);
    }
}

__global__ void
revokedRkeyPut(nixlMemViewH local, nixlMemViewH remote, uint64_t token, FaultResult *out) {
    if (blockIdx.x || threadIdx.x) {
        return;
    }
    out->accepted = NIXL_ERR_BACKEND;
    out->commit = NIXL_ERR_BACKEND;
    out->completion = NIXL_IN_PROG;
    out->later_post = NIXL_IN_PROG;
    out->release = NIXL_IN_PROG;

    nixlGpuXferStatusH status{};
    out->accepted = nixlPutDeferred<nixl_gpu_level_t::THREAD>(
        {local, 0, kGuard}, {remote, 0, kGuard}, sizeof(uint64_t), token, &status);
    if (out->accepted != NIXL_IN_PROG) {
        return;
    }
    out->commit = nixlGpuCommit({local, 0, 0}, token);
    if (out->commit != NIXL_SUCCESS) {
        return;
    }

    const uint64_t deadline = clock64() + kTimeoutCycles;
    do {
        out->completion = nixlGpuOwnerGetXferStatus<nixl_gpu_level_t::THREAD>(status, token);
    } while (out->completion == NIXL_IN_PROG && clock64() < deadline);
    if (out->completion != NIXL_ERR_BACKEND) {
        return;
    }

    nixlGpuXferStatusH rejected{};
    out->later_post = nixlPutDeferred<nixl_gpu_level_t::THREAD>(
        {local, 0, kGuard}, {remote, 0, kGuard}, sizeof(uint64_t), token, &rejected);
    out->release = nixlGpuReleaseOwner({local, 0, kGuard}, token);
}

__global__ void
revokedRkeyWarpPut(nixlMemViewH local, nixlMemViewH remote, uint64_t token, FaultResult *out) {
    __shared__ __align__(16) unsigned char bytes[64 * sizeof(nixlGpuXferStatusH)];
    auto *a = reinterpret_cast<nixlGpuXferStatusH *>(bytes);
    auto *b = a + 32;
    for (uint32_t i = threadIdx.x; i < sizeof(bytes); i += 32) {
        bytes[i] = 0;
    }
    if (threadIdx.x == 0) {
        out->accepted = out->commit = NIXL_ERR_BACKEND;
        out->completion = out->later_post = out->release = NIXL_IN_PROG;
    }
    __syncwarp();
    const uint64_t offset = kGuard + uint64_t(threadIdx.x) * kStride;
    auto rc = nixlPutDeferredWarp({local, 0, offset}, {remote, 0, offset}, 8, token, a, true);
    const bool accepted = __all_sync(0xffffffffU, rc == NIXL_IN_PROG);
    if (threadIdx.x == 0) {
        out->accepted = accepted ? NIXL_IN_PROG : NIXL_ERR_UNKNOWN;
    }
    if (!accepted) {
        return;
    }
    int completion = NIXL_IN_PROG;
    if (threadIdx.x == 0) {
        out->commit = nixlGpuCommit({local, 0, 0}, token);
        if (out->commit == NIXL_SUCCESS) {
            const uint64_t start = clock64();
            do {
                completion = nixlGpuOwnerGetXferStatus(a[0], token);
            } while (completion == NIXL_IN_PROG && clock64() - start < kTimeoutCycles);
        }
        out->completion = completion;
    }
    completion = __shfl_sync(0xffffffffU, completion, 0);
    if (completion != NIXL_ERR_BACKEND) {
        return;
    }
    rc = nixlPutDeferredWarp({local, 0, offset}, {remote, 0, offset}, 8, token, b, true);
    bool clear = true;
    for (uint32_t i = 0; i < sizeof(*b); ++i) {
        clear &= b[threadIdx.x].storage[i] == 0;
    }
    const bool rejected = __all_sync(0xffffffffU, rc == NIXL_ERR_BACKEND && clear);
    if (threadIdx.x == 0) {
        out->later_post = rejected ? NIXL_ERR_BACKEND : NIXL_ERR_UNKNOWN;
        out->release = nixlGpuReleaseOwner({local, 0, 0}, token);
    }
}

__global__ void
validate(uint8_t *dst, uint32_t epoch, Validation *out) {
    if (blockIdx.x) {
        return;
    }
    __shared__ uint32_t marker_ready;
    if (threadIdx.x == 0) {
        out->first_bad = ~uint64_t{0};
        out->checked = 0;
        out->epoch = 0;
        out->status = NIXL_ERR_BACKEND;
        marker_ready = 1;
    }
    __syncthreads();
    auto *p = reinterpret_cast<uint64_t *>(dst + kMarker);
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> marker(*p);
    const uint64_t expected = (uint64_t(epoch) << 32) | 0x4e49584cull;
    if (threadIdx.x == 0) {
        out->marker_expected = expected;
    }
    const uint64_t deadline = clock64() + kTimeoutCycles;
    uint64_t seen;
    do {
        seen = marker.load(cuda::memory_order_acquire);
    } while (seen != expected && clock64() < deadline);
    if (seen != expected) {
        atomicExch(&marker_ready, 0U);
    }
    if (threadIdx.x == 0) {
        out->epoch = uint32_t(seen >> 32);
        out->marker_observed = seen;
    }
    __syncthreads();
    if (!marker_ready) {
        if (threadIdx.x == 0) {
            out->status = NIXL_ERR_BACKEND;
        }
        return;
    }

    uint64_t checked = 0, first_bad = ~uint64_t{0};
    for (uint64_t at = threadIdx.x; at < kArena; at += blockDim.x) {
        if (at >= kMarker && at < kMarker + sizeof(uint64_t)) {
            continue;
        }
        uint8_t expected_byte = kHole;
        if (at >= kGuard && at < kMarker) {
            const uint64_t offset = at - kGuard;
            const uint64_t item = offset / kStride, within = offset % kStride;
            if (within < kPayload) {
                expected_byte = byteFor(kSeed ^ (uint64_t(epoch) << 32) ^ item, within);
            }
        }
        if (dst[at] != expected_byte && at < first_bad) {
            first_bad = at;
        }
        ++checked;
    }
    if (checked) {
        atomicAdd(reinterpret_cast<unsigned long long *>(&out->checked), checked);
    }
    if (first_bad != ~uint64_t{0}) {
        atomicMin(reinterpret_cast<unsigned long long *>(&out->first_bad), first_bad);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (out->first_bad != ~uint64_t{0}) {
            const uint64_t offset = out->first_bad;
            uint8_t expected_byte = kHole;
            if (offset >= kGuard && offset < kMarker) {
                const uint64_t slot = offset - kGuard;
                const uint64_t item = slot / kStride, within = slot % kStride;
                if (within < kPayload) {
                    expected_byte = byteFor(kSeed ^ (uint64_t(epoch) << 32) ^ item, within);
                }
            }
            out->expected_byte = expected_byte;
            out->actual_byte = dst[offset];
        }
        out->status = out->checked == kExpectedChecked && out->first_bad == ~uint64_t{0} ?
            NIXL_SUCCESS :
            NIXL_ERR_BACKEND;
    }
}

void
cudaOk(cudaError_t s, const char *what) {
    if (s != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
    }
}

void
nixlOk(nixl_status_t s, const char *what) {
    if (s != NIXL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " status=" + std::to_string(s));
    }
}

std::string
read(const std::filesystem::path &p) {
    std::ifstream f(p);
    if (!f) {
        throw std::runtime_error("missing file " + p.string());
    }
    return {std::istreambuf_iterator<char>(f), {}};
}

void
publish(const std::filesystem::path &p, const std::string &s) {
    auto tmp = p.string() + ".tmp." + std::to_string(getpid());
    {
        std::ofstream f(tmp);
        if (!(f << s)) {
            throw std::runtime_error("write " + tmp);
        }
    }
    std::filesystem::rename(tmp, p);
}

std::string
profile() {
#ifdef NIXL_GPUNETIO_OWNER_GPU_SHARING_CONTROL
    return "gpu-sharing-control";
#else
    return "exclusive";
#endif
}

struct InotifyFd {
    int value;

    ~InotifyFd() {
        if (value >= 0) {
            close(value);
        }
    }
};

void
wait(const std::filesystem::path &p, const std::filesystem::path &fail) {
    InotifyFd fd{inotify_init1(IN_NONBLOCK | IN_CLOEXEC)};
    if (fd.value < 0) {
        throw std::runtime_error("inotify_init1 failed");
    }
    const auto parent = p.parent_path().empty() ? std::filesystem::path(".") : p.parent_path();
    if (inotify_add_watch(fd.value, parent.c_str(), IN_MOVED_TO | IN_CLOSE_WRITE | IN_CREATE) < 0) {
        throw std::runtime_error("inotify_add_watch failed for " + parent.string());
    }
    const auto until = std::chrono::steady_clock::now() + std::chrono::minutes(3);
    for (;;) {
        if (std::filesystem::exists(fail)) {
            throw std::runtime_error("peer failed: " + read(fail));
        }
        if (std::filesystem::exists(p)) {
            return;
        }
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   until - std::chrono::steady_clock::now())
                                   .count();
        if (remaining <= 0) {
            throw std::runtime_error("timeout waiting for " + p.string());
        }
        pollfd event{fd.value, POLLIN, 0};
        const int ready = poll(&event, 1, static_cast<int>(remaining));
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready < 0) {
            throw std::runtime_error("poll(inotify) failed");
        }
        if (ready == 0) {
            throw std::runtime_error("timeout waiting for " + p.string());
        }
        if (!(event.revents & POLLIN)) {
            throw std::runtime_error("inotify poll closed unexpectedly");
        }
        alignas(inotify_event) char events[4096];
        for (;;) {
            const ssize_t count = ::read(fd.value, events, sizeof(events));
            if (count > 0) {
                continue;
            }
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                break;
            }
            throw std::runtime_error("read(inotify) failed");
        }
    }
}

struct Device {
    void *p{};

    explicit Device(size_t n) {
        cudaOk(cudaMalloc(&p, n), "cudaMalloc");
    }

    ~Device() {
        if (p) {
            cudaFree(p);
        }
    }
};

struct Stream {
    cudaStream_t s{};

    Stream() {
        cudaOk(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    ~Stream() {
        if (s) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
        }
    }
};

} // namespace

int
main(int argc, char **argv) {
    const std::string requested_role = argc > 1 ? argv[1] : "";
    const bool fault_mode = requested_role == "fault-sender" || requested_role == "fault-receiver";
    if (argc != 6 || (requested_role != "sender" && requested_role != "receiver" && !fault_mode)) {
        std::cerr << "usage: owner-endpoint-reproducer sender|receiver|fault-sender|fault-receiver "
                     "SHARED_COORD_DIR RUN_ID GPU OOB_PORT\n";
        return 2;
    }
    const std::string role = fault_mode ? requested_role.substr(6) : requested_role;
    const char *batch_commit_env = std::getenv("OWNER_REPRO_BATCH_COMMIT");
    if (batch_commit_env && std::strcmp(batch_commit_env, "0") != 0 &&
        std::strcmp(batch_commit_env, "1") != 0) {
        std::cerr << "OWNER_REPRO_BATCH_COMMIT must be 0 or 1\n";
        return 2;
    }
    const bool batch_commit =
        !fault_mode && batch_commit_env && std::strcmp(batch_commit_env, "1") == 0;
    const char *precommit_env = std::getenv("OWNER_REPRO_PRECOMMIT_CHECK");
    if (precommit_env && std::strcmp(precommit_env, "0") != 0 &&
        std::strcmp(precommit_env, "1") != 0) {
        std::cerr << "OWNER_REPRO_PRECOMMIT_CHECK must be 0 or 1\n";
        return 2;
    }
    const bool check_precommit = !precommit_env || std::strcmp(precommit_env, "1") == 0;
    const char *prep_env = std::getenv("OWNER_REPRO_PREP_MODE");
    if (prep_env && std::strcmp(prep_env, "0") != 0 && std::strcmp(prep_env, "1") != 0 &&
        std::strcmp(prep_env, "2") != 0) {
        std::cerr << "OWNER_REPRO_PREP_MODE must be 0 (scalar/local), 1 (scalar/shared), or 2 "
                     "(warp/shared)\n";
        return 2;
    }
    const int prep_mode = prep_env ? std::atoi(prep_env) : 0;
    if (!fault_mode && prep_mode == 2 && !batch_commit) {
        std::cerr << "warp preparation requires OWNER_REPRO_BATCH_COMMIT=1\n";
        return 2;
    }
    const std::string run = argv[3];
    const std::filesystem::path dir = argv[2];
    const int gpu = std::stoi(argv[4]);
    const std::string port = argv[5];
    if (run.empty() || run.find('/') != std::string::npos) {
        std::cerr << "invalid RUN_ID\n";
        return 2;
    }
    try {
        std::filesystem::create_directories(dir);
        cudaOk(cudaSetDevice(gpu), "cudaSetDevice");
        Device arena(kArena), result(sizeof(Result)), gate(sizeof(Gate)),
            validation(sizeof(Validation));
        cudaOk(cudaMemset(arena.p, kHole, kArena), "clear arena");
        cudaOk(cudaDeviceSynchronize(), "complete initial arena fill before NIXL/stream setup");
        std::unique_ptr<nixlAgent> agent;
        nixlAgentConfig cfg;
        cfg.useProgThread = true;
        cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
        cfg.pthrDelay = 100000;
        agent = std::make_unique<nixlAgent>("owner_" + run + "_" + role + "_" + profile(), cfg);
        nixl_b_params_t params;
        params["native_device_api"] = "true";
        params["native_device_owner"] = "true";
        params["gpu_devices"] = std::to_string(gpu);
        params["oob_port"] = port;
        for (auto [key, env] : {std::pair{"network_devices", "OWNER_REPRO_RDMA_DEVICE"},
                                {"oob_interface", "OWNER_REPRO_OOB_INTERFACE"},
                                {"gid_index", "OWNER_REPRO_GID_INDEX"}}) {
            if (const char *v = std::getenv(env)) {
                params[key] = v;
            }
        }
        nixlBackendH *backend = nullptr;
        nixlOk(agent->createBackend("GPUNETIO", params, backend), "create backend");
        nixl_opt_args_t hint;
        hint.backends.push_back(backend);
        const uintptr_t addr = reinterpret_cast<uintptr_t>(arena.p);
        nixl_reg_dlist_t registration(VRAM_SEG);
        registration.addDesc({addr, kArena, static_cast<uint64_t>(gpu), {}});
        nixlOk(agent->registerMem(registration, &hint), "register VRAM");
        nixl_blob_t md;
        nixlOk(agent->getLocalMD(md), "get local MD");
        const auto base = run + "." + profile() + "." + role;
        publish(dir / (base + ".md"), md);
        publish(dir / (base + ".desc"),
                std::to_string(addr) + " " + std::to_string(kArena) + " " + std::to_string(gpu));
        Stream stream, probe;
        nixlMemViewH local = nullptr, remote = nullptr;
        if (role == "sender") {
            const auto peer = run + "." + profile() + ".receiver";
            wait(dir / (peer + ".md"), dir / (peer + ".failed"));
            const std::string md_peer = read(dir / (peer + ".md"));
            std::string peer_name;
            nixlOk(agent->loadRemoteMD(md_peer, peer_name), "load receiver MD");
            nixlOk(agent->makeConnection(peer_name, &hint), "make connection");
            std::istringstream d(read(dir / (peer + ".desc")));
            uintptr_t peer_addr;
            size_t peer_len;
            int peer_gpu;
            if (!(d >> peer_addr >> peer_len >> peer_gpu) || peer_len < kArena) {
                throw std::runtime_error("bad peer descriptor");
            }
            nixl_local_dlist_t ld(VRAM_SEG);
            ld.addDesc({addr, kArena, static_cast<uint64_t>(gpu)});
            nixl_remote_dlist_t rd(VRAM_SEG);
            rd.addDesc({peer_addr, peer_len, static_cast<uint64_t>(peer_gpu), peer_name});
            nixlOk(agent->prepMemView(ld, local, &hint), "prep local view");
            nixlOk(agent->prepMemView(rd, remote, &hint), "prep remote view");
            if (fault_mode) {
                const auto tag = run + "." + profile() + ".fault";
                publish(dir / (tag + ".prepared"), "sender views and connection ready");
                wait(dir / (tag + ".revoked"), dir / (run + "." + profile() + ".receiver.failed"));
                if (prep_mode == 2) {
                    revokedRkeyWarpPut<<<1, 32, 0, stream.s>>>(
                        local, remote, 1, static_cast<FaultResult *>(result.p));
                } else {
                    revokedRkeyPut<<<1, 1, 0, stream.s>>>(
                        local, remote, 1, static_cast<FaultResult *>(result.p));
                }
                cudaOk(cudaGetLastError(), "launch revoked-rkey owner PUT");
                if (cudaStreamSynchronize(stream.s) != cudaSuccess) {
                    publish(dir / (tag + ".failed"), "fault owner stream synchronization failed");
                    publish(dir / (run + "." + profile() + ".sender.failed"),
                            "fault stream synchronization failed");
                    agent.reset();
                    std::_Exit(4);
                }
                FaultResult f{};
                cudaOk(cudaMemcpy(&f, result.p, sizeof(f), cudaMemcpyDeviceToHost),
                       "read CQE fault statuses only");
                std::ostringstream detail;
                detail << "accepted=" << f.accepted << " commit=" << f.commit
                       << " completion=" << f.completion << " later_post=" << f.later_post
                       << " release=" << f.release;
                const bool passed = f.accepted == NIXL_IN_PROG && f.commit == NIXL_SUCCESS &&
                    f.completion == NIXL_ERR_BACKEND && f.later_post == NIXL_ERR_BACKEND &&
                    f.release == NIXL_ERR_BACKEND;
                publish(dir / (tag + ".result"), detail.str());
                if (!passed) {
                    publish(dir / (tag + ".failed"), detail.str());
                    publish(dir / (run + "." + profile() + ".sender.failed"),
                            "revoked-rkey/CQE/fail-closed assertion failed: " + detail.str());
                    agent.reset();
                    publish(dir / (tag + ".sender_closed"), "teardown attempted");
                    std::_Exit(5);
                }
                if (local) {
                    agent->releaseMemView(local);
                    local = nullptr;
                }
                if (remote) {
                    agent->releaseMemView(remote);
                    remote = nullptr;
                }
                cudaOk(cudaStreamSynchronize(stream.s), "sync fault owner stream before close");
                agent.reset();
                publish(dir / (tag + ".sender_closed"), detail.str());
                std::cout << "fault-sender," << detail.str() << "\n" << std::flush;
                std::_Exit(0);
            }
            std::cout << "profile,role,batch,window,phase,gpu_elapsed_ns,wrong_grid_status,"
                         "capacity_status,stale_token_status,release_status,"
                         "batch_commit,precommit_check,precommit_checked,expected_commits";
            if (prep_env) {
                std::cout << ",prep_mode";
            }
            std::cout << "\n";
            for (uint32_t b = 0; b < 3 * (kWarmup + kMeasured); ++b) {
                const uint32_t rep = b % (kWarmup + kMeasured), epoch = b + 1,
                               window = kWindows[b / (kWarmup + kMeasured)];
                const std::string tag = run + "." + profile() + "." + std::to_string(b);
                wait(dir / (tag + ".ready"), dir / (tag + ".failed"));
                cudaOk(cudaMemsetAsync(gate.p, 0, sizeof(Gate), stream.s), "clear gate");
                ownerBatch<<<1, 256, 0, stream.s>>>(local,
                                                    remote,
                                                    static_cast<uint8_t *>(arena.p),
                                                    epoch,
                                                    window,
                                                    epoch,
                                                    static_cast<Result *>(result.p),
                                                    static_cast<Gate *>(gate.p),
                                                    b == 0,
                                                    batch_commit,
                                                    check_precommit,
                                                    prep_mode);
                if (b == 0) {
                    if (prep_mode == 2) {
                        wrongGridWarp<<<1, 32, 0, probe.s>>>(
                            local, remote, epoch, static_cast<Gate *>(gate.p));
                    } else {
                        wrongGrid<<<1, 1, 0, probe.s>>>(
                            local, remote, epoch, static_cast<Gate *>(gate.p));
                    }
                }
                cudaOk(cudaGetLastError(), "launch owner batch");
                if (cudaStreamSynchronize(stream.s) != cudaSuccess) {
                    publish(dir / (tag + ".failed"),
                            "owner stream synchronization failed; keeping VRAM until process exit");
                    std::_Exit(4);
                }
                if (b == 0 && cudaStreamSynchronize(probe.s) != cudaSuccess) {
                    publish(dir / (tag + ".failed"),
                            "wrong-owner stream synchronization failed; keeping VRAM until process "
                            "exit");
                    std::_Exit(4);
                }
                Result r{};
                Gate g{};
                cudaOk(cudaMemcpy(&r, result.p, sizeof(r), cudaMemcpyDeviceToHost),
                       "read status/timing only");
                if (b == 0) {
                    cudaOk(cudaMemcpy(&g, gate.p, sizeof(g), cudaMemcpyDeviceToHost),
                           "read wrong-owner status");
                }
                if ((batch_commit && check_precommit &&
                     (r.precommit_status != NIXL_IN_PROG || r.precommit_checked != kWqes)) ||
                    ((!batch_commit || !check_precommit) && r.precommit_checked != 0)) {
                    std::ostringstream detail;
                    detail << "precommit assertion: batch=" << b << " window=" << window
                           << " enabled=" << batch_commit
                           << " precommit_status=" << r.precommit_status
                           << " expected_status=" << NIXL_IN_PROG
                           << " precommit_checked=" << r.precommit_checked
                           << " expected_checked=" << (batch_commit && check_precommit ? kWqes : 0)
                           << " kernel_status=" << r.status << " release_status=" << r.release;
                    publish(dir / (tag + ".failed"), detail.str());
                    std::cerr << detail.str() << "\n";
                    throw std::runtime_error(detail.str());
                }
                if (r.status != NIXL_SUCCESS || r.release != NIXL_SUCCESS) {
                    publish(dir / (tag + ".failed"),
                            "owner batch/CQ/release/wrong-grid check failed");
                    agent.reset();
                    std::_Exit(3);
                }
                if (b == 0 && g.status != NIXL_ERR_NOT_ALLOWED) {
                    publish(dir / (tag + ".failed"), "same-token second grid was not rejected");
                    throw std::runtime_error("wrong-grid status was not NIXL_ERR_NOT_ALLOWED");
                }
                if (window == 32 && rep == 0 && r.capacity != NIXL_ERR_NOT_ALLOWED) {
                    throw std::runtime_error("33rd outstanding WQE was not rejected at the bound");
                }
                int stale_status = -999;
                if (b < 2) {
                    cudaOk(cudaMemsetAsync(gate.p, 0, sizeof(Gate), probe.s),
                           "clear stale-token result");
                    staleToken<<<1, 1, 0, probe.s>>>(local, remote, 1, static_cast<Gate *>(gate.p));
                    cudaOk(cudaGetLastError(), "launch stale-token probe");
                    if (cudaStreamSynchronize(probe.s) != cudaSuccess) {
                        publish(dir / (tag + ".failed"),
                                "stale-token probe stream synchronization failed");
                        std::_Exit(4);
                    }
                    cudaOk(cudaMemcpy(&g, gate.p, sizeof(g), cudaMemcpyDeviceToHost),
                           "read stale-token status");
                    stale_status = g.status;
                    if (stale_status != NIXL_ERR_NOT_ALLOWED) {
                        publish(dir / (tag + ".failed"), "closed/stale owner token was accepted");
                        throw std::runtime_error("stale owner token was not rejected");
                    }
                }
                publish(dir / (tag + ".done"), "ok");
                wait(dir / (tag + ".verified"), dir / (tag + ".failed"));
                std::cout << profile() << ",sender," << b << "," << window << ","
                          << (rep < kWarmup ? "warmup" : "measure") << "," << r.elapsed_ns << ","
                          << r.wrong_owner << "," << r.capacity << "," << stale_status << ","
                          << r.release << "," << batch_commit << "," << check_precommit << ","
                          << r.precommit_checked << ","
                          << (batch_commit ? kWqes / window + 1 : kWqes + 1);
                if (prep_env) {
                    std::cout << "," << prep_mode;
                }
                std::cout << "\n";
            }
        } else {
            const auto peer = run + "." + profile() + ".sender";
            wait(dir / (peer + ".md"), dir / (peer + ".failed"));
            std::string peer_name;
            nixlOk(agent->loadRemoteMD(read(dir / (peer + ".md")), peer_name), "load sender MD");
            nixl_local_dlist_t ld(VRAM_SEG);
            ld.addDesc({addr, kArena, static_cast<uint64_t>(gpu)});
            nixlOk(agent->prepMemView(ld, local, &hint), "prep receiver validation view");
            if (fault_mode) {
                const auto tag = run + "." + profile() + ".fault";
                wait(dir / (tag + ".prepared"), dir / (run + "." + profile() + ".sender.failed"));
                agent->releaseMemView(local);
                local = nullptr;
                nixlOk(agent->deregisterMem(registration, &hint), "revoke receiver MR");
                publish(dir / (tag + ".revoked"),
                        "receiver view released and target MR deregistered");
                wait(dir / (tag + ".sender_closed"),
                     dir / (run + "." + profile() + ".sender.failed"));
                agent.reset();
                publish(dir / (tag + ".receiver_closed"), "receiver engine teardown complete");
                std::cout << "fault-receiver,MR revoked after sender prepared remote view\n"
                          << std::flush;
                std::_Exit(0);
            } else {
                for (uint32_t b = 0; b < 3 * (kWarmup + kMeasured); ++b) {
                    const uint32_t epoch = b + 1;
                    const std::string tag = run + "." + profile() + "." + std::to_string(b);
                    cudaOk(cudaMemsetAsync(arena.p, kHole, kArena, stream.s),
                           "reset GPU destination");
                    cudaOk(cudaMemsetAsync(validation.p, 0, sizeof(Validation), stream.s),
                           "clear validation result");
                    cudaOk(cudaStreamSynchronize(stream.s),
                           "complete receiver resets before readiness");
                    validate<<<1, 256, 0, stream.s>>>(static_cast<uint8_t *>(arena.p),
                                                      epoch,
                                                      static_cast<Validation *>(validation.p));
                    cudaOk(cudaGetLastError(), "launch GPU validator");
                    publish(dir / (tag + ".ready"), "validator-launched");
                    wait(dir / (tag + ".done"), dir / (tag + ".failed"));
                    if (cudaStreamSynchronize(stream.s) != cudaSuccess) {
                        publish(dir / (tag + ".failed"),
                                "receiver validation stream synchronization failed; keeping VRAM "
                                "until process exit");
                        std::_Exit(4);
                    }
                    Validation v{};
                    cudaOk(cudaMemcpy(&v, validation.p, sizeof(v), cudaMemcpyDeviceToHost),
                           "read GPU validation status only");
                    if (v.status != NIXL_SUCCESS || v.epoch != epoch ||
                        v.checked != kExpectedChecked || v.first_bad != ~uint64_t{0}) {
                        std::ostringstream detail;
                        detail << "GPU validation mismatch: status=" << v.status
                               << " epoch=" << v.epoch << " expected_epoch=" << epoch
                               << " checked=" << v.checked
                               << " expected_checked=" << kExpectedChecked
                               << " first_bad=" << v.first_bad
                               << " expected_byte=" << v.expected_byte
                               << " actual_byte=" << v.actual_byte
                               << " marker_expected=" << v.marker_expected
                               << " marker_observed=" << v.marker_observed;
                        publish(dir / (tag + ".failed"), detail.str());
                        std::cerr << detail.str() << "\n";
                        throw std::runtime_error(detail.str());
                    }
                    publish(dir / (tag + ".verified"), "ok");
                    const uint32_t rep = b % (kWarmup + kMeasured),
                                   window = kWindows[b / (kWarmup + kMeasured)];
                    std::cout << profile() << ",receiver," << b << "," << window << ","
                              << (rep < kWarmup ? "warmup" : "measure") << ",0,0,0\n";
                }
            }
        }
        if (local) {
            agent->releaseMemView(local);
        }
        if (remote) {
            agent->releaseMemView(remote);
        }
        cudaOk(cudaStreamSynchronize(stream.s), "final caller stream sync");
        cudaOk(cudaStreamSynchronize(probe.s), "final probe stream sync");
        agent.reset(); // Stop the persistent progress engine before VRAM RAII frees buffers.
        return 0;
    }
    catch (const std::exception &e) {
        try {
            publish(dir / (run + "." + profile() + "." + role + ".failed"), e.what());
        }
        catch (...) {
        }
        std::cerr << "FAIL: " << e.what() << "\n";
        if (fault_mode) {
            std::_Exit(1);
        }
        return 1;
    }
}
