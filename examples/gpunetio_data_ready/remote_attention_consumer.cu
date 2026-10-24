/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
 * SPDX-License-Identifier: Apache-2.0
 */
// Public-owner-API remote-attention consumer harness; no private GET API or
// backend-private state is used.

#include <cuda.h>
#include <cuda/atomic>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <nixl.h>
#include <gpu/nixl_device.cuh>

#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <optional>
#include <poll.h>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kMaxSteps = 20;
constexpr uint32_t kPagesPerStep = 32;
constexpr uint32_t kPages = kMaxSteps * kPagesPerStep;
constexpr uint32_t kSeq = 64;
constexpr uint32_t kHead = 128;
constexpr size_t kHalfPage = kSeq * kHead;
constexpr uint32_t kRaSeq = kSeq;
constexpr uint32_t kRaHead = kHead;
constexpr size_t kRaHalfPage = kHalfPage;
constexpr size_t kPageBytes = 2 * kHalfPage * sizeof(__half);
constexpr size_t kGuardBytes = 64;
constexpr double kTolerance = 1.0e-3;
constexpr uint8_t kGuard = 0xa5;

constexpr size_t
align64(size_t x) {
    return (x + 63u) & ~size_t(63u);
}

constexpr size_t kTableOff = 0;
constexpr size_t kTableBytes = size_t(kPages) * kPageBytes;
constexpr size_t kReplyOff = align64(kTableOff + kTableBytes);
constexpr size_t kReplyStride = align64(kPageBytes + 2 * kGuardBytes);
constexpr size_t kReadyOff = align64(kReplyOff + kMaxSteps * kReplyStride);
constexpr size_t kAckOff = align64(kReadyOff + 64);
constexpr size_t kQueryOff = align64(kAckOff + 64);
constexpr size_t kPreOff = align64(kQueryOff + kHead * sizeof(__half));
constexpr size_t kPostOff = align64(kPreOff + kMaxSteps * kHead * sizeof(__half));
constexpr size_t kPageIdOff = align64(kPostOff + kMaxSteps * kHead * sizeof(__half));
constexpr size_t kSummaryOff = align64(kPageIdOff + kMaxSteps * sizeof(uint32_t));
constexpr size_t kSummaryBytes = size_t(kPages) * kHead * sizeof(__half);
constexpr size_t kStatusOff = align64(kSummaryOff + kSummaryBytes);
constexpr size_t kMetricOff = align64(kStatusOff + 4 * sizeof(nixlGpuXferStatusH));
constexpr size_t kMetricStride = 6;
constexpr size_t kErrorOff = align64(kMetricOff + kMaxSteps * kMetricStride * sizeof(uint64_t));
constexpr size_t kArenaBytes = align64(kErrorOff + 4 * sizeof(uint32_t));

struct alignas(64) ReadyRecord {
    uint64_t epoch;
    uint32_t page;
    uint32_t step;
    uint64_t epoch_xor;
    uint32_t page_xor;
    uint32_t step_xor;
    uint8_t pad[32];
};

static_assert(sizeof(ReadyRecord) == 64);

struct AckRecord {
    uint64_t epoch;
    uint32_t step;
    uint32_t step_xor;
};

static_assert(sizeof(AckRecord) == 16);

struct Args {
    std::string role, coord, run, target, arm = "paired", first_arm = "ordered", rdma,
                                          oob_interface;
    std::string inject = "none";
    uint32_t gpu = 0, steps = kMaxSteps, gid = 0;
    uint32_t episodes = 50, warmups = 1;
    uint16_t control_port = 0, oob_port = 0;
    uint64_t epoch = 0, timeout_ms = 45000;
};

[[noreturn]] void
usage(const char *why = nullptr, int exit_code = 2) {
    if (why) {
        std::cerr << "error: " << why << "\n";
    }
    std::cerr << "usage: remote_attention_consumer --role source|consumer --coord DIR "
                 "--run ID --epoch N --gpu N --control-port PORT --arm paired|ordered|cqwait "
                 "[--episodes N --warmups N --first-arm ordered|cqwait] "
                 "[--inject none|payload|ready-epoch|ack-epoch] "
                 "[--target-ip IPV4 --oob-port PORT --rdma DEV --oob-interface IFACE "
                 "--gid-index N --steps 1..20 --timeout-ms N]\n";
    std::exit(exit_code);
}

uint64_t
parseU64(const char *s, const char *name) {
    if (!s || !*s || *s == '-') {
        usage(name);
    }
    size_t used = 0;
    const auto v = std::stoull(s, &used, 0);
    if (s[used] != '\0') {
        usage(name);
    }
    return v;
}

uint32_t
parseU32(const char *s, const char *name) {
    const uint64_t v = parseU64(s, name);
    if (v > UINT32_MAX) {
        usage(name);
    }
    return static_cast<uint32_t>(v);
}

uint16_t
parsePort(const char *s, const char *name) {
    const uint64_t v = parseU64(s, name);
    if (!v || v > 65535) {
        usage(name);
    }
    return static_cast<uint16_t>(v);
}

Args
parseArgs(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--help") {
            usage(nullptr, 0);
        }
        if (i + 1 == argc) {
            usage("option requires a value");
        }
        const char *v = argv[++i];
        if (key == "--role") {
            a.role = v;
        } else if (key == "--coord") {
            a.coord = v;
        } else if (key == "--run") {
            a.run = v;
        } else if (key == "--target-ip") {
            a.target = v;
        } else if (key == "--arm") {
            a.arm = v;
        } else if (key == "--first-arm") {
            a.first_arm = v;
        } else if (key == "--rdma") {
            a.rdma = v;
        } else if (key == "--oob-interface") {
            a.oob_interface = v;
        } else if (key == "--inject") {
            a.inject = v;
        } else if (key == "--gpu") {
            a.gpu = parseU32(v, "invalid --gpu");
        } else if (key == "--steps") {
            a.steps = parseU32(v, "invalid --steps");
        } else if (key == "--episodes") {
            a.episodes = parseU32(v, "invalid --episodes");
        } else if (key == "--warmups") {
            a.warmups = parseU32(v, "invalid --warmups");
        } else if (key == "--gid-index") {
            a.gid = parseU32(v, "invalid --gid-index");
        } else if (key == "--control-port") {
            a.control_port = parsePort(v, "invalid --control-port");
        } else if (key == "--oob-port") {
            a.oob_port = parsePort(v, "invalid --oob-port");
        } else if (key == "--epoch") {
            a.epoch = parseU64(v, "invalid --epoch");
        } else if (key == "--timeout-ms") {
            a.timeout_ms = parseU64(v, "invalid --timeout-ms");
        } else {
            usage("unknown option");
        }
    }
    if (a.role != "source" && a.role != "consumer") {
        usage("role must be source or consumer");
    }
    if (a.coord.empty() || a.run.empty() || a.run.size() > 64 ||
        a.run.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
            std::string::npos) {
        usage("coord and a safe run ID are required");
    }
    const uint64_t run_episodes = a.arm == "paired" ? 2ull * (uint64_t(a.episodes) + a.warmups) :
                                                      uint64_t(a.episodes) + a.warmups;
    const uint64_t run_steps = run_episodes * a.steps;
    if (!a.control_port || !a.epoch || run_episodes == 0 || run_episodes > 2200 || run_steps == 0 ||
        run_steps - 1 > UINT64_MAX - a.epoch || !a.timeout_ms || a.timeout_ms > 600000 ||
        a.steps == 0 || a.steps > kMaxSteps) {
        usage("epoch, port, timeout, or steps is out of range");
    }
    if ((a.arm != "paired" && a.arm != "ordered" && a.arm != "cqwait") ||
        (a.first_arm != "ordered" && a.first_arm != "cqwait") || a.episodes == 0) {
        usage("invalid arm, first arm, or episode count");
    }
    if (a.inject != "none" && a.inject != "payload" && a.inject != "ready-epoch" &&
        a.inject != "ack-epoch") {
        usage("invalid failure injection");
    }
    if (a.inject != "none" && a.arm == "paired") {
        usage("failure injection requires a single --arm ordered or cqwait");
    }
    if (a.inject != "none" && (a.episodes != 1 || a.warmups != 0 || a.steps != 1)) {
        usage("failure injection requires --episodes 1 --warmups 0 --steps 1");
    }
    if (a.role == "source" && a.target.empty()) {
        usage("source requires --target-ip");
    }
    return a;
}

struct Episode {
    uint64_t index = 0, epoch = 0, owner_token = 0;
    uint32_t pair = 0;
    bool warmup = false;
    std::string arm;
};

uint64_t
episodeCount(const Args &a) {
    return a.arm == "paired" ? 2ull * (uint64_t(a.warmups) + a.episodes) :
                               uint64_t(a.warmups) + a.episodes;
}

Episode
episodeAt(const Args &a, uint64_t index) {
    Episode e;
    e.index = index;
    e.epoch = a.epoch + index * a.steps;
    e.owner_token = index + 1;
    const bool paired = a.arm == "paired";
    const uint64_t warmup_count = paired ? 2ull * a.warmups : a.warmups;
    e.warmup = index < warmup_count;
    if (!paired) {
        e.arm = a.arm;
        e.pair = static_cast<uint32_t>(index >= a.warmups ? index - a.warmups : 0);
        return e;
    }
    const uint64_t within = e.warmup ? index : index - warmup_count;
    e.pair = static_cast<uint32_t>(within / 2);
    const bool ordered_first = ((e.pair & 1u) == 0) == (a.first_arm == "ordered");
    e.arm = ((within & 1u) == 0) == ordered_first ? "ordered" : "cqwait";
    return e;
}

void
cudaCheck(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

void
nixlCheck(nixl_status_t e, const char *what) {
    if (e != NIXL_SUCCESS) {
        throw std::runtime_error(std::string(what) + " returned NIXL status " + std::to_string(e));
    }
}

class DeviceArena {
public:
    explicit DeviceArena(size_t bytes) : bytes_(bytes) {
        cudaCheck(cudaMalloc(&ptr_, bytes), "cudaMalloc arena");
    }

    ~DeviceArena() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    DeviceArena(const DeviceArena &) = delete;
    DeviceArena &
    operator=(const DeviceArena &) = delete;

    uint8_t *
    get() const {
        return static_cast<uint8_t *>(ptr_);
    }

    size_t
    size() const {
        return bytes_;
    }

private:
    void *ptr_ = nullptr;
    size_t bytes_;
};

std::filesystem::path
filePath(const Args &a, const std::string &role, const char *suffix) {
    return std::filesystem::path(a.coord) / (a.run + "." + role + suffix);
}

void
publishFile(const std::filesystem::path &p, std::string_view data) {
    auto tmp = p;
    tmp += ".tmp." + std::to_string(getpid());
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out || !out.write(data.data(), static_cast<std::streamsize>(data.size()))) {
            throw std::runtime_error("cannot write " + tmp.string());
        }
        out.flush();
        if (!out) {
            throw std::runtime_error("cannot flush " + tmp.string());
        }
    }
    int fd = ::open(tmp.c_str(), O_RDONLY);
    if (fd < 0 || ::fsync(fd)) {
        if (fd >= 0) {
            ::close(fd);
        }
        throw std::runtime_error("fsync failed: " + tmp.string());
    }
    ::close(fd);
    std::filesystem::rename(tmp, p);
}

std::string
readFile(const std::filesystem::path &p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + p.string());
    }
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::tuple<uintptr_t, size_t, uint32_t>
parseDesc(const std::string &s) {
    uint64_t address = 0, bytes = 0, gpu = 0;
    std::istringstream in(s);
    if (!(in >> address >> bytes >> gpu) || !address || bytes < kArenaBytes || gpu > UINT32_MAX) {
        throw std::runtime_error("invalid peer descriptor");
    }
    return {
        static_cast<uintptr_t>(address), static_cast<size_t>(bytes), static_cast<uint32_t>(gpu)};
}

class Control {
public:
    Control(const Args &a) {
        if (a.role == "consumer") {
            int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (listen_fd < 0) {
                throw std::runtime_error("control socket failed");
            }
            int one = 1;
            ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
            addr.sin_port = htons(a.control_port);
            if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) ||
                ::listen(listen_fd, 1)) {
                ::close(listen_fd);
                throw std::runtime_error("control bind/listen failed");
            }
            pollfd p{listen_fd, POLLIN, 0};
            if (::poll(&p, 1, static_cast<int>(a.timeout_ms)) <= 0) {
                ::close(listen_fd);
                throw std::runtime_error("control accept timeout");
            }
            fd_ = ::accept(listen_fd, nullptr, nullptr);
            ::close(listen_fd);
            if (fd_ < 0) {
                throw std::runtime_error("control accept failed");
            }
        } else {
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(a.control_port);
            if (::inet_pton(AF_INET, a.target.c_str(), &addr.sin_addr) != 1) {
                throw std::runtime_error("target IP must be numeric IPv4");
            }
            const auto end =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(a.timeout_ms);
            do {
                fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
                if (fd_ < 0) {
                    throw std::runtime_error("control socket failed");
                }
                if (::connect(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
                    break;
                }
                ::close(fd_);
                fd_ = -1;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } while (std::chrono::steady_clock::now() < end);
            if (fd_ < 0) {
                throw std::runtime_error("control connect timeout");
            }
        }
        int one = 1;
        ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        timeval tv{static_cast<time_t>(a.timeout_ms / 1000),
                   static_cast<suseconds_t>((a.timeout_ms % 1000) * 1000)};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

    ~Control() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    void
    send(char c) {
        if (::send(fd_, &c, 1, MSG_NOSIGNAL) != 1) {
            throw std::runtime_error("control send failed");
        }
    }

    void
    expect(char c) {
        char got = 0;
        if (::recv(fd_, &got, 1, MSG_WAITALL) != 1 || got != c) {
            throw std::runtime_error(std::string("control protocol expected ") + c);
        }
    }

    char
    receive() {
        char got = 0;
        if (::recv(fd_, &got, 1, MSG_WAITALL) != 1) {
            throw std::runtime_error("control receive failed");
        }
        return got;
    }

    void
    sendText(std::string_view s) {
        const uint32_t n = static_cast<uint32_t>(s.size());
        sendAll(&n, sizeof(n));
        sendAll(s.data(), s.size());
    }

    std::string
    receiveText() {
        uint32_t n = 0;
        receiveAll(&n, sizeof(n));
        if (n > 4096) {
            throw std::runtime_error("control frame too large");
        }
        std::string s(n, '\0');
        receiveAll(s.data(), s.size());
        return s;
    }

private:
    void
    sendAll(const void *data, size_t n) {
        const auto *p = static_cast<const char *>(data);
        while (n) {
            const ssize_t done = ::send(fd_, p, n, MSG_NOSIGNAL);
            if (done <= 0) {
                throw std::runtime_error("control frame send failed");
            }
            p += done;
            n -= static_cast<size_t>(done);
        }
    }

    void
    receiveAll(void *data, size_t n) {
        auto *p = static_cast<char *>(data);
        while (n) {
            const ssize_t done = ::recv(fd_, p, n, 0);
            if (done <= 0) {
                throw std::runtime_error("control frame receive failed");
            }
            p += done;
            n -= static_cast<size_t>(done);
        }
    }

    int fd_ = -1;
};

__host__ __device__ inline uint16_t
raKBits(size_t page, size_t pos, size_t d) {
    const int x = static_cast<int>((page * 17 + pos * 13 + d * 7) % 101) - 50;
    return __half_as_ushort(__float2half(static_cast<float>(x) / 32.0f));
}

__host__ __device__ inline uint16_t
raVBits(size_t page, size_t pos, size_t d) {
    const int x = static_cast<int>((page * 29 + pos * 5 + d * 11 + 3) % 127) - 63;
    return __half_as_ushort(__float2half(static_cast<float>(x) / 40.0f));
}

std::vector<__half>
raMakeTable() {
    std::vector<__half> out(size_t(kPages) * 2 * kHalfPage);
    for (size_t p = 0; p < kPages; ++p) {
        for (size_t i = 0; i < kHalfPage; ++i) {
            out[p * 2 * kHalfPage + i] = __ushort_as_half(raKBits(p, i / kHead, i % kHead));
            out[p * 2 * kHalfPage + kHalfPage + i] =
                __ushort_as_half(raVBits(p, i / kHead, i % kHead));
        }
    }
    return out;
}

std::vector<uint16_t>
raMakeSummary() {
    std::vector<uint16_t> out(size_t(kPages) * kHead);
    for (size_t p = 0; p < kPages; ++p) {
        for (size_t d = 0; d < kHead; ++d) {
            float sum = 0;
            for (size_t t = 0; t < kSeq; ++t) {
                sum += __half2float(__ushort_as_half(raKBits(p, t, d)));
            }
            out[p * kHead + d] = __half_as_ushort(__float2half_rn(sum / kSeq));
        }
    }
    return out;
}

// Kept byte-for-byte in algorithm and body with remote_attention_pilot.cu::raAttention.
__device__ void
raAttention(__half *q,
            const __half *page,
            __half *qpre,
            __half *qpost,
            uint32_t *page_ids,
            uint32_t step,
            uint32_t selected_page) {
    __shared__ float scores[kRaSeq];
    __shared__ float weights[kRaSeq];
    if (threadIdx.x < kRaHead) {
        qpre[step * kRaHead + threadIdx.x] = q[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        page_ids[step] = selected_page;
    }
    __syncthreads();
    if (threadIdx.x < kRaSeq) {
        float dot = 0.0f;
        for (int d = 0; d < kRaHead; ++d) {
            dot += __half2float(q[d]) * __half2float(page[threadIdx.x * kRaHead + d]);
        }
        scores[threadIdx.x] = dot * 0.08838834764831845f; // 1 / sqrt(128)
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float max_score = scores[0];
        for (int i = 1; i < kRaSeq; ++i) {
            max_score = fmaxf(max_score, scores[i]);
        }
        float sum = 0.0f;
        for (int i = 0; i < kRaSeq; ++i) {
            weights[i] = expf(scores[i] - max_score);
            sum += weights[i];
        }
        for (int i = 0; i < kRaSeq; ++i) {
            weights[i] /= sum;
        }
    }
    __syncthreads();
    if (threadIdx.x < kRaHead) {
        float value = 0.0f;
        const __half *v = page + kRaHalfPage;
        for (int i = 0; i < kRaSeq; ++i) {
            value += weights[i] * __half2float(v[i * kRaHead + threadIdx.x]);
        }
        q[threadIdx.x] = __float2half_rn(0.5f * __half2float(q[threadIdx.x]) + 0.5f * value);
        qpost[step * kRaHead + threadIdx.x] = q[threadIdx.x];
    }
}

__device__ uint64_t
globalTimerNs() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

__device__ inline uint32_t
raRoute(const __half *q, const __half *summary, uint32_t step) {
    float best_score = -CUDART_INF_F;
    uint32_t best_page = UINT32_MAX;
    for (uint32_t c = 0; c < 16; ++c) {
        const uint32_t logical_page = (31u * c + 17u * step + 3u) % kPagesPerStep;
        const uint32_t page = step * kPagesPerStep + logical_page;
        float score = 0;
        for (uint32_t d = 0; d < kHead; ++d) {
            score += __half2float(q[d]) * __half2float(summary[page * kHead + d]);
        }
        if (score > best_score || (score == best_score && page < best_page)) {
            best_score = score;
            best_page = page;
        }
    }
    return best_page;
}

__device__ nixl_status_t
waitOwnerStatus(nixlGpuXferStatusH &status, uint64_t owner_token, uint64_t deadline) {
    nixl_status_t rc;
    do {
        rc = nixlGpuOwnerGetXferStatus<nixl_gpu_level_t::THREAD>(status, owner_token);
        if (rc == NIXL_IN_PROG && globalTimerNs() >= deadline) {
            return NIXL_IN_PROG;
        }
        if (rc == NIXL_IN_PROG) {
            __nanosleep(64);
        }
    } while (rc == NIXL_IN_PROG);
    return rc;
}

__device__ bool
waitReady(uint64_t *word, uint64_t epoch, uint64_t deadline, uint64_t *seen_ns, uint32_t *error) {
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ready(*word);
    for (;;) {
        const uint64_t got = ready.load(cuda::memory_order_acquire);
        if (got == epoch) {
            *seen_ns = globalTimerNs();
            return true;
        }
        if (got > epoch) {
            *error = 2;
            return false;
        }
        if (globalTimerNs() >= deadline) {
            *error = 1;
            return false;
        }
        __nanosleep(64);
    }
}

__global__ void
sourceActor(nixlMemViewH local_view,
            nixlMemViewH remote_view,
            uint8_t *base,
            uint32_t steps,
            uint64_t epoch,
            uint64_t owner_token,
            uint64_t timeout_ns,
            bool cqwait_arm,
            uint32_t *safe_to_release,
            bool inject_ready_epoch,
            bool inject_payload) {
    if (blockIdx.x) {
        return;
    }
    auto *statuses = reinterpret_cast<nixlGpuXferStatusH *>(base + kStatusOff);
    auto *errors = reinterpret_cast<uint32_t *>(base + kErrorOff);
    auto *pages = reinterpret_cast<const uint32_t *>(base + kPageIdOff);
    auto *metrics = reinterpret_cast<uint64_t *>(base + kMetricOff);
    auto *ack_record = reinterpret_cast<AckRecord *>(base + kAckOff);
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ack(ack_record->epoch);
    __shared__ uint32_t page, stop;
    bool data_live = false, ready_live = false;
    bool claimed = false, unsafe = false;
    const nixlMemViewElem owner_elem{local_view, 0, kTableOff};
    for (uint32_t step = 0; step < steps; ++step) {
        if (threadIdx.x == 0) {
            stop = 0;
            page = pages[step];
            if (page >= kPages) {
                errors[0] = 9;
                stop = 1;
            }
        }
        __syncthreads();
        if (stop) {
            break;
        }

        if (threadIdx.x == 0) {
            const uint64_t step_epoch = epoch + step;
            const size_t dst_page = kReplyOff + size_t(step) * kReplyStride + kGuardBytes;
            if (inject_payload && step == 0) {
                auto *word =
                    reinterpret_cast<uint16_t *>(base + kTableOff + size_t(page) * kPageBytes);
                *word ^= 1u; // isolated negative test only; normal source pages stay immutable
                __threadfence_system();
            }
            auto &data_status = statuses[0];
            auto &ready_status = statuses[1];
            data_status = nixlGpuXferStatusH{};
            ready_status = nixlGpuXferStatusH{};
            data_live = false;
            ready_live = false;
            metrics[step * kMetricStride] = globalTimerNs(); // source post-data RTT origin
            auto rc = nixlPutDeferred<nixl_gpu_level_t::THREAD>(
                {local_view, 0, kTableOff + size_t(page) * kPageBytes},
                {remote_view, 0, dst_page},
                kPageBytes,
                owner_token,
                &data_status);
            if (rc != NIXL_IN_PROG) {
                errors[0] = 10;
                stop = 1;
            } else {
                claimed = true;
                data_live = true;
                rc = nixlGpuCommit(owner_elem, owner_token);
                if (rc != NIXL_SUCCESS) {
                    errors[0] = 11;
                    unsafe = true;
                    stop = 1;
                }
            }
            if (!stop && cqwait_arm) {
                rc = waitOwnerStatus(data_status, owner_token, globalTimerNs() + timeout_ns);
                if (rc == NIXL_IN_PROG) {
                    errors[0] = 12;
                    unsafe = true;
                    stop = 1;
                } else {
                    data_live = false;
                    if (rc != NIXL_SUCCESS) {
                        errors[0] = 13;
                        stop = 1;
                    }
                }
            }
            if (!stop) {
                auto *record = reinterpret_cast<ReadyRecord *>(base + kReadyOff);
                record->epoch = step_epoch + (inject_ready_epoch ? 1u : 0u);
                record->page = page;
                record->step = step;
                record->epoch_xor = ~step_epoch;
                record->page_xor = ~page;
                record->step_xor = ~step;
                __threadfence_system();
                rc = nixlPutDeferred<nixl_gpu_level_t::THREAD>({local_view, 0, kReadyOff},
                                                               {remote_view, 0, kReadyOff},
                                                               sizeof(ReadyRecord),
                                                               owner_token,
                                                               &ready_status);
                if (rc != NIXL_IN_PROG) {
                    errors[0] = 14;
                    stop = 1;
                } else {
                    ready_live = true;
                    rc = nixlGpuCommit(owner_elem, owner_token);
                    if (rc != NIXL_SUCCESS) {
                        errors[0] = 15;
                        unsafe = true;
                        stop = 1;
                    }
                }
            }
            if (!stop) {
                const uint64_t ack_deadline = globalTimerNs() + timeout_ns;
                for (;;) {
                    const uint64_t observed = ack.load(cuda::memory_order_acquire);
                    if (observed == step_epoch) {
                        if (ack_record->step == step && ack_record->step_xor == ~step) {
                            break;
                        }
                        errors[0] = 25;
                        stop = 1;
                        break;
                    }
                    if (observed > step_epoch) {
                        errors[0] = 25;
                        stop = 1;
                        break;
                    }
                    if (globalTimerNs() >= ack_deadline) {
                        errors[0] = 16;
                        stop = 1;
                        break;
                    }
                    __nanosleep(64);
                }
                if (!stop) {
                    metrics[step * kMetricStride + 1] = globalTimerNs();
                }
            }

            if (!stop && !cqwait_arm) {
                rc = waitOwnerStatus(data_status, owner_token, globalTimerNs() + timeout_ns);
                if (rc == NIXL_IN_PROG) {
                    errors[0] = 17;
                    unsafe = true;
                    stop = 1;
                } else {
                    data_live = false;
                    if (rc != NIXL_SUCCESS) {
                        errors[0] = 18;
                        stop = 1;
                    }
                }
            }
            if (!stop) {
                rc = waitOwnerStatus(ready_status, owner_token, globalTimerNs() + timeout_ns);
                if (rc == NIXL_IN_PROG) {
                    errors[0] = 19;
                    unsafe = true;
                    stop = 1;
                } else {
                    ready_live = false;
                    if (rc != NIXL_SUCCESS) {
                        errors[0] = 20;
                        stop = 1;
                    }
                }
            }
            if (!stop) {
                ack.store(0, cuda::memory_order_release);
            }

            // On a recoverable arm/data failure, retire every accepted WQE before
            // releasing the owner. A timeout while a status is still IN_PROG fails closed.
            if (stop && !unsafe) {
                if (data_live) {
                    rc = waitOwnerStatus(data_status, owner_token, globalTimerNs() + timeout_ns);
                    if (rc == NIXL_IN_PROG) {
                        unsafe = true;
                    } else {
                        data_live = false;
                    }
                }
                if (!unsafe && ready_live) {
                    rc = waitOwnerStatus(ready_status, owner_token, globalTimerNs() + timeout_ns);
                    if (rc == NIXL_IN_PROG) {
                        unsafe = true;
                    } else {
                        ready_live = false;
                    }
                }
            }
            if (unsafe) {
                stop = 1;
            }
        }
        __syncthreads();
        if (stop) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        if (claimed && !unsafe && !data_live && !ready_live) {
            const auto rc = nixlGpuReleaseOwner(owner_elem, owner_token);
            if (rc == NIXL_SUCCESS) {
                *safe_to_release = 1;
            } else {
                errors[0] = 22;
                unsafe = true;
            }
        } else if (!claimed) {
            *safe_to_release = 1;
        }
        if (unsafe) {
            errors[0] = errors[0] ? errors[0] : 23;
            *safe_to_release = 0;
        }
    }
}

__global__ void
consumerActor(nixlMemViewH local_view,
              nixlMemViewH remote_view,
              uint8_t *base,
              uint32_t steps,
              uint64_t epoch,
              uint64_t owner_token,
              uint64_t timeout_ns,
              uint32_t *safe_to_release,
              bool inject_ack_epoch) {
    if (blockIdx.x) {
        return;
    }
    auto *errors = reinterpret_cast<uint32_t *>(base + kErrorOff);
    auto *metrics = reinterpret_cast<uint64_t *>(base + kMetricOff);
    auto *q = reinterpret_cast<__half *>(base + kQueryOff);
    auto *summary = reinterpret_cast<const __half *>(base + kSummaryOff);
    auto *pre = reinterpret_cast<__half *>(base + kPreOff);
    auto *post = reinterpret_cast<__half *>(base + kPostOff);
    auto *page_ids = reinterpret_cast<uint32_t *>(base + kPageIdOff);
    auto *statuses = reinterpret_cast<nixlGpuXferStatusH *>(base + kStatusOff);
    __shared__ uint32_t selected, bad;
    bool claimed = false, unsafe = false;
    const nixlMemViewElem owner_elem{local_view, 0, kAckOff};
    for (uint32_t step = 0; step < steps; ++step) {
        const uint64_t step_epoch = epoch + step;
        if (threadIdx.x == 0) {
            bad = 0;
            auto *record = reinterpret_cast<ReadyRecord *>(base + kReadyOff);
            uint64_t ready_ns = 0;
            if (!waitReady(&record->epoch,
                           step_epoch,
                           globalTimerNs() + timeout_ns,
                           &ready_ns,
                           &errors[1])) {
                bad = 1;
            } else {
                metrics[step * kMetricStride + 2] = ready_ns;
            }
            if (!bad &&
                (record->epoch_xor != ~step_epoch || record->step != step ||
                 record->step_xor != ~step || record->page_xor != ~record->page ||
                 record->page >= kPages || record->page / kPagesPerStep != step)) {
                errors[1] = 2;
                bad = 1;
            }
            if (!bad) {
                selected = raRoute(q, summary, step);
                if (selected != record->page) {
                    errors[1] = 3;
                    bad = 1;
                }
            }
        }
        __syncthreads();
        if (bad) {
            break;
        }
        const size_t payload_off = kReplyOff + size_t(step) * kReplyStride + kGuardBytes;
        const auto *payload = reinterpret_cast<const uint16_t *>(base + payload_off);
        const auto *guard_before = base + payload_off - kGuardBytes;
        const auto *guard_after = base + payload_off + kPageBytes;
        for (size_t i = threadIdx.x; i < kPageBytes / sizeof(uint16_t); i += blockDim.x) {
            const size_t p = selected;
            const uint16_t expected = i < kHalfPage ?
                raKBits(p, i / kHead, i % kHead) :
                raVBits(p, (i - kHalfPage) / kHead, (i - kHalfPage) % kHead);
            if (payload[i] != expected) {
                atomicExch(&bad, 1u);
            }
        }
        for (size_t i = threadIdx.x; i < kGuardBytes; i += blockDim.x) {
            if (guard_before[i] != kGuard || guard_after[i] != kGuard) {
                atomicExch(&bad, 1u);
            }
        }
        __syncthreads();
        if (bad) {
            if (threadIdx.x == 0) {
                errors[1] = 4;
            }
            __syncthreads();
            break;
        }
        if (threadIdx.x == 0) {
            metrics[step * kMetricStride + 3] = globalTimerNs(); // full-page validation complete
            page_ids[step] = selected;
        }
        __syncthreads();
        raAttention(
            q, reinterpret_cast<const __half *>(payload), pre, post, page_ids, step, selected);
        __syncthreads();
        if (threadIdx.x == 0) {
            metrics[step * kMetricStride + 4] = globalTimerNs(); // attention complete
            auto *ack = reinterpret_cast<AckRecord *>(base + kAckOff);
            ack->epoch = step_epoch + (inject_ack_epoch ? 1u : 0u);
            ack->step = step;
            ack->step_xor = ~step;
            __threadfence_system();
            metrics[step * kMetricStride + 5] = globalTimerNs(); // ACK post begins
            auto &status = statuses[0];
            status = nixlGpuXferStatusH{};
            auto rc = nixlPutDeferred<nixl_gpu_level_t::THREAD>({local_view, 0, kAckOff},
                                                                {remote_view, 0, kAckOff},
                                                                sizeof(AckRecord),
                                                                owner_token,
                                                                &status);
            if (rc != NIXL_IN_PROG) {
                errors[1] = 5;
                bad = 1;
            } else {
                claimed = true;
                rc = nixlGpuCommit(owner_elem, owner_token);
                if (rc != NIXL_SUCCESS) {
                    errors[1] = 6;
                    unsafe = true;
                    bad = 1;
                } else {
                    rc = waitOwnerStatus(status, owner_token, globalTimerNs() + timeout_ns);
                    if (rc == NIXL_IN_PROG) {
                        errors[1] = 7;
                        unsafe = true;
                        bad = 1;
                    } else if (rc != NIXL_SUCCESS) {
                        errors[1] = 8;
                        bad = 1;
                    } else if (inject_ack_epoch) {
                        errors[1] = 10;
                        bad = 1;
                    }
                }
            }
        }
        __syncthreads();
        if (bad) {
            break;
        }
    }
    if (threadIdx.x == 0) {
        if (claimed && !unsafe) {
            const auto rc = nixlGpuReleaseOwner(owner_elem, owner_token);
            if (rc == NIXL_SUCCESS) {
                *safe_to_release = 1;
            } else {
                errors[1] = 9;
                *safe_to_release = 0;
            }
        } else if (!claimed) {
            *safe_to_release = 1;
        }
        if (unsafe) {
            *safe_to_release = 0;
        }
    }
}

// The following CPU references mirror remote_attention_pilot.cu. They execute
// only after GPU timing has completed (except source-side page-schedule setup).
std::vector<float>
raCpuAttention(const std::vector<uint16_t> &q, uint32_t page) {
    std::vector<float> scores(kSeq), weights(kSeq), out(kHead);
    float max_score = -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < kSeq; ++t) {
        float dot = 0;
        for (size_t d = 0; d < kHead; ++d) {
            dot += __half2float(__ushort_as_half(q[d])) *
                __half2float(__ushort_as_half(raKBits(page, t, d)));
        }
        scores[t] = dot * 0.08838834764831845f;
        max_score = std::max(max_score, scores[t]);
    }
    float sum = 0;
    for (size_t t = 0; t < kSeq; ++t) {
        weights[t] = std::exp(scores[t] - max_score);
        sum += weights[t];
    }
    for (float &w : weights) {
        w /= sum;
    }
    for (size_t d = 0; d < kHead; ++d) {
        float v = 0;
        for (size_t t = 0; t < kSeq; ++t) {
            v += weights[t] * __half2float(__ushort_as_half(raVBits(page, t, d)));
        }
        out[d] = 0.5f * __half2float(__ushort_as_half(q[d])) + 0.5f * v;
    }
    return out;
}

uint32_t
raCpuRoute(const std::vector<uint16_t> &q, const std::vector<uint16_t> &summary, uint32_t step) {
    float best_score = -std::numeric_limits<float>::infinity();
    uint32_t best_page = UINT32_MAX;
    for (uint32_t c = 0; c < 16; ++c) {
        const uint32_t page = step * kPagesPerStep + (31u * c + 17u * step + 3u) % kPagesPerStep;
        float score = 0;
        for (size_t d = 0; d < kHead; ++d) {
            score += __half2float(__ushort_as_half(q[d])) *
                __half2float(__ushort_as_half(summary[page * kHead + d]));
        }
        if (score > best_score || (score == best_score && page < best_page)) {
            best_score = score;
            best_page = page;
        }
    }
    return best_page;
}

std::vector<uint16_t>
raCpuInitialQuery(uint64_t epoch) {
    std::vector<uint16_t> q(kHead);
    for (size_t d = 0; d < kHead; ++d) {
        const int x = static_cast<int>((epoch * 19 + d * 23 + 7) % 97) - 48;
        q[d] = __half_as_ushort(__float2half_rn(static_cast<float>(x) / 48.0f));
    }
    return q;
}

std::vector<uint32_t>
makePageSchedule(uint64_t epoch, uint32_t steps, const std::vector<uint16_t> &summary) {
    auto q = raCpuInitialQuery(epoch);
    std::vector<uint32_t> pages;
    pages.reserve(steps);
    for (uint32_t s = 0; s < steps; ++s) {
        const auto page = raCpuRoute(q, summary, s);
        pages.push_back(page);
        const auto y = raCpuAttention(q, page);
        for (size_t d = 0; d < kHead; ++d) {
            q[d] = __half_as_ushort(__float2half_rn(y[d]));
        }
    }
    return pages;
}

std::string
configText(const Args &a) {
    return "epoch=" + std::to_string(a.epoch) + "\nsteps=" + std::to_string(a.steps) +
        "\narm=" + a.arm + "\nfirst_arm=" + a.first_arm + "\ninject=" + a.inject +
        "\nepisodes=" + std::to_string(a.episodes) + "\nwarmups=" + std::to_string(a.warmups) +
        "\npage_bytes=" + std::to_string(kPageBytes) + "\n";
}

size_t
armIndex(const std::string &arm) {
    return arm == "ordered" ? 0 : 1;
}

uint64_t
nearestRank(std::vector<uint64_t> values, uint32_t percentile) {
    if (values.empty()) {
        return 0;
    }
    std::sort(values.begin(), values.end());
    const size_t rank = (size_t(percentile) * values.size() + 99) / 100;
    return values[std::max<size_t>(1, rank) - 1];
}

void
printPercentiles(const char *prefix, const std::array<std::vector<uint64_t>, 2> &samples) {
    for (size_t arm = 0; arm < samples.size(); ++arm) {
        if (samples[arm].empty()) {
            continue;
        }
        const char *name = arm == 0 ? "ordered" : "cqwait";
        std::cout << prefix << " arm=" << name << " samples=" << samples[arm].size()
                  << " p50_ns=" << nearestRank(samples[arm], 50)
                  << " p99_ns=" << nearestRank(samples[arm], 99) << '\n';
    }
}

struct ReceiverSamples {
    std::array<std::vector<uint64_t>, 2> payload_check, attention, ack_post;
};

void
printReceiverPercentiles(const ReceiverSamples &samples) {
    for (size_t arm = 0; arm < 2; ++arm) {
        if (samples.attention[arm].empty()) {
            continue;
        }
        const char *name = arm == 0 ? "ordered" : "cqwait";
        std::cout << "REMOTE_ATTENTION_CONSUMER_P99 arm=" << name
                  << " samples=" << samples.attention[arm].size()
                  << " ready_to_payload_check_p50_ns="
                  << nearestRank(samples.payload_check[arm], 50)
                  << " ready_to_payload_check_p99_ns="
                  << nearestRank(samples.payload_check[arm], 99)
                  << " ready_to_attention_p50_ns=" << nearestRank(samples.attention[arm], 50)
                  << " ready_to_attention_p99_ns=" << nearestRank(samples.attention[arm], 99)
                  << " ready_to_ack_post_p50_ns=" << nearestRank(samples.ack_post[arm], 50)
                  << " ready_to_ack_post_p99_ns=" << nearestRank(samples.ack_post[arm], 99) << '\n';
    }
}

struct Endpoint {
    std::unique_ptr<nixlAgent> agent;
    nixlBackendH *backend = nullptr;
    nixlMemViewH local = nullptr, remote = nullptr;
    nixl_reg_dlist_t registration{VRAM_SEG};
    nixl_opt_args_t hint;
};

void
setupEndpoint(const Args &a, DeviceArena &arena, const std::string &role, Endpoint &ep) {
    nixlAgentConfig cfg;
    cfg.useProgThread = true;
    cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    ep.agent = std::make_unique<nixlAgent>("ra_consumer_" + role + "_" + a.run, cfg);
    nixl_b_params_t params;
    params["native_device_api"] = "true";
    params["native_device_owner"] = "true";
    params["gpu_devices"] = std::to_string(a.gpu);
    if (!a.rdma.empty()) {
        params["network_devices"] = a.rdma;
    }
    if (!a.oob_interface.empty()) {
        params["oob_interface"] = a.oob_interface;
    }
    if (a.gid) {
        params["gid_index"] = std::to_string(a.gid);
    }
    if (a.oob_port) {
        params["oob_port"] = std::to_string(a.oob_port);
    }
    nixlCheck(ep.agent->createBackend("GPUNETIO", params, ep.backend), "create GPUNETIO backend");
    if (!ep.backend) {
        throw std::runtime_error("createBackend returned null");
    }
    ep.hint.backends.push_back(ep.backend);
    ep.registration.addDesc({reinterpret_cast<uintptr_t>(arena.get()), arena.size(), a.gpu, {}});
    nixlCheck(ep.agent->registerMem(ep.registration, &ep.hint), "register VRAM arena");
}

void
publishEndpoint(const Args &a, Endpoint &ep, DeviceArena &arena, const std::string &role) {
    nixl_blob_t md;
    nixlCheck(ep.agent->getLocalMD(md), "getLocalMD");
    publishFile(filePath(a, role, ".md"), md);
    publishFile(filePath(a, role, ".desc"),
                std::to_string(reinterpret_cast<uintptr_t>(arena.get())) + " " +
                    std::to_string(arena.size()) + " " + std::to_string(a.gpu));
}

void
loadPeerAndViews(const Args &a, Endpoint &ep, DeviceArena &arena, const std::string &peer_role) {
    std::string peer;
    nixlCheck(ep.agent->loadRemoteMD(readFile(filePath(a, peer_role, ".md")), peer),
              "loadRemoteMD");
    nixlCheck(ep.agent->makeConnection(peer, &ep.hint), "makeConnection");
    const auto [address, bytes, peer_gpu] = parseDesc(readFile(filePath(a, peer_role, ".desc")));
    nixl_local_dlist_t local(VRAM_SEG);
    local.addDesc({reinterpret_cast<uintptr_t>(arena.get()), arena.size(), a.gpu});
    nixl_remote_dlist_t remote(VRAM_SEG);
    remote.addDesc({address, bytes, peer_gpu, peer});
    nixlCheck(ep.agent->prepMemView(local, ep.local, &ep.hint), "prep local mem view");
    nixlCheck(ep.agent->prepMemView(remote, ep.remote, &ep.hint), "prep remote mem view");
    if (!ep.local || !ep.remote) {
        throw std::runtime_error("prepMemView returned null");
    }
}

void
checkReceiverVisibility() {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
    CUdevice dev{};
    int ordering = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuCtxGetDevice(&dev) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, dev) !=
            CUDA_SUCCESS) {
        throw std::runtime_error("cannot query GPUDirect write ordering");
    }
    if (ordering < CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER) {
        throw std::runtime_error(
            "consumer requires OWNER-ordered GPUDirect writes for GPU system acquire");
    }
#else
    throw std::runtime_error("CUDA headers lack OWNER write-ordering attribute");
#endif
}

void
initializeStaticArena(const Args &a,
                      DeviceArena &arena,
                      const std::vector<__half> &table,
                      const std::vector<uint16_t> &summary) {
    auto *base = arena.get();
    cudaCheck(cudaMemset(base, 0, arena.size()), "zero arena");
    if (a.role == "source") {
        cudaCheck(cudaMemcpy(base + kTableOff, table.data(), kTableBytes, cudaMemcpyHostToDevice),
                  "copy page table");
    } else {
        cudaCheck(
            cudaMemcpy(base + kSummaryOff, summary.data(), kSummaryBytes, cudaMemcpyHostToDevice),
            "copy routing summary");
    }
    cudaCheck(cudaDeviceSynchronize(), "initialize arena sync");
}

void
initializeEpisode(const Args &a,
                  DeviceArena &arena,
                  uint64_t epoch,
                  const std::vector<uint32_t> &pages) {
    auto *base = arena.get();
    cudaCheck(cudaMemset(base + kReadyOff, 0, 64), "reset ready record");
    cudaCheck(cudaMemset(base + kAckOff, 0, 64), "reset ACK record");
    cudaCheck(cudaMemset(base + kStatusOff, 0, 4 * sizeof(nixlGpuXferStatusH)),
              "reset transfer statuses");
    cudaCheck(cudaMemset(base + kMetricOff, 0, kMaxSteps * kMetricStride * sizeof(uint64_t)),
              "reset GPU metrics");
    cudaCheck(cudaMemset(base + kErrorOff, 0, 4 * sizeof(uint32_t)), "reset episode errors");
    cudaCheck(cudaMemset(base + kPageIdOff, 0, kMaxSteps * sizeof(uint32_t)), "reset page IDs");
    if (a.role == "source") {
        cudaCheck(cudaMemcpy(base + kPageIdOff,
                             pages.data(),
                             pages.size() * sizeof(uint32_t),
                             cudaMemcpyHostToDevice),
                  "copy episode route schedule");
    } else {
        cudaCheck(cudaMemset(base + kPreOff, 0, kMaxSteps * kHead * sizeof(__half)), "reset qpre");
        cudaCheck(cudaMemset(base + kPostOff, 0, kMaxSteps * kHead * sizeof(__half)),
                  "reset qpost");
        const auto q0 = raCpuInitialQuery(epoch);
        cudaCheck(
            cudaMemcpy(
                base + kQueryOff, q0.data(), q0.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "copy episode initial query");
        for (uint32_t s = 0; s < a.steps; ++s) {
            const size_t payload = kReplyOff + size_t(s) * kReplyStride + kGuardBytes;
            cudaCheck(cudaMemset(base + payload - kGuardBytes, kGuard, kGuardBytes),
                      "reset leading page guard");
            cudaCheck(cudaMemset(base + payload + kPageBytes, kGuard, kGuardBytes),
                      "reset trailing page guard");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "initialize episode sync");
}

bool
validateCpu(const Args &a,
            DeviceArena &arena,
            uint64_t epoch,
            const std::vector<uint16_t> &summary,
            uint32_t *gpu_errors_out,
            double *max_error_out) {
    auto *base = arena.get();
    std::vector<uint16_t> pre(size_t(a.steps) * kHead), post(size_t(a.steps) * kHead);
    std::vector<uint32_t> ids(a.steps);
    std::vector<uint8_t> pages(size_t(a.steps) * kPageBytes);
    cudaCheck(
        cudaMemcpy(gpu_errors_out, base + kErrorOff, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "copy device errors");
    cudaCheck(
        cudaMemcpy(
            pre.data(), base + kPreOff, pre.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qpre");
    cudaCheck(
        cudaMemcpy(
            post.data(), base + kPostOff, post.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
        "copy qpost");
    cudaCheck(
        cudaMemcpy(
            ids.data(), base + kPageIdOff, ids.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
        "copy page ids");
    for (uint32_t s = 0; s < a.steps; ++s) {
        const size_t off = kReplyOff + size_t(s) * kReplyStride + kGuardBytes;
        cudaCheck(cudaMemcpy(pages.data() + size_t(s) * kPageBytes,
                             base + off,
                             kPageBytes,
                             cudaMemcpyDeviceToHost),
                  "copy received page");
        std::array<uint8_t, kGuardBytes> g0{}, g1{};
        cudaCheck(
            cudaMemcpy(g0.data(), base + off - kGuardBytes, kGuardBytes, cudaMemcpyDeviceToHost),
            "copy leading guard");
        cudaCheck(
            cudaMemcpy(g1.data(), base + off + kPageBytes, kGuardBytes, cudaMemcpyDeviceToHost),
            "copy trailing guard");
        if (!std::all_of(g0.begin(), g0.end(), [](uint8_t x) { return x == kGuard; }) ||
            !std::all_of(g1.begin(), g1.end(), [](uint8_t x) { return x == kGuard; })) {
            ++gpu_errors_out[1];
        }
        for (size_t i = 0; i < kPageBytes / sizeof(uint16_t); ++i) {
            const size_t p = ids[s];
            const uint16_t expected = i < kHalfPage ?
                raKBits(p, i / kHead, i % kHead) :
                raVBits(p, (i - kHalfPage) / kHead, (i - kHalfPage) % kHead);
            if (reinterpret_cast<const uint16_t *>(pages.data() + size_t(s) * kPageBytes)[i] !=
                expected) {
                ++gpu_errors_out[1];
            }
        }
    }
    auto q = raCpuInitialQuery(epoch);
    uint32_t chain_errors = 0;
    double max_error = 0;
    for (uint32_t s = 0; s < a.steps; ++s) {
        if (ids[s] >= kPages || raCpuRoute(q, summary, s) != ids[s]) {
            ++chain_errors;
        }
        for (size_t d = 0; d < kHead; ++d) {
            if (pre[size_t(s) * kHead + d] != q[d]) {
                ++chain_errors;
            }
        }
        const auto expected = raCpuAttention(q, ids[s]);
        for (size_t d = 0; d < kHead; ++d) {
            const auto actual = __half2float(__ushort_as_half(post[size_t(s) * kHead + d]));
            max_error = std::max(max_error, static_cast<double>(std::abs(expected[d] - actual)));
        }
        q.assign(post.begin() + size_t(s) * kHead, post.begin() + size_t(s + 1) * kHead);
    }
    *max_error_out = max_error;
    gpu_errors_out[1] += chain_errors;
    return gpu_errors_out[0] == 0 && gpu_errors_out[1] == 0 && max_error <= kTolerance;
}

void
cleanup(Endpoint &ep, const Args &a) {
    if (ep.agent) {
        if (ep.local) {
            ep.agent->releaseMemView(ep.local);
            ep.local = nullptr;
        }
        if (ep.remote) {
            ep.agent->releaseMemView(ep.remote);
            ep.remote = nullptr;
        }
        nixlCheck(ep.agent->deregisterMem(ep.registration, &ep.hint), "deregister VRAM arena");
        ep.agent.reset();
    }
    (void)a;
}

} // namespace

int
main(int argc, char **argv) {
    Args args;
    try {
        args = parseArgs(argc, argv);
    }
    catch (const std::exception &e) {
        usage(e.what());
    }

    bool actors_launched = false;
    try {
        if (!std::filesystem::is_directory(args.coord)) {
            throw std::runtime_error("--coord must be an existing shared directory");
        }
        cudaCheck(cudaSetDevice(static_cast<int>(args.gpu)), "cudaSetDevice");
        cudaCheck(cudaFree(nullptr), "initialize CUDA context");
        auto summary = raMakeSummary();
        std::vector<__half> table;
        if (args.role == "source") {
            table = raMakeTable();
        }
        DeviceArena arena(kArenaBytes);
        initializeStaticArena(args, arena, table, summary);
        if (args.role == "consumer") {
            checkReceiverVisibility();
        }

        Endpoint ep;
        setupEndpoint(args, arena, args.role, ep);
        publishEndpoint(args, ep, arena, args.role);

        Control control(args);
        if (args.role == "consumer") {
            control.send('M');
            control.expect('C');
            if (control.receiveText() != configText(args)) {
                throw std::runtime_error("source/consumer epoch, steps, or arm differ");
            }
            control.send('S');
            loadPeerAndViews(args, ep, arena, "source");
            control.send('T');
            control.expect('T');
        } else {
            control.expect('M');
            control.send('C');
            control.sendText(configText(args));
            control.expect('S');
            loadPeerAndViews(args, ep, arena, "consumer");
            control.send('T');
            control.expect('T');
        }

        cudaStream_t stream{};
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        uint32_t *safe = nullptr;
        cudaCheck(cudaMalloc(&safe, sizeof(uint32_t)), "allocate safe-release flag");
        const uint64_t timeout_ns = args.timeout_ms * 1'000'000ull;
        std::array<std::vector<uint64_t>, 2> source_rtt;
        std::array<std::vector<uint64_t>, 2> source_step_intervals;
        ReceiverSamples receiver_samples;
        const uint64_t total_episodes = episodeCount(args);
        if (args.role == "consumer") {
            for (uint64_t i = 0; i < total_episodes; ++i) {
                const Episode episode = episodeAt(args, i);
                initializeEpisode(args, arena, episode.epoch, {});
                // Actor stream is nonblocking: order this reset on that stream,
                // then synchronize before either actor launch or host-ready R.
                cudaCheck(cudaMemsetAsync(safe, 0, sizeof(uint32_t), stream),
                          "reset consumer release gate");
                cudaCheck(cudaStreamSynchronize(stream),
                          "sync consumer release-gate reset before ready");
                consumerActor<<<1, 256, 0, stream>>>(ep.local,
                                                     ep.remote,
                                                     arena.get(),
                                                     args.steps,
                                                     episode.epoch,
                                                     episode.owner_token,
                                                     timeout_ns,
                                                     safe,
                                                     args.inject == "ack-epoch");
                cudaCheck(cudaGetLastError(), "launch consumer episode actor");
                actors_launched = true;
                control.send('R');
                const auto sync = cudaStreamSynchronize(stream);
                if (sync != cudaSuccess) {
                    throw std::runtime_error(std::string("consumer stream failed: ") +
                                             cudaGetErrorString(sync));
                }
                uint32_t safe_host = 0;
                cudaCheck(cudaMemcpy(&safe_host, safe, sizeof(safe_host), cudaMemcpyDeviceToHost),
                          "read consumer release gate");
                if (!safe_host) {
                    control.expect('D');
                    control.send('F');
                    std::cerr << "unsafe owner/view teardown; retaining registrations until "
                                 "process exit\n";
                    std::_Exit(10);
                }
                actors_launched = false;
                control.expect('D');
                uint32_t errors[2]{};
                double max_error = 0;
                const bool ok =
                    validateCpu(args, arena, episode.epoch, summary, errors, &max_error);
                uint64_t metrics[kMaxSteps * kMetricStride]{};
                cudaCheck(cudaMemcpy(metrics,
                                     arena.get() + kMetricOff,
                                     args.steps * kMetricStride * sizeof(uint64_t),
                                     cudaMemcpyDeviceToHost),
                          "copy receiver episode metrics");
                bool phases_valid = true;
                const size_t arm = armIndex(episode.arm);
                std::cout << "REMOTE_ATTENTION_CONSUMER_EPISODE episode=" << i
                          << " pair=" << episode.pair << " arm=" << episode.arm
                          << " sample=" << (episode.warmup ? "warmup" : "measured")
                          << " epoch=" << episode.epoch << " lease_token=" << episode.owner_token
                          << " steps=" << args.steps << " cpu_errors=" << errors[0] << ','
                          << errors[1] << " max_attention_error=" << max_error
                          << " receiver_clock=globaltimer_ns ready_to_payload_check_ns=";
                for (uint32_t s = 0; s < args.steps; ++s) {
                    const auto ready = metrics[s * kMetricStride + 2];
                    const auto checked = metrics[s * kMetricStride + 3];
                    const bool valid = ready && checked >= ready;
                    phases_valid &= valid;
                    std::cout << (s ? "," : "") << (valid ? std::to_string(checked - ready) : "NA");
                    if (!episode.warmup && valid) {
                        receiver_samples.payload_check[arm].push_back(checked - ready);
                    }
                }
                std::cout << " ready_to_attention_ns=";
                for (uint32_t s = 0; s < args.steps; ++s) {
                    const auto ready = metrics[s * kMetricStride + 2];
                    const auto done = metrics[s * kMetricStride + 4];
                    const bool valid = ready && done >= ready;
                    phases_valid &= valid;
                    std::cout << (s ? "," : "") << (valid ? std::to_string(done - ready) : "NA");
                    if (!episode.warmup && valid) {
                        receiver_samples.attention[arm].push_back(done - ready);
                    }
                }
                std::cout << " ready_to_ack_post_ns=";
                for (uint32_t s = 0; s < args.steps; ++s) {
                    const auto ready = metrics[s * kMetricStride + 2];
                    const auto ack = metrics[s * kMetricStride + 5];
                    const bool valid = ready && ack >= ready;
                    phases_valid &= valid;
                    std::cout << (s ? "," : "") << (valid ? std::to_string(ack - ready) : "NA");
                    if (!episode.warmup && valid) {
                        receiver_samples.ack_post[arm].push_back(ack - ready);
                    }
                }
                const bool episode_ok = ok && phases_valid;
                control.send(episode_ok ? 'V' : 'F');
                std::cout << " validation=" << (episode_ok ? "PASS" : "FAIL") << '\n';
                if (!episode_ok) {
                    cudaCheck(cudaStreamDestroy(stream), "destroy stream");
                    cudaCheck(cudaFree(safe), "free safe flag");
                    cleanup(ep, args);
                    return 1;
                }
            }
            printReceiverPercentiles(receiver_samples);
        } else {
            for (uint64_t i = 0; i < total_episodes; ++i) {
                const Episode episode = episodeAt(args, i);
                const auto pages = makePageSchedule(episode.epoch, args.steps, summary);
                initializeEpisode(args, arena, episode.epoch, pages);
                cudaCheck(cudaMemsetAsync(safe, 0, sizeof(uint32_t), stream),
                          "reset source release gate");
                cudaCheck(cudaStreamSynchronize(stream),
                          "sync source release-gate reset before actor");
                control.expect('R');
                sourceActor<<<1, 1, 0, stream>>>(ep.local,
                                                 ep.remote,
                                                 arena.get(),
                                                 args.steps,
                                                 episode.epoch,
                                                 episode.owner_token,
                                                 timeout_ns,
                                                 episode.arm == "cqwait",
                                                 safe,
                                                 args.inject == "ready-epoch",
                                                 args.inject == "payload");
                cudaCheck(cudaGetLastError(), "launch source episode actor");
                actors_launched = true;
                const auto sync = cudaStreamSynchronize(stream);
                if (sync != cudaSuccess) {
                    throw std::runtime_error(std::string("source stream failed: ") +
                                             cudaGetErrorString(sync));
                }
                uint32_t safe_host = 0, error = 0;
                cudaCheck(cudaMemcpy(&safe_host, safe, sizeof(safe_host), cudaMemcpyDeviceToHost),
                          "read source release gate");
                cudaCheck(
                    cudaMemcpy(
                        &error, arena.get() + kErrorOff, sizeof(error), cudaMemcpyDeviceToHost),
                    "read source error");
                uint64_t metrics[kMaxSteps * kMetricStride]{};
                cudaCheck(cudaMemcpy(metrics,
                                     arena.get() + kMetricOff,
                                     args.steps * kMetricStride * sizeof(uint64_t),
                                     cudaMemcpyDeviceToHost),
                          "copy source episode metrics");
                if (!safe_host) {
                    control.send('D');
                    (void)control.receive();
                    std::cerr << "unsafe owner/view teardown; retaining registrations until "
                                 "process exit\n";
                    std::_Exit(10);
                }
                actors_launched = false;
                control.send('D');
                const char consumer_result = control.receive();
                bool samples_valid = true;
                bool step_samples_valid = true;
                std::vector<uint64_t> step_intervals;
                step_intervals.reserve(args.steps - 1);
                const size_t arm = armIndex(episode.arm);
                std::cout << "REMOTE_ATTENTION_SOURCE_EPISODE episode=" << i
                          << " pair=" << episode.pair << " arm=" << episode.arm
                          << " sample=" << (episode.warmup ? "warmup" : "measured")
                          << " epoch=" << episode.epoch << " lease_token=" << episode.owner_token
                          << " steps=" << args.steps << " source_data_wqes=" << 2 * args.steps
                          << " source_commits=" << 2 * args.steps << " source_error=" << error
                          << " source_clock=globaltimer_ns post_to_ack_rtt_ns=";
                for (uint32_t s = 0; s < args.steps; ++s) {
                    const uint64_t start = metrics[s * kMetricStride];
                    const uint64_t ack = metrics[s * kMetricStride + 1];
                    const bool valid = start && ack >= start && ack != 0;
                    samples_valid &= valid;
                    const uint64_t rtt = valid ? ack - start : 0;
                    std::cout << (s ? "," : "") << (valid ? std::to_string(rtt) : "NA");
                    if (!episode.warmup && valid) {
                        source_rtt[arm].push_back(rtt);
                    }
                }
                if (!episode.warmup) {
                    for (uint32_t s = 0; s + 1 < args.steps; ++s) {
                        const uint64_t current = metrics[s * kMetricStride];
                        const uint64_t next = metrics[(s + 1) * kMetricStride];
                        const bool valid = current && next > current;
                        step_samples_valid &= valid;
                        if (valid) {
                            step_intervals.push_back(next - current);
                        }
                    }
                }
                const bool episode_ok =
                    error == 0 && samples_valid && step_samples_valid && consumer_result == 'V';
                std::cout << " validation=" << (episode_ok ? "PASS" : "FAIL") << '\n';
                if (!episode_ok) {
                    cudaCheck(cudaStreamDestroy(stream), "destroy stream");
                    cudaCheck(cudaFree(safe), "free safe flag");
                    cleanup(ep, args);
                    return 1;
                }
                if (!episode.warmup) {
                    std::cout << "REMOTE_ATTENTION_SOURCE_STEP_INTERVAL episode=" << i
                              << " pair=" << episode.pair << " arm=" << episode.arm
                              << " sample=measured steps=" << args.steps
                              << " samples=" << step_intervals.size() << " post_to_next_post_ns=";
                    for (size_t s = 0; s < step_intervals.size(); ++s) {
                        std::cout << (s ? "," : "") << step_intervals[s];
                    }
                    std::cout << '\n';
                    source_step_intervals[arm].insert(source_step_intervals[arm].end(),
                                                      step_intervals.begin(),
                                                      step_intervals.end());
                }
            }
            printPercentiles("REMOTE_ATTENTION_SOURCE_P99", source_rtt);
            printPercentiles("REMOTE_ATTENTION_SOURCE_STEP_P99", source_step_intervals);
        }

        const size_t expected_samples = size_t(args.episodes) * args.steps;
        for (size_t arm = 0; arm < 2; ++arm) {
            const bool selected = args.arm == "paired" || (args.arm == "ordered") == (arm == 0);
            if (args.role == "source") {
                const size_t expected = selected ? expected_samples : 0;
                const size_t expected_steps =
                    selected ? size_t(args.episodes) * (args.steps - 1) : 0;
                if (source_rtt[arm].size() != expected ||
                    source_step_intervals[arm].size() != expected_steps) {
                    throw std::runtime_error(
                        "source arm produced an unexpected measured sample count");
                }
            } else {
                const size_t expected = selected ? expected_samples : 0;
                if (receiver_samples.attention[arm].size() != expected ||
                    receiver_samples.payload_check[arm].size() != expected ||
                    receiver_samples.ack_post[arm].size() != expected) {
                    throw std::runtime_error(
                        "consumer arm produced an unexpected measured sample count");
                }
            }
        }
        cudaCheck(cudaStreamDestroy(stream), "destroy stream");
        cudaCheck(cudaFree(safe), "free safe flag");
        cleanup(ep, args);
        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "REMOTE_ATTENTION_HARNESS_FAIL: " << e.what() << '\n';
        if (actors_launched) {
            std::cerr << "actors launched; skipping destructors to avoid releasing buffers/views "
                         "with unknown in-flight state\n";
            std::_Exit(10);
        }
        return 1;
    }
}
