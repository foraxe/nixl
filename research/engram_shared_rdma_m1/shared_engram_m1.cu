// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gpu/nixl_device.cuh>
#include <nixl.h>

#include "../gpu_remote_kv/experimental_get.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr uint32_t kRows = 256;
constexpr uint32_t kWidth = 256;
constexpr uint32_t kScaleGroups = 8;
constexpr uint32_t kScaleGroupWidth = 32;
constexpr size_t kPayloadBytes = size_t(kRows) * kWidth;
constexpr size_t kScaleBytes = size_t(kRows) * kScaleGroups;
constexpr size_t kArenaBytes = 256 * 1024;
constexpr size_t kPayloadGuard = 64;
constexpr size_t kPayloadOffset = 64;
constexpr size_t kPayloadTail = kPayloadOffset + kWidth;
constexpr size_t kScaleGuard = 384;
constexpr size_t kScaleOffset = 448;
constexpr size_t kScaleTail = 456;
constexpr size_t kDumpOffset = 224 * 1024;
constexpr uint8_t kGuard = 0xa5;
constexpr auto kTimeout = std::chrono::minutes(3);

void ck(cudaError_t rc, const char *where) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(rc));
}
void require(nixl_status_t rc, const char *where) {
    if (rc != NIXL_SUCCESS) throw std::runtime_error(std::string(where) + ": NIXL status " + std::to_string(rc));
}
std::string env(const char *key, std::string fallback = {}) {
    const char *value = std::getenv(key);
    return value ? value : std::move(fallback);
}
void atomicWrite(const fs::path &path, const std::string &data) {
    fs::create_directories(path.parent_path());
    auto tmp = path.string() + ".tmp." + std::to_string(getpid());
    { std::ofstream out(tmp, std::ios::binary); out.write(data.data(), data.size()); out.flush();
      if (!out) throw std::runtime_error("write failed: " + tmp); }
    fs::rename(tmp, path);
}
std::string readAll(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

uint8_t codeFor(uint32_t row, uint32_t col) {
    const uint8_t sign = ((row * 13u + col * 7u) & 1u) ? 0x80 : 0;
    const uint8_t exponent = static_cast<uint8_t>((row * 3u + col / 5u) % 15u);
    uint8_t mantissa = static_cast<uint8_t>((row * 5u + col * 3u) & 7u);
    if (exponent == 15 && mantissa == 7) mantissa = 6; // E4M3FN finite range.
    return static_cast<uint8_t>(sign | (exponent << 3) | mantissa);
}
uint8_t scaleFor(uint32_t row, uint32_t group) {
    return static_cast<uint8_t>(120u + ((row + group * 3u) % 15u)); // E8M0 exponent bytes.
}
uint32_t rowId(uint32_t i, uint32_t seed) {
    switch (i & 7u) {
    case 0: return 0;
    case 1: return kRows - 1;
    case 2: return kRows / 2;
    case 3: return 0; // repeated row
    default: {
        uint32_t x = seed + i * 0x9e3779b9u;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        return x % kRows;
    }
    }
}
float decodeE4M3(uint8_t bits) {
    const float sign = (bits & 0x80) ? -1.0f : 1.0f;
    const uint8_t e = (bits >> 3) & 0x0f;
    const uint8_t m = bits & 7;
    if (e == 0) return sign * static_cast<float>(m) * 0x1p-9f;
    if (e == 15 && m == 7) return NAN;
    return sign * (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, int(e) - 7);
}
uint16_t toBf16Bits(float value) {
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}
uint16_t expectedBf16(uint32_t row, uint32_t col) {
    const float code = decodeE4M3(codeFor(row, col));
    const float scale = std::ldexp(1.0f, int(scaleFor(row, col / kScaleGroupWidth)) - 127);
    return toBf16Bits(code * scale);
}

__device__ float decodeE4M3Device(uint8_t bits) {
    const float sign = (bits & 0x80) ? -1.0f : 1.0f;
    const uint8_t e = (bits >> 3) & 0x0f;
    const uint8_t m = bits & 7;
    if (e == 0) return sign * static_cast<float>(m) * 0x1p-9f;
    if (e == 15 && m == 7) return __int_as_float(0x7fc00000);
    return sign * (1.0f + static_cast<float>(m) / 8.0f) * ldexpf(1.0f, int(e) - 7);
}
__device__ uint8_t codeForDevice(uint32_t row, uint32_t col) {
    const uint8_t sign = ((row * 13u + col * 7u) & 1u) ? 0x80 : 0;
    const uint8_t exponent = static_cast<uint8_t>((row * 3u + col / 5u) % 15u);
    uint8_t mantissa = static_cast<uint8_t>((row * 5u + col * 3u) & 7u);
    if (exponent == 15 && mantissa == 7) mantissa = 6;
    return static_cast<uint8_t>(sign | (exponent << 3) | mantissa);
}
__device__ uint8_t scaleForDevice(uint32_t row, uint32_t group) {
    return static_cast<uint8_t>(120u + ((row + group * 3u) % 15u));
}
__global__ void makeIds(uint32_t *ids, uint32_t count, uint32_t seed) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    switch (i & 7u) {
    case 0: ids[i] = 0; break;
    case 1: ids[i] = kRows - 1; break;
    case 2: ids[i] = kRows / 2; break;
    case 3: ids[i] = 0; break;
    default: {
        uint32_t x = seed + i * 0x9e3779b9u;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        ids[i] = x % kRows;
    }
    }
}
__global__ void decodeLocalReference(const uint8_t *payload, const uint8_t *scales,
                                    uint32_t count, __nv_bfloat16 *out) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = count * kWidth;
    if (index >= total) return;
    const uint32_t q = index / kWidth;
    const uint32_t col = index % kWidth;
    const uint8_t scale = scales[size_t(q) * kScaleGroups + col / kScaleGroupWidth];
    out[index] = __float2bfloat16(decodeE4M3Device(payload[index]) * ldexpf(1.0f, int(scale) - 127));
}
__device__ bool getOne(const nixl::gpu::memViewElem &local,
                       const nixl::gpu::memViewElem &remote,
                       const nixl::gpu::memViewElem &dump,
                       size_t bytes) {
    nixl::gpu::xferStatusH status{};
    auto rc = nixl::gpu::experimental::get(local, remote, bytes, dump, &status);
    if (rc != NIXL_IN_PROG) return false;
    const uint64_t deadline = clock64() + 12000000000ull;
    do {
        rc = nixl::gpu::getXferStatus<nixl::gpu::level_t::THREAD>(status);
        if (rc == NIXL_SUCCESS) return true;
        if (rc != NIXL_IN_PROG || clock64() >= deadline) return false;
    } while (true);
}
__global__ void fetchAndDecode(uint32_t *ids, uint32_t count, uint8_t *arena,
                               __nv_bfloat16 *out, nixlMemViewH local_view,
                               nixlMemViewH remote_view, uint32_t *errors) {
    for (uint32_t q = 0; q < count; ++q) {
        const uint32_t row = ids[q];
        if (threadIdx.x == 0) {
            const nixl::gpu::memViewElem dump{local_view, 0, kDumpOffset};
            const nixl::gpu::memViewElem p_dst{local_view, 0, kPayloadOffset};
            const nixl::gpu::memViewElem p_src{remote_view, 0, size_t(row) * kWidth};
            const nixl::gpu::memViewElem s_dst{local_view, 0, kScaleOffset};
            const nixl::gpu::memViewElem s_src{remote_view, 1, size_t(row) * kScaleGroups};
            if (!getOne(p_dst, p_src, dump, kWidth) ||
                !getOne(s_dst, s_src, dump, kScaleGroups)) atomicExch(errors, 1u);
        }
        __syncthreads();
        for (uint32_t col = threadIdx.x; col < kWidth; col += blockDim.x) {
            const uint8_t got = arena[kPayloadOffset + col];
            const uint8_t expected = codeForDevice(row, col);
            if (got != expected) atomicAdd(errors, 1u);
            const uint8_t scale = arena[kScaleOffset + col / kScaleGroupWidth];
            if (scale != scaleForDevice(row, col / kScaleGroupWidth)) atomicAdd(errors, 1u);
            const int exponent = int(scale) - 127;
            out[size_t(q) * kWidth + col] = __float2bfloat16(decodeE4M3Device(got) * ldexpf(1.0f, exponent));
        }
        for (uint32_t i = threadIdx.x; i < kPayloadGuard; i += blockDim.x) {
            if (arena[i] != kGuard || arena[kPayloadTail + i] != kGuard ||
                arena[kScaleGuard + i] != kGuard || arena[kScaleTail + i] != kGuard)
                atomicAdd(errors, 1u);
        }
        __syncthreads(); // Do not overwrite row scratch until every output lane consumed it.
    }
}

struct HostTable {
    uint8_t *payload = nullptr;
    uint8_t *scales = nullptr;
    ~HostTable() { if (payload) cudaFreeHost(payload); if (scales) cudaFreeHost(scales); }
    void init() {
        ck(cudaHostAlloc(reinterpret_cast<void **>(&payload), kPayloadBytes, cudaHostAllocPortable), "pin payload");
        ck(cudaHostAlloc(reinterpret_cast<void **>(&scales), kScaleBytes, cudaHostAllocPortable), "pin scales");
        for (uint32_t r = 0; r < kRows; ++r) {
            for (uint32_t c = 0; c < kWidth; ++c) payload[size_t(r) * kWidth + c] = codeFor(r, c);
            for (uint32_t g = 0; g < kScaleGroups; ++g) scales[size_t(r) * kScaleGroups + g] = scaleFor(r, g);
        }
    }
};
struct DeviceArena {
    uint8_t *arena = nullptr;
    uint32_t *ids = nullptr;
    uint32_t *errors = nullptr;
    __nv_bfloat16 *out = nullptr;
    uint8_t *reference_payload = nullptr;
    uint8_t *reference_scales = nullptr;
    __nv_bfloat16 *reference_out = nullptr;
    uint32_t queries = 0;
    ~DeviceArena() {
        if (reference_out) cudaFree(reference_out);
        if (reference_scales) cudaFree(reference_scales);
        if (reference_payload) cudaFree(reference_payload);
        if (out) cudaFree(out);
        if (errors) cudaFree(errors);
        if (ids) cudaFree(ids);
        if (arena) cudaFree(arena);
    }
    void init(uint32_t q) {
        queries = q;
        ck(cudaMalloc(&arena, kArenaBytes), "device arena");
        ck(cudaMalloc(&ids, size_t(q) * sizeof(uint32_t)), "device ids");
        ck(cudaMalloc(&errors, sizeof(uint32_t)), "device errors");
        ck(cudaMalloc(&out, size_t(q) * kWidth * sizeof(__nv_bfloat16)), "device output");
        ck(cudaMalloc(&reference_payload, size_t(q) * kWidth), "local reference payload");
        ck(cudaMalloc(&reference_scales, size_t(q) * kScaleGroups), "local reference scales");
        ck(cudaMalloc(&reference_out, size_t(q) * kWidth * sizeof(__nv_bfloat16)), "local reference output");
        ck(cudaMemset(arena, kGuard, kArenaBytes), "init guard");
        ck(cudaMemset(errors, 0, sizeof(uint32_t)), "init errors");
    }
};

nixl_b_params_t backendParams(uint32_t gpu, uint16_t port) {
    nixl_b_params_t p;
    p["native_device_api"] = "true";
    p["gpu_devices"] = std::to_string(gpu);
    p["oob_port"] = std::to_string(port);
    for (auto [env_name, key] : {std::pair{"GPUNETIO_DEVICE_API_RDMA", "network_devices"},
                                 std::pair{"GPUNETIO_DEVICE_API_OOB_INTERFACE", "oob_interface"},
                                 std::pair{"GPUNETIO_DEVICE_API_GID_INDEX", "gid_index"}}) {
        auto value = env(env_name);
        if (!value.empty()) p[key] = value;
    }
    return p;
}
void waitFor(const fs::path &path) {
    const auto until = std::chrono::steady_clock::now() + kTimeout;
    while (!fs::exists(path)) {
        if (std::chrono::steady_clock::now() >= until) throw std::runtime_error("timeout waiting for " + path.string());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
int owner(const fs::path &coord, uint32_t gpu, uint16_t port) {
    ck(cudaSetDevice(static_cast<int>(gpu)), "select owner GPU");
    HostTable table; table.init();
    nixlAgentConfig cfg; cfg.useProgThread = true; cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    nixlAgent agent("engram_owner_m1", cfg);
    nixlBackendH *backend = nullptr;
    require(agent.createBackend("GPUNETIO", backendParams(gpu, port), backend), "create owner backend");
    nixl_opt_args_t hint; hint.backends.push_back(backend);
    void *pageable_probe = std::malloc(4096);
    if (pageable_probe == nullptr) throw std::bad_alloc();
    nixl_reg_dlist_t pageable_desc(DRAM_SEG);
    pageable_desc.addDesc({reinterpret_cast<uintptr_t>(pageable_probe), 4096, gpu, {}});
    const auto pageable_status = agent.registerMem(pageable_desc, &hint);
    std::free(pageable_probe);
    if (pageable_status == NIXL_SUCCESS)
        throw std::runtime_error("pageable host DRAM registration unexpectedly succeeded");
    nixl_reg_dlist_t registration(DRAM_SEG);
    registration.addDesc({reinterpret_cast<uintptr_t>(table.payload), kPayloadBytes, gpu, {}});
    registration.addDesc({reinterpret_cast<uintptr_t>(table.scales), kScaleBytes, gpu, {}});
    require(agent.registerMem(registration, &hint), "register host DRAM table");
    nixl_blob_t md; require(agent.getLocalMD(md), "owner metadata");
    atomicWrite(coord / "owner.md", md);
    std::ostringstream desc;
    desc << reinterpret_cast<uintptr_t>(table.payload) << ' ' << kPayloadBytes << ' ' << gpu << '\n'
         << reinterpret_cast<uintptr_t>(table.scales) << ' ' << kScaleBytes << ' ' << gpu << '\n';
    atomicWrite(coord / "owner.desc", desc.str());
    atomicWrite(coord / "owner.generation", "m1-single-owner-generation-1\n");
    atomicWrite(coord / "owner.ready", "loaded_once=1\n");
    waitFor(coord / "stop-owner");
    require(agent.deregisterMem(registration, &hint), "deregister host DRAM table");
    std::cout << "OWNER_EXIT registered_host_bytes=" << kPayloadBytes + kScaleBytes
              << " pageable_registration_status=" << pageable_status
              << " owner_generation=m1-single-owner-generation-1 load_count=1\n";
    return 0;
}
int reader(const fs::path &coord, uint32_t gpu, uint16_t port, const std::string &name,
           uint32_t seed, uint32_t queries, uint32_t loops, uint32_t pause_ms) {
    ck(cudaSetDevice(static_cast<int>(gpu)), "select reader GPU");
    waitFor(coord / "owner.ready");
    std::istringstream desc(readAll(coord / "owner.desc"));
    uintptr_t payload_addr = 0, scales_addr = 0;
    size_t payload_len = 0, scales_len = 0;
    uint32_t payload_dev = 0, scales_dev = 0;
    if (!(desc >> payload_addr >> payload_len >> payload_dev >> scales_addr >> scales_len >> scales_dev) ||
        payload_len != kPayloadBytes || scales_len != kScaleBytes)
        throw std::runtime_error("invalid owner descriptor file");
    DeviceArena mem; mem.init(queries);
    nixlAgentConfig cfg; cfg.useProgThread = true; cfg.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_RW;
    nixlAgent agent("engram_reader_m1_" + name, cfg);
    nixlBackendH *backend = nullptr;
    require(agent.createBackend("GPUNETIO", backendParams(gpu, port), backend), "create reader backend");
    nixl_opt_args_t hint; hint.backends.push_back(backend);
    nixl_reg_dlist_t registration(VRAM_SEG);
    registration.addDesc({reinterpret_cast<uintptr_t>(mem.arena), kArenaBytes, gpu, {}});
    require(agent.registerMem(registration, &hint), "register reader scratch");
    void *host_probe = nullptr;
    ck(cudaHostAlloc(&host_probe, 4096, cudaHostAllocPortable), "allocate local host-view probe");
    nixl_reg_dlist_t host_probe_registration(DRAM_SEG);
    host_probe_registration.addDesc({reinterpret_cast<uintptr_t>(host_probe), 4096, gpu, {}});
    require(agent.registerMem(host_probe_registration, &hint), "register pinned local host-view probe");
    nixl_local_dlist_t unsupported_local(DRAM_SEG);
    unsupported_local.addDesc({reinterpret_cast<uintptr_t>(host_probe), 4096, gpu});
    nixlMemViewH rejected_local_view = nullptr;
    const auto local_host_status = agent.prepMemView(unsupported_local, rejected_local_view, &hint);
    if (local_host_status == NIXL_SUCCESS || rejected_local_view != nullptr)
        throw std::runtime_error("pinned DRAM was incorrectly accepted as a local device view");
    require(agent.deregisterMem(host_probe_registration, &hint), "deregister local host-view probe");
    ck(cudaFreeHost(host_probe), "free local host-view probe");
    std::string peer;
    require(agent.loadRemoteMD(readAll(coord / "owner.md"), peer), "load owner metadata");
    require(agent.makeConnection(peer, &hint), "connect owner");
    nixl_local_dlist_t local(VRAM_SEG);
    local.addDesc({reinterpret_cast<uintptr_t>(mem.arena), kArenaBytes, gpu});
    nixl_remote_dlist_t remote(DRAM_SEG);
    remote.addDesc({payload_addr, payload_len, payload_dev, peer});
    remote.addDesc({scales_addr, scales_len, scales_dev, peer});
    nixlMemViewH local_view = nullptr, remote_view = nullptr;
    require(agent.prepMemView(local, local_view, &hint), "prepare local scratch view");
    require(agent.prepMemView(remote, remote_view, &hint), "prepare remote host table view");
    cudaEvent_t event_start{}, event_stop{};
    ck(cudaEventCreate(&event_start), "create lookup start event");
    ck(cudaEventCreate(&event_stop), "create lookup stop event");
    std::vector<float> lookup_event_ms;
    lookup_event_ms.reserve(loops);
    atomicWrite(coord / ("reader." + name + ".started"), "connected=1\n");
    for (uint32_t loop = 0; loop < loops; ++loop) {
        const uint32_t loop_seed = seed + loop * 17u;
        // Build a selected-row HBM reference only for validation. It is not used by the GET path.
        std::vector<uint8_t> reference_payload(size_t(queries) * kWidth);
        std::vector<uint8_t> reference_scales(size_t(queries) * kScaleGroups);
        for (uint32_t i = 0; i < queries; ++i) {
            const auto row = rowId(i, loop_seed);
            for (uint32_t c = 0; c < kWidth; ++c)
                reference_payload[size_t(i) * kWidth + c] = codeFor(row, c);
            for (uint32_t g = 0; g < kScaleGroups; ++g)
                reference_scales[size_t(i) * kScaleGroups + g] = scaleFor(row, g);
        }
        ck(cudaMemcpy(mem.reference_payload, reference_payload.data(), reference_payload.size(),
                      cudaMemcpyHostToDevice), "stage selected local reference payload");
        ck(cudaMemcpy(mem.reference_scales, reference_scales.data(), reference_scales.size(),
                      cudaMemcpyHostToDevice), "stage selected local reference scales");
        decodeLocalReference<<<(queries * kWidth + 255) / 256, 256>>>(
            mem.reference_payload, mem.reference_scales, queries, mem.reference_out);
        ck(cudaGetLastError(), "launch local HBM reference dequantization");
        makeIds<<<(queries + 127) / 128, 128>>>(mem.ids, queries, loop_seed);
        ck(cudaGetLastError(), "make GPU ids");
        ck(cudaEventRecord(event_start), "record lookup start");
        fetchAndDecode<<<1, 256>>>(mem.ids, queries, mem.arena, mem.out, local_view, remote_view, mem.errors);
        ck(cudaGetLastError(), "launch GPUNETIO Engram lookup");
        ck(cudaEventRecord(event_stop), "record lookup stop");
        ck(cudaEventSynchronize(event_stop), "complete lookup event");
        float elapsed_ms = 0.0f;
        ck(cudaEventElapsedTime(&elapsed_ms, event_start, event_stop), "read lookup event time");
        lookup_event_ms.push_back(elapsed_ms);
        uint32_t errors = 0;
        ck(cudaMemcpy(&errors, mem.errors, sizeof(errors), cudaMemcpyDeviceToHost), "read GPU validation status");
        if (errors) throw std::runtime_error("GPU payload/scale/guard validation failed: " + std::to_string(errors));
        std::vector<uint16_t> got(size_t(queries) * kWidth);
        std::vector<uint16_t> local_reference(size_t(queries) * kWidth);
        ck(cudaMemcpy(got.data(), mem.out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read lookup output");
        ck(cudaMemcpy(local_reference.data(), mem.reference_out,
                      local_reference.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
           "read local HBM reference output");
        for (uint32_t i = 0; i < queries; ++i) {
            const uint32_t row = rowId(i, loop_seed);
            for (uint32_t c = 0; c < kWidth; ++c) {
                const auto index = size_t(i) * kWidth + c;
                if (local_reference[index] != expectedBf16(row, c))
                    throw std::runtime_error("native CUDA local reference mismatch reader=" + name +
                                             " query=" + std::to_string(i) + " col=" + std::to_string(c));
                if (got[index] != local_reference[index])
                    throw std::runtime_error("native CUDA dequant mismatch reader=" + name +
                                             " query=" + std::to_string(i) + " col=" + std::to_string(c));
            }
        }
        ck(cudaMemset(mem.errors, 0, sizeof(uint32_t)), "clear validation status");
        atomicWrite(coord / ("reader." + name + ".progress"), std::to_string(loop + 1) + "\n");
        if (pause_ms) std::this_thread::sleep_for(std::chrono::milliseconds(pause_ms));
    }
    agent.releaseMemView(remote_view);
    agent.releaseMemView(local_view);
    require(agent.deregisterMem(registration, &hint), "deregister reader scratch");
    ck(cudaEventDestroy(event_start), "destroy lookup start event");
    ck(cudaEventDestroy(event_stop), "destroy lookup stop event");
    std::ostringstream timing;
    timing << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < lookup_event_ms.size(); ++i) {
        if (i) timing << ',';
        timing << lookup_event_ms[i];
    }
    atomicWrite(coord / ("reader." + name + ".done"),
                "queries=" + std::to_string(uint64_t(queries) * loops) +
                    " lookup_event_ms=" + timing.str() + " validation=PASS\n");
    std::cout << "READER_PASS name=" << name << " gpu=" << gpu
              << " generation=m1-single-owner-generation-1 table_copy_bytes=0"
              << " local_dram_view_status=" << local_host_status
              << " scratch_bytes=" << kArenaBytes + size_t(queries) * sizeof(uint32_t) +
                     size_t(queries) * kWidth * sizeof(__nv_bfloat16)
              << " reference_only_bytes=" << size_t(queries) *
                     (kWidth + kScaleGroups + kWidth * sizeof(__nv_bfloat16))
              << " queries=" << uint64_t(queries) * loops
              << " lookup_event_ms=" << timing.str() << " validation=PASS\n";
    return 0;
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 6) throw std::runtime_error("usage: shared_engram_m1 owner|reader coord gpu port name [seed queries loops pause_ms]");
        const std::string role = argv[1];
        const fs::path coord = argv[2];
        const auto gpu = static_cast<uint32_t>(std::stoul(argv[3]));
        const auto port = static_cast<uint16_t>(std::stoul(argv[4]));
        const std::string name = argv[5];
        if (role == "owner") return owner(coord, gpu, port);
        if (role == "reader") {
            if (argc != 10) throw std::runtime_error("reader requires seed, queries, loops, pause_ms");
            return reader(coord, gpu, port, name, std::stoul(argv[6]),
                          std::stoul(argv[7]), std::stoul(argv[8]), std::stoul(argv[9]));
        }
        throw std::runtime_error("unknown role: " + role);
    } catch (const std::exception &e) {
        std::cerr << "M1_FAIL " << e.what() << '\n';
        return 1;
    }
}
