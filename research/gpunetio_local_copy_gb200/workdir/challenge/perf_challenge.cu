// SPDX-License-Identifier: Apache-2.0
#include "local_copy.cuh"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <bit>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

extern "C" void *lc_open(void *, size_t, void *, size_t, int, const char *, int);
extern "C" const char *lc_error();
extern "C" void *lc_device(void *);
extern "C" int lc_close(void *);

__device__ __forceinline__ uint64_t ns() {
    uint64_t t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

__global__ void load_cost(local_copy::Copy *copy, int *local, uint64_t *out) {
    if (threadIdx.x) return;
    int sum = 0;
    auto host = cuda::atomic_ref<int, cuda::thread_scope_system>(copy->recovery->fault);
    auto gpu = cuda::atomic_ref<int, cuda::thread_scope_device>(*local);
    uint64_t a = ns();
    for (int i = 0; i < 1024; ++i) sum += host.load(cuda::memory_order_acquire);
    uint64_t b = ns();
    for (int i = 0; i < 1024; ++i) sum += gpu.load(cuda::memory_order_acquire);
    uint64_t c = ns();
    out[0] = b - a; out[1] = c - b; out[2] = sum;
}

// Minimal real consumer. Full validation is a separate, untimed GPU kernel.
__global__ void transfer(local_copy::Copy *copy, const uint8_t *src, uint8_t *dst,
                         size_t bytes, int count, int window, int path,
                         int compute, uint64_t *out, const int *indices) {
    __shared__ int status;
    __shared__ unsigned sample_slot;
    __shared__ local_copy::Ticket ticket;
    if (threadIdx.x == 0) { status = 0; sample_slot = out[10]++; }
    __syncthreads();
    uint64_t begin = ns(), issue = 0, wait = 0;
    float independent = threadIdx.x + 0.5f;
    for (int start = 0; start < count; start += window) {
        int n = min(window, count - start);
        if (threadIdx.x == 0 && path == 0 && status == 0) {
            auto a = ns();
            for (int k = 0; k < n; ++k) {
                int j = start + k;
                auto t = copy->copy_async(dst + 128 + j * (bytes + 128),
                    src + 128 + indices[j] * (bytes + 128), bytes);
                if (t.error) { status = t.error; break; }
                ticket = t;
            }
            issue += ns() - a;
        }
        if (path == 3 && threadIdx.x < 32 && status == 0) {
            auto a = ns();
            int j = start + threadIdx.x;
            bool active = threadIdx.x < n;
            auto t = copy->copy_async_warp(active ? dst + 128 + j * (bytes + 128) : dst,
                active ? src + 128 + indices[j] * (bytes + 128) : src, bytes, active);
            if (threadIdx.x == 0) { ticket = t; status = t.error; issue += ns() - a; }
        }
        if (path == 2) {
            for (unsigned unit = threadIdx.x; unit < n * bytes / 16; unit += blockDim.x) {
                unsigned j = start + (unit * 16) / bytes, offset = (unit * 16) % bytes;
                auto *from = reinterpret_cast<const uint4 *>(src + 128 + indices[j] * (bytes + 128) + offset);
                auto *to = reinterpret_cast<uint4 *>(dst + 128 + j * (bytes + 128) + offset);
                *to = *from;
            }
        }
        for (int k = 0; k < compute; ++k) independent = __fmaf_rn(independent, 0.9999f, 0.001f);
        if (threadIdx.x == 0 && (path == 0 || path == 3) && status == 0) {
            auto a = ns();
            status = copy->wait(ticket);
            wait += ns() - a;
        }
        __syncthreads();
    }
    if (status == 0) {
        size_t j = threadIdx.x % count, offset = threadIdx.x % bytes;
        uint8_t x = reinterpret_cast<volatile uint8_t *>(dst)[128 + j * (bytes + 128) + offset];
        out[16 + sample_slot * 128 + threadIdx.x] = __float_as_uint(independent) + x;
    }
    if (threadIdx.x == 0) {
        out[0] = static_cast<int64_t>(status);
        out[1] += issue; out[2] += wait; out[3] += ns() - begin;
        out[8] = copy->retired;
    }
}

__global__ void verify(const uint8_t *dst, size_t size, size_t bytes, int count, int epoch,
                       uint64_t *out, const int *indices) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x) {
        uint8_t expected = 0xcd;
        if (i >= 128) {
            size_t j = (i - 128) / (bytes + 128), offset = (i - 128) % (bytes + 128);
            if (j < static_cast<size_t>(count) && offset < bytes) {
                size_t source = 128 + indices[j] * (bytes + 128) + offset;
                expected = (source * 17 + epoch * 31) % 251;
            }
        }
        if (dst[i] != expected) atomicAdd(reinterpret_cast<unsigned long long *>(out + 7), 1ULL);
    }
}

#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)
int main(int argc, char **argv) {
    int gpu = argc > 1 ? atoi(argv[1]) : 7;
    int blocks = argc > 2 ? atoi(argv[2]) : 7;
    int repeats = argc > 3 ? atoi(argv[3]) : 100;
    int compute = argc > 4 ? atoi(argv[4]) : 0;
    int only_count = argc > 5 ? atoi(argv[5]) : 0;
    size_t only_bytes = argc > 6 ? strtoull(argv[6], nullptr, 10) : 0;
    std::string only_arm = argc > 7 && std::string(argv[7]) != "all" ? argv[7] : "";
    unsigned layout_seed = argc > 8 ? strtoul(argv[8], nullptr, 10) : 0;
    CUDA(cudaSetDevice(gpu));
    for (int count : {1, 8, 32, 128}) for (size_t bytes : {size_t(64), size_t(4096)}) {
        if ((only_count && count != only_count) || (only_bytes && bytes != only_bytes)) continue;
        size_t size = 256 + count * (bytes + 128);
        uint8_t *src, *dst;
        std::vector<int> mapping(count);
        std::iota(mapping.begin(), mapping.end(), 0);
        if (layout_seed) {
            std::mt19937 rng(layout_seed + count);
            std::shuffle(mapping.begin(), mapping.end(), rng);
        } else for (int j = 0; j < count; ++j) mapping[j] = (j*13)%count;
        printf("{\"kind\":\"layout\",\"count\":%d,\"bytes\":%zu,\"seed\":%u,\"indices\":[", count, bytes, layout_seed);
        for (int j = 0; j < count; ++j) printf("%s%d", j ? "," : "", mapping[j]);
        printf("]}\n");
        int *indices;
        CUDA(cudaMalloc(&indices, count*sizeof(int)));
        CUDA(cudaMemcpy(indices, mapping.data(), count*sizeof(int), cudaMemcpyHostToDevice));
        uint64_t *out;
        std::vector<uint64_t> result(16 + 128 * (repeats > 10 ? repeats : 10));
        const size_t out_bytes = result.size() * sizeof(uint64_t);
        int *flag;
        CUDA(cudaMallocHost(&src, size)); CUDA(cudaMalloc(&dst, size));
        uint8_t *src_device;
        CUDA(cudaHostGetDevicePointer(&src_device, src, 0));
        CUDA(cudaMalloc(&out, out_bytes)); CUDA(cudaMalloc(&flag, sizeof(*flag)));
        CUDA(cudaMemset(flag, 0, sizeof(*flag)));
        void *handle = lc_open(src, size, dst, size, gpu, "mlx5_bond_0", 3);
        if (!handle) { fprintf(stderr, "%s\n", lc_error()); return 3; }
        auto *copy = static_cast<local_copy::Copy *>(lc_device(handle));
        if (count == 1 && bytes == 64) {
            for (int i = 0; i < 7; ++i) {
                load_cost<<<1, 1>>>(copy, flag, out);
                CUDA(cudaDeviceSynchronize()); CUDA(cudaMemcpy(result.data(), out, out_bytes, cudaMemcpyDeviceToHost));
                printf("{\"kind\":\"flag_load\",\"sample\":%d,\"host_ns\":%.3f,\"device_ns\":%.3f}\n", i, result[0]/1024., result[1]/1024.);
            }
        }
        cudaStream_t stream;
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        std::vector<void *> srcs(count), dsts(count);
        std::vector<size_t> sizes(count, bytes);
        for (int j = 0; j < count; ++j) {
            srcs[j] = src + 128 + mapping[j] * (bytes + 128);
            dsts[j] = dst + 128 + j * (bytes + 128);
        }
        // Discover batch support outside capture. No dependent/overlapping copies.
        cudaMemcpyAttributes attr{};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
        size_t idx = 0, failed = SIZE_MAX;
        auto batch_status = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), count,
            &attr, &idx, 1, &failed, stream);
        bool batch = batch_status == cudaSuccess;
        printf("{\"kind\":\"batch_support\",\"count\":%d,\"bytes\":%zu,\"status\":%d,\"message\":\"%s\"}\n", count, bytes, batch_status, cudaGetErrorString(batch_status));
        if (!batch) cudaGetLastError();
        CUDA(cudaStreamSynchronize(stream));
        std::vector<std::string> names{"NIC", "CUDA_GRAPH", "SM_GATHER"};
        if (batch) names.push_back("CUDA_BATCH_GRAPH");
        names.push_back("CUDA_PARALLEL_GRAPH");
        names.push_back("CUDA_2D_GRAPH");
        names.push_back("CUDA_2D_PARALLEL_GRAPH");
        names.push_back("CUDA_2D_PACKED_SERIAL_GRAPH");
        names.push_back("CUDA_2D_PACKED_GRAPH");
        names.push_back("NIC_WARP");
        // Also allow nonadjacent destination rows in a pitched copy. This is
        // a deterministic packing heuristic, not an optimal-cover claim.
        struct Run { int first, step, delta, height; };
        std::vector<Run> packed;
        std::vector<bool> used(count, false);
        for (;;) {
            Run best{-1,1,1,1};
            for (int i = 0; i < count; ++i) if (!used[i]) {
                if (best.first < 0) best.first = i;
                for (int j = i+1; j < count; ++j) if (!used[j] && mapping[j] > mapping[i]) {
                    int step = j-i, delta = mapping[j]-mapping[i], height = 2;
                    while (i+height*step < count && !used[i+height*step] &&
                        mapping[i+height*step] == mapping[i]+height*delta) ++height;
                    if (height > best.height) best = {i,step,delta,height};
                }
            }
            if (best.first < 0) break;
            packed.push_back(best);
            for (int h = 0; h < best.height; ++h) used[best.first+h*best.step] = true;
        }
        std::vector<cudaGraph_t> graphs(names.size());
        std::vector<cudaGraphExec_t> execs(names.size());
        for (size_t arm = 0; arm < names.size(); ++arm) {
            if (names[arm] == "CUDA_PARALLEL_GRAPH" || names[arm] == "CUDA_2D_GRAPH" ||
                names[arm] == "CUDA_2D_PARALLEL_GRAPH" || names[arm] == "CUDA_2D_PACKED_GRAPH" ||
                names[arm] == "CUDA_2D_PACKED_SERIAL_GRAPH") {
                CUDA(cudaGraphCreate(&graphs[arm], 0));
                std::vector<cudaGraphNode_t> nodes;
                if (names[arm] == "CUDA_2D_PACKED_GRAPH" || names[arm] == "CUDA_2D_PACKED_SERIAL_GRAPH")
                    for (const auto &run : packed) {
                        cudaMemcpy3DParms p{};
                        p.srcPtr = make_cudaPitchedPtr(srcs[run.first],run.delta*(bytes+128),bytes,run.height);
                        p.dstPtr = make_cudaPitchedPtr(dsts[run.first],run.step*(bytes+128),bytes,run.height);
                        p.extent = make_cudaExtent(bytes,run.height,1); p.kind = cudaMemcpyHostToDevice;
                        const bool serial = names[arm] == "CUDA_2D_PACKED_SERIAL_GRAPH" && !nodes.empty();
                        cudaGraphNode_t node;
                        CUDA(cudaGraphAddMemcpyNode(&node,graphs[arm],serial ? &nodes.back() : nullptr,
                            serial ? 1 : 0,&p));
                        nodes.push_back(node);
                    }
                else for (int j = 0; j < count;) {
                    cudaGraphNode_t node;
                    if (names[arm] == "CUDA_PARALLEL_GRAPH") {
                        CUDA(cudaGraphAddMemcpyNode1D(&node, graphs[arm], nullptr, 0,
                            dsts[j], srcs[j], bytes, cudaMemcpyHostToDevice));
                        ++j;
                    } else {
                        int start = mapping[j], delta = j+1 < count ? mapping[j+1]-start : 0;
                        int height = 1;
                        if (delta > 0) while (j+height < count &&
                            mapping[j+height] == start+height*delta) ++height;
                        cudaMemcpy3DParms p{};
                        p.srcPtr = make_cudaPitchedPtr(srcs[j], (height > 1 ? delta : 1)*(bytes+128), bytes, height);
                        p.dstPtr = make_cudaPitchedPtr(dsts[j], bytes+128, bytes, height);
                        p.extent = make_cudaExtent(bytes, height, 1); p.kind = cudaMemcpyHostToDevice;
                        const bool serial = names[arm] == "CUDA_2D_GRAPH" && !nodes.empty();
                        CUDA(cudaGraphAddMemcpyNode(&node, graphs[arm], serial ? &nodes.back() : nullptr,
                            serial ? 1 : 0, &p));
                        j += height;
                    }
                    nodes.push_back(node);
                }
                int window = 32, path = 1;
                void *args[] = {&copy, &src, &dst, &bytes, &count, &window, &path, &compute, &out, &indices};
                cudaKernelNodeParams params{};
                params.func = reinterpret_cast<void *>(transfer);
                params.gridDim = dim3(1); params.blockDim = dim3(128); params.kernelParams = args;
                cudaGraphNode_t consumer;
                CUDA(cudaGraphAddKernelNode(&consumer, graphs[arm], nodes.data(), nodes.size(), &params));
                CUDA(cudaGraphInstantiate(&execs[arm], graphs[arm], nullptr, nullptr, 0));
                printf("{\"kind\":\"graph_nodes\",\"count\":%d,\"bytes\":%zu,\"arm\":\"%s\",\"copy_nodes\":%zu}\n",count,bytes,names[arm].c_str(),nodes.size());
                continue;
            }
            CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            if (arm == 1) for (int j = 0; j < count; ++j)
                CUDA(cudaMemcpyAsync(dsts[j], srcs[j], bytes, cudaMemcpyHostToDevice, stream));
            if (names[arm] == "CUDA_BATCH_GRAPH")
                CUDA(cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), count,
                    &attr, &idx, 1, &failed, stream));
            transfer<<<1, 128, 0, stream>>>(copy, arm == 2 ? src_device : src, dst,
                bytes, count, 32, arm == 0 ? 0 : (arm == 2 ? 2 : (names[arm] == "NIC_WARP" ? 3 : 1)), compute, out, indices);
            CUDA(cudaStreamEndCapture(stream, &graphs[arm]));
            CUDA(cudaGraphInstantiate(&execs[arm], graphs[arm], nullptr, nullptr, 0));
        }
        cudaEvent_t a, b;
        CUDA(cudaEventCreate(&a)); CUDA(cudaEventCreate(&b));
        size_t free_before, total;
        CUDA(cudaMemGetInfo(&free_before, &total));
        for (int block = 0; block < blocks; ++block) {
            size_t free_now;
            CUDA(cudaMemGetInfo(&free_now, &total));
            if (free_now + 8*1024*1024 < free_before) { fprintf(stderr,"INVALID GPU memory occupancy changed\n"); return 10; }
            const int epoch = block + 1;
            for (size_t i = 0; i < size; ++i) src[i] = (i * 17 + epoch * 31) % 251;
            for (size_t order = 0; order < names.size(); ++order) {
                size_t arm = block % 2 ? names.size()-1-order : order;
                if (!only_arm.empty() && names[arm] != only_arm) continue;
                CUDA(cudaMemsetAsync(dst, 0xcd, size, stream));
                CUDA(cudaMemsetAsync(out, 0, out_bytes, stream));
                for (int w = 0; w < 10; ++w) CUDA(cudaGraphLaunch(execs[arm], stream));
                CUDA(cudaStreamSynchronize(stream));
                // Reset destination so measured work cannot rely on warmup payload.
                CUDA(cudaMemsetAsync(dst, 0xcd, size, stream));
                CUDA(cudaMemsetAsync(out, 0, out_bytes, stream));
                CUDA(cudaStreamSynchronize(stream));
                const auto begin = std::chrono::steady_clock::now();
                CUDA(cudaEventRecord(a, stream));
                for (int i = 0; i < repeats; ++i) CUDA(cudaGraphLaunch(execs[arm], stream));
                CUDA(cudaEventRecord(b, stream)); CUDA(cudaEventSynchronize(b));
                const auto end = std::chrono::steady_clock::now();
                float elapsed;
                CUDA(cudaEventElapsedTime(&elapsed, a, b));
                verify<<<32, 128, 0, stream>>>(dst, size, bytes, count, epoch, out, indices);
                CUDA(cudaStreamSynchronize(stream)); CUDA(cudaMemcpy(result.data(), out, out_bytes, cudaMemcpyDeviceToHost));
                if (result[0] || result[7]) { fprintf(stderr,"FAIL status=%lld mismatch=%llu\n",(long long)result[0],(unsigned long long)result[7]); return 5; }
                if (result[10] != static_cast<unsigned>(repeats)) return 11;
                for (int r = 0; r < repeats; ++r) for (unsigned lane = 0; lane < 128; ++lane) {
                    size_t source = 128 + mapping[lane%count]*(bytes+128) + lane%bytes;
                    float f = lane + 0.5f;
                    for (int k = 0; k < compute*((count+31)/32); ++k) f = std::fma(f,0.9999f,0.001f);
                    uint64_t expected = uint64_t(std::bit_cast<uint32_t>(f)) + (source*17+epoch*31)%251;
                    if (result[16 + r*128 + lane] != expected) {
                        fprintf(stderr,"FAIL consumed output arm=%s request=%d lane=%u\n",names[arm].c_str(),r,lane); return 12;
                    }
                }
                printf("{\"kind\":\"perf\",\"count\":%d,\"bytes\":%zu,\"block\":%d,\"arm\":\"%s\",\"compute\":%d,\"device_us\":%.4f,\"wall_us\":%.4f,\"issue_us\":%.4f,\"wait_us\":%.4f,\"kernel_us\":%.4f,\"mismatch\":0,\"retired\":%llu}\n",
                    count,bytes,block,names[arm].c_str(),compute,elapsed*1000/repeats,
                    std::chrono::duration<double,std::micro>(end-begin).count()/repeats,
                    result[1]/(1000.*repeats), result[2]/(1000.*repeats), result[3]/(1000.*repeats),
                    (unsigned long long)result[8]);
                fflush(stdout);
            }
        }
        for (size_t arm = 0; arm < names.size(); ++arm) { CUDA(cudaGraphExecDestroy(execs[arm])); CUDA(cudaGraphDestroy(graphs[arm])); }
        CUDA(cudaEventDestroy(a)); CUDA(cudaEventDestroy(b)); CUDA(cudaStreamDestroy(stream));
        if (lc_close(handle)) { fprintf(stderr,"%s\n",lc_error()); return 6; }
        CUDA(cudaFree(indices)); CUDA(cudaFree(flag)); CUDA(cudaFree(out)); CUDA(cudaFree(dst)); CUDA(cudaFreeHost(src));
    }
}
