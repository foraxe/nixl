// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
extern "C" void *lc_open(void *, size_t, void *, size_t, int, const char *, int);
extern "C" const char *lc_error();
extern "C" void *lc_device(void *);
extern "C" int lc_close(void *);
extern "C" int lc_probe(void *, void *, void *, size_t, int, int, int, int, int, void *, void *);
#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)
int main(int argc, char **argv) {
    int gpu = argc > 1 ? atoi(argv[1]) : 7;
    CUDA(cudaSetDevice(gpu));
    for (auto bytes : {size_t(64), size_t(4096)}) {
        const int count = 32, window = 16, repeats = 100;
        size_t sn = 256 + count * (bytes + 17), dn = 256 + count * (bytes + 32);
        uint8_t *src, *dst;
        uint64_t *out, result[6];
        CUDA(cudaMallocHost(&src, sn)); CUDA(cudaMalloc(&dst, dn)); CUDA(cudaMalloc(&out, sizeof(result)));
        for (size_t i = 0; i < sn; ++i) src[i] = (i * 17 + 31) % 251;
        void *handle = lc_open(src, sn, dst, dn, gpu, "mlx5_bond_0", 3);
        if (!handle) { fprintf(stderr, "%s\n", lc_error()); return 3; }
        cudaStream_t stream;
        CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int j = 0; j < count; ++j)
            CUDA(cudaMemcpyAsync(dst + 128 + j * (bytes + 32),
                src + 128 + ((j * 13) % count) * (bytes + 17), bytes, cudaMemcpyHostToDevice, stream));
        if (lc_probe(lc_device(handle), src, dst, bytes, count, window, 1, 1, -1, out, stream)) return 4;
        CUDA(cudaStreamEndCapture(stream, &graph));
        CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        cudaEvent_t a, b;
        CUDA(cudaEventCreate(&a)); CUDA(cudaEventCreate(&b));
        for (int block = 0; block < 5; ++block) {
            for (int arm = 0; arm < 2; ++arm) {
                const bool nic = (arm ^ (block & 1)) == 0;
                CUDA(cudaMemsetAsync(dst, 0xcd, dn, stream));
                CUDA(cudaMemsetAsync(out, 0, sizeof(result), stream));
                for (int warm = 0; warm < 10; ++warm) {
                    if (nic) {
                        if (lc_probe(lc_device(handle), src, dst, bytes, count, window, 1, 1, -2, out, stream)) return 4;
                    } else CUDA(cudaGraphLaunch(executable, stream));
                }
                CUDA(cudaStreamSynchronize(stream));
                const auto begin = std::chrono::steady_clock::now();
                CUDA(cudaEventRecord(a, stream));
                for (int i = 0; i < repeats; ++i) {
                    if (nic) {
                        if (lc_probe(lc_device(handle), src, dst, bytes, count, window, 1, 1, -2, out, stream)) return 4;
                    } else CUDA(cudaGraphLaunch(executable, stream));
                }
                CUDA(cudaEventRecord(b, stream)); CUDA(cudaEventSynchronize(b));
                const auto end = std::chrono::steady_clock::now();
                float elapsed;
                CUDA(cudaEventElapsedTime(&elapsed, a, b));
                CUDA(cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost));
                if (result[0] || result[1]) { fprintf(stderr, "validation failed\n"); return 5; }
                printf("{\"bytes\":%zu,\"fragments\":%d,\"block\":%d,\"arm\":\"%s\",\"device_us\":%.3f,\"wall_us\":%.3f,\"mismatch\":%llu}\n",
                    bytes, count, block, nic ? "NIC" : "CUDA_GRAPH",
                    elapsed * 1000 / repeats,
                    std::chrono::duration<double, std::micro>(end - begin).count() / repeats,
                    (unsigned long long)result[0]);
            }
        }
        CUDA(cudaEventDestroy(a)); CUDA(cudaEventDestroy(b));
        CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
        CUDA(cudaStreamDestroy(stream));
        if (lc_close(handle)) { fprintf(stderr, "%s\n", lc_error()); return 6; }
        CUDA(cudaFree(out)); CUDA(cudaFree(dst)); CUDA(cudaFreeHost(src));
    }
}
