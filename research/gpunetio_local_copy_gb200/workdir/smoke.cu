// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
extern "C" void *lc_open(void *, size_t, void *, size_t, int, const char *, int);
extern "C" const char *lc_error();
extern "C" void *lc_device(void *);
extern "C" int lc_close(void *);
extern "C" int lc_probe(void *, void *, void *, size_t, int, int, int, int, int, void *, void *);
int main(int argc, char **argv) {
    int gpu = argc > 1 ? atoi(argv[1]) : 7;
    size_t bytes = argc > 2 ? strtoull(argv[2], nullptr, 10) : 4096;
    int count = argc > 3 ? atoi(argv[3]) : 1;
    int window = argc > 4 ? atoi(argv[4]) : 1;
    int rounds = argc > 5 ? atoi(argv[5]) : 3;
    int mode = argc > 6 ? atoi(argv[6]) : 0;
    cudaSetDevice(gpu);
    size_t sn = 256 + count * (bytes + 17), dn = 256 + count * (bytes + 32);
    uint8_t *src, *dst;
    uint64_t *out, result[6]{};
    if (cudaMallocHost(&src, sn) || cudaMalloc(&dst, dn) || cudaMalloc(&out, sizeof(result))) return 2;
    for (size_t i = 0; i < sn; ++i) src[i] = (i * 17 + 31) % 251;
    const char* nic = argc > 7 ? argv[7] : "mlx5_bond_0";
    int gid = argc > 8 ? atoi(argv[8]) : 3;
    auto *handle = lc_open(src, sn, dst, dn, gpu, nic, gid);
    if (!handle) { fprintf(stderr, "OPEN_FAIL %s\n", lc_error()); return 3; }
    cudaMemset(out, 0, sizeof(result));
    auto launch = lc_probe(lc_device(handle), src, dst, bytes, count, window, rounds, 1, mode, out, nullptr);
    auto sync = cudaDeviceSynchronize();
    cudaMemcpy(result, out, sizeof(result), cudaMemcpyDeviceToHost);
    int close = lc_close(handle);
    printf("RESULT launch=%d sync=%d mismatch=%llu status=%lld posted=%llu retired=%llu mcst=%llu compute=%llu close=%d\n",
           launch, sync, (unsigned long long)result[0], (long long)result[1],
           (unsigned long long)result[2], (unsigned long long)result[3],
           (unsigned long long)result[4], (unsigned long long)result[5], close);
    if (close) fprintf(stderr, "CLOSE_FAIL %s\n", lc_error());
    if (!close) { cudaFree(out); cudaFree(dst); cudaFreeHost(src); }
    return launch || sync || result[0] || result[1] || close ? 1 : 0;
}
