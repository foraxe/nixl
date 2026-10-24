// SPDX-License-Identifier: Apache-2.0
#include "local_copy.cuh"

__global__ void transform(local_copy::Copy *copy, const float *src, float *dst,
                           float *out, int *error) {
    __shared__ local_copy::Ticket ticket;
    __shared__ int status;
    if (threadIdx.x == 0) ticket = copy->copy_async(dst, src, 128 * sizeof(float));

    const float bias = threadIdx.x * 0.5f;  // independent work while NIC copies

    if (threadIdx.x == 0) status = copy->wait(ticket);
    __syncthreads();  // publish completion to this CTA's consumers
    if (status == 0) out[threadIdx.x] = dst[threadIdx.x] + bias;
    if (threadIdx.x == 0) *error = status;
}

extern "C" int lc_example(void *copy, const void *src, void *dst, void *out,
                           void *error, void *stream) {
    transform<<<1, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<local_copy::Copy *>(copy), static_cast<const float *>(src),
        static_cast<float *>(dst), static_cast<float *>(out), static_cast<int *>(error));
    return cudaGetLastError();
}
