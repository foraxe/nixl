#!/usr/bin/env bash
set -euo pipefail
: "${LOCAL_COPY_ROOT:?qualified Copy header/runtime root}"
: "${DOCA_ROOT:?DOCA prefix}"
: "${CUDA_ROOT:?CUDA prefix}"
: "${CUBLAS_ROOT:?cuBLAS include/lib prefix}"
cd "$(dirname "$0")"
"$CUDA_ROOT/bin/nvcc" -std=c++20 -arch=sm_90 -O3 -I"$LOCAL_COPY_ROOT" -I"$DOCA_ROOT/include" \
    -I"$CUBLAS_ROOT/include" scenario.cu -L"$LOCAL_COPY_ROOT" -llocal_copy \
    -L"$CUBLAS_ROOT/lib" -l:libcublas.so.12 -o scenario
