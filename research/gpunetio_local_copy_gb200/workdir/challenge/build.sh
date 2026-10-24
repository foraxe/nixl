#!/usr/bin/env bash
set -euo pipefail
: "${LOCAL_COPY_ROOT:?directory with local_copy.cuh and liblocal_copy.so}"
: "${DOCA_ROOT:?DOCA prefix}"
: "${CUDA_ROOT:?CUDA prefix}"
cd "$LOCAL_COPY_ROOT"
"$CUDA_ROOT/bin/nvcc" -std=c++20 -arch=sm_90 -O3 -I. -I"$DOCA_ROOT/include" \
    perf_challenge.cu -L. -llocal_copy -o perf_challenge
