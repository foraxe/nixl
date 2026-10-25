#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tools_root=/workspace/gpunetio/kernel-copy-tools
out="$tools_root/primitive${PRIMITIVE_BUILD_SUFFIX:-}"
mkdir -p "$out"
E_ROOT="$tools_root/e" DOCA_ROOT="$tools_root/doca35/root/opt/mellanox/doca" CUDA_ROOT=/usr/local/cuda \
 CUDA_ARCH=100 LOCAL_COPY_BUILD_DIR="$out" LOCAL_COPY_EXTRA_SOURCES="$here/in_kernel.cu $here/example.cu" \
 LOCAL_COPY_TEST_FLAGS="${LOCAL_COPY_TEST_FLAGS:-}" bash "$here/../../workdir/build.sh"
/usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_100 --cudart shared -O3 -I /workspace/gpunetio/nccl-core-workload/build/include \
 "$here/run.cu" -L /workspace/gpunetio/nccl-core-workload/build/lib -lnccl -lcuda -ldl -o "$out/run"
