#!/usr/bin/env bash
set -euo pipefail
: "${E_ROOT:?path to pinned E build/install/source root}"
: "${DOCA_ROOT:?path to DOCA 3.5.0098 prefix}"
: "${CUDA_ROOT:?CUDA toolkit prefix}"
cd "$(dirname "$0")"
export PKG_CONFIG_PATH="$E_ROOT/pkgconfig:$DOCA_ROOT/../../../usr/lib/x86_64-linux-gnu/pkgconfig"
export LD_LIBRARY_PATH="$E_ROOT/install/lib/x86_64-linux-gnu:$DOCA_ROOT/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
"$CUDA_ROOT/bin/nvcc" -std=c++20 -arch=sm_90 -O3 -shared -Xcompiler=-fPIC \
    -I"$DOCA_ROOT/include" -I"$E_ROOT/source/src/plugins/gpunetio" \
    -I"$E_ROOT/source/src/utils" -I"$E_ROOT/source/subprojects/abseil-cpp-20250814.1" \
    ${LOCAL_COPY_TEST_FLAGS:-} runtime.cu probe.cu example.cu -L"$E_ROOT/install/lib/x86_64-linux-gnu/plugins" \
    -lplugin_GPUNETIO -L"$DOCA_ROOT/lib/x86_64-linux-gnu" \
    -ldoca_gpunetio -ldoca_verbs -ldoca_common -libverbs -lcuda -lpthread \
    -o "${LOCAL_COPY_LIBRARY:-liblocal_copy.so}"
"$CUDA_ROOT/bin/nvcc" -std=c++20 -O2 smoke.cu -L. -llocal_copy -o smoke
"$CUDA_ROOT/bin/nvcc" -std=c++20 -O2 benchmark.cu -L. -llocal_copy -o benchmark
