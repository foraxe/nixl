#!/usr/bin/env bash
set -euo pipefail
: "${E_ROOT:?path to pinned E build/install/source root}"
: "${DOCA_ROOT:?path to DOCA 3.5.0098 prefix}"
: "${CUDA_ROOT:?CUDA toolkit prefix}"
cd "$(dirname "$0")"
multiarch=${MULTIARCH:-$(gcc -dumpmachine)}
arch=${CUDA_ARCH:-90}
build_dir=${LOCAL_COPY_BUILD_DIR:-$PWD}
mkdir -p "$build_dir"
library=${LOCAL_COPY_LIBRARY:-liblocal_copy.so}
[[ $library == /* ]] || library="$build_dir/$library"
export PKG_CONFIG_PATH="$E_ROOT/pkgconfig:$DOCA_ROOT/../../../usr/lib/$multiarch/pkgconfig"
export LD_LIBRARY_PATH="$E_ROOT/install/lib/$multiarch/plugins:$E_ROOT/install/lib/$multiarch:$DOCA_ROOT/lib/$multiarch:$DOCA_ROOT/../../../usr/lib/$multiarch:$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
"$CUDA_ROOT/bin/nvcc" -std=c++20 -arch="sm_$arch" --cudart shared -O3 -shared -Xcompiler=-fPIC \
    -I"$DOCA_ROOT/../../../usr/include" \
    -I"$DOCA_ROOT/include" -I"$E_ROOT/source/src/plugins/gpunetio" \
    -I"$E_ROOT/source/src/utils" -I"$E_ROOT/source/subprojects/abseil-cpp-20250814.1" \
    ${LOCAL_COPY_TEST_FLAGS:-} runtime.cu probe.cu example.cu ${LOCAL_COPY_EXTRA_SOURCES:-} -L"$E_ROOT/install/lib/$multiarch/plugins" \
    -lplugin_GPUNETIO -L"$DOCA_ROOT/lib/$multiarch" -L"$DOCA_ROOT/../../../usr/lib/$multiarch" \
    -ldoca_gpunetio -ldoca_verbs -ldoca_common -libverbs -lcuda -lpthread \
    -o "$library"
"$CUDA_ROOT/bin/nvcc" -std=c++20 -arch="sm_$arch" --cudart shared -O2 smoke.cu -L"$build_dir" -llocal_copy -o "$build_dir/smoke"
"$CUDA_ROOT/bin/nvcc" -std=c++20 -arch="sm_$arch" --cudart shared -O2 benchmark.cu -L"$build_dir" -llocal_copy -o "$build_dir/benchmark"
