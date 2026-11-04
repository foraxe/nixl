#!/usr/bin/env bash
# Build the experimental ABI-3 adapter against an installed NIXL or source/build.
set -euo pipefail
: "${DOCA_INCLUDE:?Set DOCA_INCLUDE to the DOCA 3.5 include directory}"
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA toolkit root}"
: "${OUTPUT:?Set OUTPUT to the target shared library path}"
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n ${NIXL_PREFIX:-} ]]; then
    : "${NIXL_LIBDIR:?Set NIXL_LIBDIR to the installed NIXL library directory}"
    nixl_includes=(-I"$NIXL_PREFIX/include")
    nixl_libdir=$NIXL_LIBDIR
else
    : "${NIXL_SOURCE:?Set NIXL_SOURCE to the NIXL source root}"
    : "${NIXL_BUILD:?Set NIXL_BUILD to its configured build root}"
    nixl_includes=(-I"$NIXL_SOURCE/src/utils" -I"$NIXL_SOURCE/src/api/device" -I"$NIXL_SOURCE/src/api/cpp")
    nixl_libdir=$NIXL_BUILD/src/core
fi
"$CUDA_HOME/bin/nvcc" -std=c++20 -arch="${CUDA_ARCH:-sm_90}" -shared -Xcompiler -fPIC \
    -DNIXL_ENABLE_GPUNETIO_DEVICE_API=1 \
    "${nixl_includes[@]}" -I"$DOCA_INCLUDE" \
    "$source_dir/engram_native_nixl35.cu" -L"$nixl_libdir" -lnixl -o "$OUTPUT"
