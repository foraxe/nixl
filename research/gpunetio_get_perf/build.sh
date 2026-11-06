#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${NCCL_ROOT:?Set NCCL_ROOT to the selected NCCL build prefix (include/ and lib/)}"
: "${CUDA_ARCH:?Set CUDA_ARCH, e.g. sm_90 for H20 or sm_100 for GB200}"
task_binary="${BENCH_BINARY:-$task_dir/gin_nixl_bench_gdaki}"
case "${GIN_SHARING:-GPU}" in GPU|CTA) ;; *) echo 'GIN_SHARING must be GPU or CTA' >&2; exit 2 ;; esac
[[ ! -e "$task_binary" ]] || { echo 'Refusing to replace existing binary; choose BENCH_BINARY' >&2; exit 2; }
"${NVCC:-nvcc}" -std=c++17 -O2 -lineinfo -arch="$CUDA_ARCH" \
  -DNCCL_GIN_GDAKI_ENABLE=1 -DNCCL_GIN_PROXY_ENABLE=0 \
  -DNCCL_GIN_GPI_ENABLE=0 -DNCCL_GIN_EFA_GDA_ENABLE=0 \
  -DBENCH_GIN_SHARING=NCCL_GIN_RESOURCE_SHARING_"${GIN_SHARING:-GPU}" \
  -I"$NCCL_ROOT/include" "$task_dir/gin_nixl_bench.cu" \
  -L"$NCCL_ROOT/lib" -Xlinker -rpath -Xlinker "$NCCL_ROOT/lib" \
  -lnccl -lcuda -ldl -o "$task_binary"
sha256sum "$task_dir/gin_nixl_bench.cu" "$task_binary" "$NCCL_ROOT/lib/libnccl.so.2"
