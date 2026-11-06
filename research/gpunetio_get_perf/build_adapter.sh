#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_source="${NIXL_SOURCE:-$(cd "$task_dir/../.." && pwd)}"
task_tools=/workspace/gpunetio/kernel-copy-tools
: "${OUTPUT:?Set a fresh OUTPUT .so path}"
[[ ! -e "$OUTPUT" ]] || { echo 'Refusing binary overwrite' >&2; exit 2; }
export LD_LIBRARY_PATH="$task_tools/gdaki-compare/install/lib/aarch64-linux-gnu:$task_tools/doca35/root/usr/lib/aarch64-linux-gnu:$task_tools/doca35/root/opt/mellanox/doca/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
/usr/local/cuda/bin/nvcc -std=c++20 -arch=sm_100 --cudart shared -O3 -shared -Xcompiler -fPIC \
  -DNIXL_ENABLE_GPUNETIO_DEVICE_API=1 -DGET_PERF_TERMINAL_GATE="${TERMINAL_GATE:-0}" \
  -DCOOP_FAIL_STAGE="${COOP_FAIL_STAGE:-0}" \
  -I"$task_source/src/utils" -I"$task_source/src/api/device" -I"$task_source/src/api/cpp" \
  -I"$task_tools/doca35/root/opt/mellanox/doca/include" \
  "$task_dir/engram_native_nixl35.cu" -L"$task_tools/gdaki-compare/install/lib/aarch64-linux-gnu" \
  -lnixl -o "$OUTPUT"
sha256sum "$OUTPUT" "$task_source/src/api/device/gpu/nixl_device_batch.cuh"
