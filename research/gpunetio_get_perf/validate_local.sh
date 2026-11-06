#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_tools=/workspace/gpunetio/kernel-copy-tools
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NIXL_PLUGIN_DIR="$task_tools/get-perf/build/src/plugins/gpunetio"
export LD_LIBRARY_PATH="$task_tools/gdaki-compare/install/lib/aarch64-linux-gnu/plugins:$task_tools/gdaki-compare/install/lib/aarch64-linux-gnu:$task_tools/doca35/root/opt/mellanox/doca/lib/aarch64-linux-gnu:$task_tools/doca35/root/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export NIXL_LOG_LEVEL=info
task_out="$task_dir/results/${1:?unique validation tag}"
mkdir "$task_out"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 3
"$task_tools/get-perf/registration_access_test" > "$task_out/access-policy.log" 2>&1
task_port=31700
for task_order in strict relaxed; do
  for task_memory in DRAM VRAM; do
    if [[ "$task_order" == relaxed ]]; then
      export NIXL_GPUNETIO_TEST_RELAXED_ORDERING=1
    else
      unset NIXL_GPUNETIO_TEST_RELAXED_ORDERING
    fi
    timeout --kill-after=15s 90s taskset -c 0-15 "$task_tools/get-perf/device_get_test" \
      mlx5_0 lo 3 "$task_port" 0 "$task_memory" 1 > "$task_out/$task_order-$task_memory.log" 2>&1
    task_port=$((task_port + 2))
  done
done
sha256sum "$task_tools/get-perf/device_get_test" "$NIXL_PLUGIN_DIR/libplugin_GPUNETIO.so" > "$task_out/hashes.sha256"
printf 'PASS CPU registration policy and strict/relaxed x Host/VRAM scalar/batch GPU consumption\n'
