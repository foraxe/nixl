#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_tools=/workspace/gpunetio/kernel-copy-tools
task_unit="${POOL_TEST_BINARY:-$task_tools/get-pool/device_get_test_fault}"
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_out="$task_dir/results/${1:?unique validation tag}"
mkdir "$task_out"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || exit 3
export NIXL_PLUGIN_DIR="$task_tools/get-pool/build/src/plugins/gpunetio"
export LD_LIBRARY_PATH="$task_tools/gdaki-compare/install/lib/aarch64-linux-gnu/plugins:$task_tools/gdaki-compare/install/lib/aarch64-linux-gnu:$task_tools/doca35/root/opt/mellanox/doca/lib/aarch64-linux-gnu:$task_tools/doca35/root/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export NIXL_LOG_LEVEL=info
"$task_tools/get-pool/native_options_test" > "$task_out/options.log"
task_port=31800
for task_qps in 1 4; do
  for task_order in strict relaxed; do
    for task_memory in DRAM VRAM; do
      export NIXL_GPUNETIO_TEST_QPS="$task_qps"
      unset NIXL_GPUNETIO_TEST_RELAXED_ORDERING
      [[ "$task_order" != relaxed ]] || export NIXL_GPUNETIO_TEST_RELAXED_ORDERING=1
      timeout --kill-after=15s 120s taskset -c 0-15 "$task_unit" \
        mlx5_0 lo 3 "$task_port" 0 "$task_memory" 1 > "$task_out/q$task_qps-$task_order-$task_memory.log" 2>&1
      task_port=$((task_port+2))
    done
  done
done
NIXL_GPUNETIO_TEST_QPS=32 NIXL_GPUNETIO_TEST_RELAXED_ORDERING=1 \
 timeout --kill-after=15s 120s taskset -c 0-15 "$task_unit" \
 mlx5_0 lo 3 "$task_port" 0 DRAM 1 > "$task_out/q32-relaxed-DRAM.log" 2>&1
task_port=$((task_port+2))
for task_pair in '1 4' '4 1' '2 4'; do
  read -r task_owner task_reader <<< "$task_pair"
  NIXL_GPUNETIO_TEST_QPS="$task_reader" NIXL_GPUNETIO_TEST_PEER_QPS="$task_owner" \
   timeout --kill-after=10s 30s "$task_unit" \
   mlx5_0 lo 3 "$task_port" 0 DRAM 1 > "$task_out/mismatch-$task_owner-$task_reader.log" 2>&1
  task_port=$((task_port+2))
done
NIXL_GPUNETIO_TEST_QPS=4 NIXL_GPUNETIO_TEST_RELAXED_ORDERING=1 NIXL_GPUNETIO_TEST_RETENTION=1 NIXL_GPUNETIO_TEST_CQ_ERROR=1 \
 timeout --kill-after=15s 180s taskset -c 0-15 /usr/local/cuda/bin/compute-sanitizer \
 --tool memcheck --report-api-errors no --error-exitcode 86 "$task_unit" \
 mlx5_0 lo 3 "$task_port" 0 DRAM 1 > "$task_out/retention-memcheck.log" 2>&1
sha256sum "$task_unit" "$NIXL_PLUGIN_DIR/libplugin_GPUNETIO.so" > "$task_out/hashes.sha256"
printf 'PASS default/pool x strict/relaxed x Host/VRAM, channel isolation, generations, wrap, retention\n'
