#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_tools=/workspace/gpunetio/kernel-copy-tools
task_tag=${1:?unique experiment prefix}
[[ "$task_tag" =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
export NCCL_GIN_GDAKI_QP_DEPTH=2048 NIXL_LOG_LEVEL=info
export NIXL_PLUGIN_DIR="$task_tools/get-perf/build/src/plugins/gpunetio"
export BENCH_BINARY="$task_tools/get-perf/gin_nixl_bench_gate"
export NIXL_SHIM="$task_tools/get-perf/adapter_gate.so"
export WARMUPS=20 SAMPLES=100
for task_pair in 1 2 3; do
  for task_arm in pre relaxed post; do
    task_run="$task_tag-$task_pair-$task_arm"
    [[ ! -e "$task_tools/$task_run.log" ]] || exit 2
    if [[ "$task_arm" == relaxed ]]; then
      export GPUNETIO_DEVICE_API_RELAXED_ORDERING=true
    else
      export GPUNETIO_DEVICE_API_RELAXED_ORDERING=false
    fi
    python3 "$task_dir/run_local.py" "$task_run" > "$task_tools/$task_run.log" 2>&1
    printf 'Completed %s\n' "$task_run"
  done
done
