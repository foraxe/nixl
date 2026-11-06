#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_tools=/workspace/gpunetio/kernel-copy-tools
export NCCL_GIN_GDAKI_QP_DEPTH=2048 NIXL_LOG_LEVEL=info
export NIXL_PLUGIN_DIR="$task_tools/get-pool/build/src/plugins/gpunetio"
export GPUNETIO_DEVICE_API_RELAXED_ORDERING=true
export NIXL_SHIM="$task_tools/get-pool/adapter_pool_final.so"
export WARMUPS=20 SAMPLES=100
unset BENCH_EXPECT_COOP_FAILURE BENCH_EXTRA_ROWS
task_prefix=${1:?unique prefix}
for task_trial in 1 2 3; do
  for task_case in p1 p32 i32c4 p32c4; do
    task_binary=$task_case
    unset BENCH_NIXL_SINGLE_CTA
    case "$task_case" in
      p32c4) task_binary=p32; export BENCH_NIXL_SINGLE_CTA=4 ;;
      i32c4) task_binary=i32; export BENCH_NIXL_SINGLE_CTA=4 ;;
    esac
    export BENCH_BINARY="$task_tools/get-pool/bench_$task_binary"
    task_tag="$task_prefix-$task_case-$task_trial"
    [[ ! -e "$task_tools/$task_tag.log" ]] || exit 2
    python3 "$task_dir/run_local.py" "$task_tag" > "$task_tools/$task_tag.log" 2>&1
    printf 'Completed %s\n' "$task_tag"
  done
done
