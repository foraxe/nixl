#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  owner) task_rank=0; task_addr=0.0.0.0 ;;
  reader) task_rank=1; task_addr="${OWNER_ADDR:?Set OWNER_ADDR to the owner IPv4}" ;;
  *) echo 'usage: bash run_existing_rank.sh owner|reader' >&2; exit 2 ;;
esac
: "${NCCL_ROOT:?Set NCCL_ROOT to the matching build prefix}"
: "${NIXL_SHIM:?Set NIXL_SHIM to the existing engram_* device-GET shim .so}"
task_binary="${BENCH_BINARY:-$task_dir/gin_nixl_bench_gdaki}"
task_output="${RESULT_DIR:-$task_dir/results}/${RUN_TAG:?Set the same unique RUN_TAG on both ranks}/rank$task_rank"
mkdir -p "$(dirname -- "$task_output")"
mkdir "$task_output" # Never overwrite an earlier run.
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export NCCL_GIN_TYPE=3 NCCL_GIN_GDAKI_NIC_HANDLER=2
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,REG
export NCCL_DEBUG_FILE="$task_output/nccl.%h.%p.log"
# Retain the selected NIC/GID/interface settings and NIXL library paths.
# GIN_VARIANT labels provenance only; it does NOT select a library or queue depth.
{
  printf 'variant=%s\nNCCL_ROOT=%s\nNIXL_SHIM=%s\n' "${GIN_VARIANT:-stock}" "$NCCL_ROOT" "$NIXL_SHIM"
  sha256sum "$task_dir/gin_nixl_bench.cu" "$task_binary" "$NIXL_SHIM" "$NCCL_ROOT/lib/libnccl.so.2"
  [[ ! -f "$task_dir/engram_native_nixl35.cu" ]] || sha256sum "$task_dir/engram_native_nixl35.cu"
  [[ ! -f "${NIXL_PLUGIN_DIR:-}/libplugin_GPUNETIO.so" ]] || sha256sum "$NIXL_PLUGIN_DIR/libplugin_GPUNETIO.so"
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv
  env | sort | sed -n '/^NCCL_/p; /^NIXL_/p; /^GPUNETIO_/p; /^BENCH_/p; /^CUDA_VISIBLE_DEVICES=/p; /^LD_LIBRARY_PATH=/p'
} > "$task_output/provenance.txt"
set +e
timeout --signal=TERM --kill-after=15s "${TIMEOUT_SECONDS:-300}s" \
  "$task_binary" "$task_rank" "$task_addr" "${CONTROL_PORT:-29580}" \
  "${NIXL_PORT:-29581}" "$NIXL_SHIM" "$task_output/results.json" \
  "${WARMUPS:-20}" "${SAMPLES:-100}" 2>&1 | tee "$task_output/stdout.log"
task_status=${PIPESTATUS[0]}
set -e
[[ "$task_status" == 0 ]] || exit "$task_status"
if [[ "$task_rank" == 1 && -n "${BENCH_EXPECT_COOP_FAILURE:-}" ]]; then
  grep -q "EXPECTED_COOP_FAILURE stage=$BENCH_EXPECT_COOP_FAILURE sticky=1" "$task_output/stdout.log"
  exit 0
fi
if [[ "$task_rank" == 0 && -n "${BENCH_EXPECT_COOP_FAILURE:-}" ]]; then
  grep -q 'EXPECTED_REMOTE_COOP_FAILURE' "$task_output/stdout.log"
  exit 0
fi
grep -q "GDAKI_VERIFIED rank=$task_rank " "$task_output/stdout.log"
grep -q "CLEAN_CLOSE rank=$task_rank" "$task_output/stdout.log"
if [[ "$task_rank" == 1 ]]; then
  if [[ -n "${BENCH_EXTRA_ROWS:-}" ]]; then
    grep -q "PASS rows=$BENCH_EXTRA_ROWS arms=4 samples=${SAMPLES:-100}" "$task_output/stdout.log"
    printf 'EXTRA_ROWS_VALIDATED=%s\n' "$BENCH_EXTRA_ROWS"
    exit 0
  fi
  python3 "$task_dir/summarize.py" "$task_output/results.json"
  printf 'RESULT_JSON=%s\n' "$task_output/results.json"
fi
