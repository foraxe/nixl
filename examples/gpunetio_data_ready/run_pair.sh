#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if (($# != 3)); then
  echo "usage: $0 SNAPSHOT_DIR BINARY RESULT_DIR" >&2
  exit 2
fi
SNAPSHOT_DIR=$(cd "$1" && pwd -P)
BINARY=$(realpath "$2")
mkdir -p "$3"
RESULT_DIR=$(cd "$3" && pwd -P)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)

[[ -x "$BINARY" ]] || { echo "not executable: $BINARY" >&2; exit 2; }
[[ -f "$SNAPSHOT_DIR/libnixl_device_allocator_cuda.so" ]] || {
  echo "snapshot lacks libnixl_device_allocator_cuda.so: $SNAPSHOT_DIR" >&2; exit 2;
}

SDK_ROOT=${SDK_ROOT:-}
CUDA_COMPAT_LIB=${CUDA_COMPAT_LIB:-/usr/local/cuda/compat/lib}
CUDA_LIB_DIR=${CUDA_LIB_DIR:-/usr/local/cuda/targets/x86_64-linux/lib}
DOCA_LIB_DIR=${DOCA_LIB_DIR:-}
OOB_INTERFACE=${OOB_INTERFACE:-eth0}
RDMA_DEVICE=${RDMA_DEVICE:?Set RDMA_DEVICE for your machine}
PHYSICAL_GPUS=${PHYSICAL_GPUS:-0,1}
GID_INDEX=${GID_INDEX:-0}
CONTROL_PORT=${CONTROL_PORT:-6546}
CONSUMER_OOB_PORT=${CONSUMER_OOB_PORT:-6544}
SOURCE_OOB_PORT=${SOURCE_OOB_PORT:-6545}
NEGATIVE_TIMEOUT_MS=${NEGATIVE_TIMEOUT_MS:-5000}

LOCAL_IP=${LOCAL_IP:-}
if [[ -z "$LOCAL_IP" ]] && command -v ip >/dev/null 2>&1; then
  LOCAL_IP=$(ip -o -4 addr show dev "$OOB_INTERFACE" scope global 2>/dev/null |
    awk 'NR==1 {sub(/\/.*/, "", $4); print $4}')
fi
if [[ -z "$LOCAL_IP" ]]; then
  read -r -a host_ips <<<"$(hostname -I 2>/dev/null || true)"
  if ((${#host_ips[@]} == 1)); then
    LOCAL_IP=${host_ips[0]}
  else
    echo "cannot select one IPv4 on $OOB_INTERFACE; hostname -I returned ${#host_ips[@]} addresses. Set LOCAL_IP explicitly." >&2
    exit 2
  fi
fi
if [[ ! "$LOCAL_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "LOCAL_IP must be a numeric IPv4 address: $LOCAL_IP" >&2
  exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$PHYSICAL_GPUS
export NIXL_PLUGIN_DIR=$SNAPSHOT_DIR
export LD_LIBRARY_PATH="$CUDA_COMPAT_LIB:$SNAPSHOT_DIR${DOCA_LIB_DIR:+:$DOCA_LIB_DIR}${SDK_ROOT:+:$SDK_ROOT/usr/lib/x86_64-linux-gnu}:$CUDA_LIB_DIR:/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_pair() {
  local name=$1 epoch=$2 mode=$3 first=$4 episodes=$5 warmups=$6 steps=$7 timeout_ms=$8 expected=$9
  local inject=${10}
  local coord="$RESULT_DIR/coord-$name"
  local source_log="$RESULT_DIR/$name.source.log" consumer_log="$RESULT_DIR/$name.consumer.log"
  mkdir "$coord"
  printf 'first_arm=%s\nepoch=%s\nmode=%s\n' "$first" "$epoch" "$mode" >"$RESULT_DIR/$name.meta"

  local common=(--coord "$coord" --run "$name" --epoch "$epoch" --steps "$steps"
    --arm "$mode" --episodes "$episodes" --warmups "$warmups" --first-arm "$first"
    --control-port "$CONTROL_PORT" --rdma "$RDMA_DEVICE" --oob-interface "$OOB_INTERFACE"
    --gid-index "$GID_INDEX" --timeout-ms "$timeout_ms")
  [[ -z "$inject" ]] || common+=(--inject "$inject")

  "$BINARY" --role consumer "${common[@]}" --gpu 1 --oob-port "$CONSUMER_OOB_PORT" \
    >"$consumer_log" 2>&1 &
  local consumer_pid=$!
  "$BINARY" --role source "${common[@]}" --gpu 0 --oob-port "$SOURCE_OOB_PORT" \
    --target-ip "$LOCAL_IP" >"$source_log" 2>&1 &
  local source_pid=$!

  local consumer_rc=0 source_rc=0
  wait "$consumer_pid" || consumer_rc=$?
  wait "$source_pid" || source_rc=$?
  printf '%s\n' "$consumer_rc" >"$RESULT_DIR/$name.consumer.rc"
  printf '%s\n' "$source_rc" >"$RESULT_DIR/$name.source.rc"

  if [[ $expected == pass ]]; then
    [[ $consumer_rc -eq 0 && $source_rc -eq 0 ]] || {
      echo "$name failed: consumer=$consumer_rc source=$source_rc ($consumer_log, $source_log)" >&2; return 1;
    }
  else
    [[ $consumer_rc -eq 1 && $source_rc -eq 1 ]] || {
      echo "$name expected protocol exits 1/1, got consumer=$consumer_rc source=$source_rc" >&2; return 1;
    }
    grep -q 'validation=FAIL' "$consumer_log" && grep -q 'validation=FAIL' "$source_log" || {
      echo "$name failed outside expected GPU validation gates; inspect logs" >&2; return 1;
    }
  fi
  echo "$name expected=$expected consumer_rc=$consumer_rc source_rc=$source_rc"
}

echo "snapshot=$SNAPSHOT_DIR binary=$BINARY result_dir=$RESULT_DIR"
echo "physical GPUs=${CUDA_VISIBLE_DEVICES}; source=logical0 consumer=logical1"
echo "rdma=$RDMA_DEVICE gid=$GID_INDEX oob=$OOB_INTERFACE local_ip=$LOCAL_IP"
sha256sum "$BINARY" "$SNAPSHOT_DIR"/*.so > "$RESULT_DIR/runtime.sha256"

epoch_base=${EPOCH_BASE:-$(date +%s)}
stamp=$(date +%Y%m%dT%H%M%S)-$$
for run in 1 2 3 4 5; do
  if ((run % 2)); then first=ordered; else first=cqwait; fi
  printf -v tag 'run-%02d-%s' "$run" "$stamp"
  run_pair "$tag" "$((epoch_base + run * 100000))" paired "$first" 50 1 20 45000 pass ""
done

python3 "$SCRIPT_DIR/analyze_pair_results.py" "$RESULT_DIR"

if [[ ${PAIRED_ONLY:-0} != 1 ]]; then
  for inject in payload ready-epoch ack-epoch; do
    run_pair "negative-$inject" "$((epoch_base + 900000 + RANDOM))" ordered ordered 1 0 1 "$NEGATIVE_TIMEOUT_MS" fail "$inject"
  done
fi

sha256sum -c "$RESULT_DIR/runtime.sha256"
if [[ ${PAIRED_ONLY:-0} == 1 ]]; then
  echo "PASS: five paired runs; results=$RESULT_DIR"
else
  echo "PASS: five paired runs and three fail-closed injections; results=$RESULT_DIR"
fi
