#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${QP_TEST:?path to the compiled replay}"
: "${MAIN_PLUGIN:?directory containing the main plugin}"
: "${CANDIDATE_PLUGIN:?directory containing the candidate plugin}"
: "${SOURCE_GPU:?one physical GPU ordinal or UUID}"
: "${TARGET_GPUS:?two other comma-separated GPU ordinals or UUIDs}"
: "${SOURCE_CPU:?source CPU affinity}"
: "${TARGET_CPU:?target CPU affinity}"
out=${RESULTS_DIR:-$(mktemp -d)}
mkdir -p "$out"
export CUDA_MODULE_LOADING=LAZY QP_WARMUP=1200
unset CUDA_DEVICE_MAX_CONNECTIONS LD_PRELOAD CUDA_INJECTION64_PATH NVBIT_TOOL QP_GRAPH_INTERPOSER
export NIXL_QP_PROGRESS_TARGET_IPV4=127.0.0.1
export NIXL_QP_PROGRESS_DATA_QPS=1
source_pid= target_pid=
cleanup() {
    for pid in "$source_pid" "$target_pid"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
for shape in 4k 2m mixed; do
    export QP_MEASURED=1000 NIXL_QP_PROGRESS_CONTROL_BYTES=4096
    filter=SingleQpReadWriteControl
    if [[ "$shape" == 4k ]]; then export QP_MEASURED=10000; fi
    if [[ "$shape" == 2m ]]; then export NIXL_QP_PROGRESS_CONTROL_BYTES=2097152; fi
    if [[ "$shape" == mixed ]]; then filter=PerformanceMixed2MiBAnd4KiB; fi
    for repeat in 1 2 3 4 5; do
        arms=(main candidate)
        if ((repeat % 2 == 0)); then arms=(candidate main); fi
        for arm in "${arms[@]}"; do
            export NIXL_PLUGIN_DIR="$MAIN_PLUGIN"
            if [[ "$arm" == candidate ]]; then export NIXL_PLUGIN_DIR="$CANDIDATE_PLUGIN"; fi
            export NIXL_QP_PROGRESS_COORD_DIR
            NIXL_QP_PROGRESS_COORD_DIR=$(mktemp -d "$out/$shape-$repeat-$arm.XXXXXXXX")
            port=$(python3 - <<'PY'
import random, socket
for _ in range(100):
    base = random.randrange(17000, 24000)
    sockets = []
    try:
        for p in range(base, base + 4):
            s = socket.socket(); sockets.append(s); s.bind(('0.0.0.0', p))
        print(base); break
    except OSError:
        pass
    finally:
        for s in sockets: s.close()
else:
    raise RuntimeError('no free four-port range')
PY
)
            export NIXL_QP_PROGRESS_CONTROL_PORT="$port"
            export NIXL_QP_PROGRESS_SOURCE_OOB_PORT="$((port+1))"
            export NIXL_QP_PROGRESS_TARGET_A_OOB_PORT="$((port+2))"
            export NIXL_QP_PROGRESS_TARGET_B_OOB_PORT="$((port+3))"
            sha256sum "$QP_TEST" "$NIXL_PLUGIN_DIR/libplugin_GPUNETIO.so" > "$NIXL_QP_PROGRESS_COORD_DIR/identities.sha256"
            CUDA_VISIBLE_DEVICES="$TARGET_GPUS" NIXL_QP_PROGRESS_ROLE=target \
                taskset -c "$TARGET_CPU" timeout --kill-after=3s 120s "$QP_TEST" \
                --gtest_filter="QpProgress.$filter" > "$NIXL_QP_PROGRESS_COORD_DIR/target.log" 2>&1 & target_pid=$!
            CUDA_VISIBLE_DEVICES="$SOURCE_GPU" NIXL_QP_PROGRESS_ROLE=source \
                taskset -c "$SOURCE_CPU" timeout --kill-after=3s 120s "$QP_TEST" \
                --gtest_filter="QpProgress.$filter" > "$NIXL_QP_PROGRESS_COORD_DIR/source.log" 2>&1 & source_pid=$!
            a=0; b=0
            wait "$source_pid" || a=$?
            wait "$target_pid" || b=$?
            source_pid=; target_pid=
            printf 'SOURCE=%s TARGET=%s\n' "$a" "$b" > "$NIXL_QP_PROGRESS_COORD_DIR/exits.txt"
            ((a == 0 && b == 0))
            for side in source target; do
                grep -q '\[  PASSED  \] 1 test' "$NIXL_QP_PROGRESS_COORD_DIR/$side.log"
                if grep -q '\[  SKIPPED \]' "$NIXL_QP_PROGRESS_COORD_DIR/$side.log"; then exit 1; fi
            done
            echo "PASS shape=$shape repeat=$repeat arm=$arm path=$NIXL_QP_PROGRESS_COORD_DIR"
        done
    done
done
echo "RESULTS=$out"
