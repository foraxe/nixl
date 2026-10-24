#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if (($# != 2 && $# != 4)); then
    echo "usage: $0 BUILD_DIR NEW_RESULT_DIR [IMMUTABLE_PARENT_BINARY PARENT_SHA256]" >&2
    exit 2
fi
build=$(realpath "$1")
exe="$build/examples/gpunetio_device_owner/owner-endpoint-exclusive"
test -x "$exe"
: "${OWNER_REPRO_RDMA_DEVICE:?set the RDMA device}"
: "${OWNER_REPRO_OOB_INTERFACE:?set the OOB interface}"
: "${OWNER_REPRO_GID_INDEX:?set the fabric GID index}"
modes=(mode0 mode1 mode2)
analyzer_args=()
parent=
if (($# == 4)); then
    parent=$(realpath "$3")
    test -x "$parent"
    [[ "$parent" != "$exe" && "$4" =~ ^[[:xdigit:]]{64}$ ]] || {
        echo "parent must be a distinct executable with a known SHA256" >&2; exit 2;
    }
    actual=$(sha256sum "$parent" | cut -d ' ' -f 1)
    [[ "$actual" == "$(printf '%s' "$4" | tr '[:upper:]' '[:lower:]')" ]] || {
        echo "parent SHA256 mismatch" >&2; exit 2;
    }
    modes+=(parent)
    analyzer_args+=(--include-parent)
fi
mkdir "$2"
results=$(realpath "$2")
sha256sum "$exe" > "$results/executables.sha256"
if [[ -n "$parent" ]]; then sha256sum "$parent" >> "$results/executables.sha256"; fi
export OWNER_REPRO_BATCH_COMMIT=1 OWNER_REPRO_PRECOMMIT_CHECK=0
unset OWNER_REPRO_WARP_QUALIFY
printf '%s\n' 'Recorded by run_preparation_comparison.sh before launching each block.' \
    > "$results/order_provenance.txt"
source_prefix=(); receiver_prefix=()
if [[ -n ${OWNER_REPRO_SOURCE_CPU:-} ]]; then source_prefix=(taskset -c "$OWNER_REPRO_SOURCE_CPU"); fi
if [[ -n ${OWNER_REPRO_RECEIVER_CPU:-} ]]; then receiver_prefix=(taskset -c "$OWNER_REPRO_RECEIVER_CPU"); fi
sender= receiver=
cleanup() {
    for pid in "$sender" "$receiver"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
for pair in 0 1 2 3 4; do
    order=()
    for ((i=0; i<${#modes[@]}; i++)); do
        order+=("${modes[(pair+i)%${#modes[@]}]}")
    done
    if ((pair%2)); then
        reversed=()
        for ((i=${#order[@]}-1; i>=0; i--)); do reversed+=("${order[i]}"); done
        order=("${reversed[@]}")
    fi
    printf '%s\n' "${order[@]}" > "$results/pair$pair.order"
    for mode in "${order[@]}"; do
        selected=$exe
        if [[ "$mode" == parent ]]; then
            selected=$parent
            unset OWNER_REPRO_PREP_MODE
        else
            export OWNER_REPRO_PREP_MODE=${mode#mode}
        fi
        stem="pair$pair-$mode"
        coord="$results/$stem.coord"
        mkdir "$coord"
        port=$(python3 - <<'PY'
import random
import socket

for _ in range(100):
    port = random.randrange(24000, 30000)
    sockets = []
    try:
        for number in (port, port + 1):
            sock = socket.socket()
            sockets.append(sock)
            sock.bind(('0.0.0.0', number))
        print(port)
        break
    except OSError:
        pass
    finally:
        for sock in sockets:
            sock.close()
else:
    raise RuntimeError('no available OOB ports')
PY
)
        "${receiver_prefix[@]}" timeout 180s "$selected" receiver "$coord" "$stem" 1 "$port" \
            > "$results/$stem.receiver.log" 2>&1 & receiver=$!
        "${source_prefix[@]}" timeout 180s "$selected" sender "$coord" "$stem" 0 "$((port+1))" \
            > "$results/$stem.sender.log" 2>&1 & sender=$!
        tx=0; rx=0
        wait "$sender" || tx=$?
        wait "$receiver" || rx=$?
        sender= receiver=
        printf 'SOURCE=%s TARGET=%s\n' "$tx" "$rx" > "$results/$stem.exit"
        if ((tx || rx)); then echo "$stem failed: sender=$tx receiver=$rx" >&2; exit 1; fi
        echo "PREPARATION_RUN_PASS $stem"
    done
done
sha256sum -c "$results/executables.sha256"
python3 "$(dirname "$0")/compare_preparation.py" "$results" "${analyzer_args[@]}" \
    | tee "$results/summary.json"
