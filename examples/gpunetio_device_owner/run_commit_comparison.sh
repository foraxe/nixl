#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if (($# != 2)); then
    echo "usage: $0 BUILD_DIR NEW_RESULT_DIR" >&2
    exit 2
fi
build=$(realpath "$1")
results=$2
mkdir "$results"
results=$(realpath "$results")
exe="$build/examples/gpunetio_device_owner/owner-endpoint-exclusive"
test -x "$exe"
: "${OWNER_REPRO_RDMA_DEVICE:?set the RDMA device}"
: "${OWNER_REPRO_OOB_INTERFACE:?set the OOB interface}"
: "${OWNER_REPRO_GID_INDEX:?set the fabric GID index}"
export OWNER_REPRO_PRECOMMIT_CHECK=0
unset OWNER_REPRO_PREP_MODE
unset OWNER_REPRO_WARP_QUALIFY
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
sha256sum "$exe" > "$results/executable.sha256"
for pair in 0 1 2 3 4; do
    modes=(fixed grouped)
    if ((pair%2)); then modes=(grouped fixed); fi
    for mode in "${modes[@]}"; do
        grouped=0; if [[ $mode == grouped ]]; then grouped=1; fi
        export OWNER_REPRO_BATCH_COMMIT=$grouped
        stem="pair$pair-$mode"
        coord="$results/$stem.coord"
        mkdir "$coord"
        port=$(python3 - <<'PY'
import random, socket
for _ in range(100):
    port = random.randrange(24000, 30000)
    sockets = []
    try:
        for number in (port, port + 1):
            sock = socket.socket(); sockets.append(sock); sock.bind(('0.0.0.0', number))
        print(port); break
    except OSError: pass
    finally:
        for sock in sockets: sock.close()
else: raise RuntimeError('no available OOB ports')
PY
)
        "${receiver_prefix[@]}" timeout 180s "$exe" receiver "$coord" "$stem" 1 "$port" \
            > "$results/$stem.receiver.log" 2>&1 & receiver=$!
        "${source_prefix[@]}" timeout 180s "$exe" sender "$coord" "$stem" 0 "$((port + 1))" \
            > "$results/$stem.sender.log" 2>&1 & sender=$!
        tx=0; rx=0
        wait "$sender" || tx=$?
        wait "$receiver" || rx=$?
        sender= receiver=
        printf 'SOURCE=%s TARGET=%s\n' "$tx" "$rx" > "$results/$stem.exit"
        if ((tx || rx)); then echo "$stem failed: sender=$tx receiver=$rx" >&2; exit 1; fi
        echo "COMMIT_PAIR_PASS $stem"
    done
done
sha256sum -c "$results/executable.sha256"
python3 "$(dirname "$0")/compare_commits.py" "$results" | tee "$results/summary.json"
