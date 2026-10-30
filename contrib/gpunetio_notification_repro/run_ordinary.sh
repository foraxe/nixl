#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NIXL contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${BASELINE_PLUGIN_DIR:?}" "${CANDIDATE_PLUGIN_DIR:?}" "${NIXL_LIBDIR:?}" "${NIC:?}" "${OOB_INTERFACE:?}" "${GID_INDEX:?}" "${SOURCE_GPU:?}" "${TARGET_GPU:?}"
[[ "$SOURCE_GPU" != "$TARGET_GPU" ]] || { echo 'Select two distinct GPUs' >&2;exit 2; }
r=${RESULT_DIR:-$(mktemp -d)};mkdir -p "$r"
bin=${ORDINARY_NOTIF_BIN:-./ordinary-notif}
export CUDA_MODULE_LOADING=EAGER NIXL_LOG_LEVEL=ERROR LD_LIBRARY_PATH="$NIXL_LIBDIR:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD CUDA_INJECTION64_PATH NVBIT_TOOL
sha256sum "$bin" "$NIXL_LIBDIR/libnixl.so" "$BASELINE_PLUGIN_DIR/libplugin_GPUNETIO.so" "$CANDIDATE_PLUGIN_DIR/libplugin_GPUNETIO.so" > "$r/before.sha256"
for repeat in 1 2 3 4 5;do
 arms=(baseline candidate);((repeat%2))||arms=(candidate baseline)
 for arm in "${arms[@]}";do
  plugin=$BASELINE_PLUGIN_DIR;[[ "$arm" = candidate ]]&&plugin=$CANDIDATE_PLUGIN_DIR
  port=$(python3 - <<'PY'
import random,socket
for _ in range(100):
 p=random.randrange(12000,28000);ss=[]
 try:
  for n in range(p,p+2):
   q=socket.socket();ss.append(q);q.bind(('0.0.0.0',n))
  print(p);break
 except OSError:pass
 finally:
  for q in ss:q.close()
else:raise RuntimeError('no free port range')
PY
)
  prefix=();[[ -z ${CPUSET:-} ]]||prefix=(taskset -c "$CPUSET")
  pre="$r/$repeat-$arm";status=0
  CUDA_VISIBLE_DEVICES="$SOURCE_GPU,$TARGET_GPU" NIXL_PLUGIN_DIR="$plugin" "${prefix[@]}" timeout --kill-after=5s 90s "$bin" "$NIC" "$OOB_INTERFACE" "$GID_INDEX" "$port" > "$pre.log" 2>&1 || status=$?
  echo "$status" > "$pre.exit";if ((status));then tail -25 "$pre.log";exit "$status";fi
  grep 'ORDINARY_NOTIF_PASS sent=4096 received=4096 timed=3968' "$pre.log"
 done
done
sha256sum "$bin" "$NIXL_LIBDIR/libnixl.so" "$BASELINE_PLUGIN_DIR/libplugin_GPUNETIO.so" "$CANDIDATE_PLUGIN_DIR/libplugin_GPUNETIO.so" > "$r/after.sha256";cmp "$r/before.sha256" "$r/after.sha256"
python3 - "$r" <<'PY'
import json,re,statistics,sys
from pathlib import Path
r=Path(sys.argv[1]);results={}
for arm in ('baseline','candidate'):
 results[arm]=[]
 for i in range(1,6):
  p=r/f'{i}-{arm}'
  if p.with_suffix('.exit').read_text().strip()!='0':raise RuntimeError(f'worker failed: {p}')
  m=re.search(r'ORDINARY_NOTIF_PASS sent=4096 received=4096 timed=3968 mean_us=([\d.]+) p50_us=([\d.]+) p99_us=([\d.]+)',p.with_suffix('.log').read_text())
  if m is None:raise RuntimeError(f'missing or malformed result: {p}')
  results[arm].append(dict(zip(('mean_us','p50_us','p99_us'),map(float,m.groups()))))
summary={}
for metric in ('mean_us','p50_us','p99_us'):
 base=[x[metric] for x in results['baseline']];candidate=[x[metric] for x in results['candidate']];pct=[100*(c/b-1) for b,c in zip(base,candidate)]
 summary[metric]={'baseline_median':statistics.median(base),'candidate_median':statistics.median(candidate),'median_paired_pct':statistics.median(pct),'candidate_slower_pairs':sum(x>0 for x in pct),'paired_pct':pct}
report={'runs':results,'summary':summary};(r/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))
PY
printf 'Results: %s\n' "$r"
