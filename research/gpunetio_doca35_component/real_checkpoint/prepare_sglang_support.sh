#!/usr/bin/env bash
# Assemble the tested research reader modules into a new isolated directory.
set -euo pipefail
: "${TARGET:?Set TARGET to a new directory}"
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
component=$(cd -- "$here/.." && pwd)
legacy=$(cd -- "$component/../engram_shared_rdma_m2" && pwd)
mkdir -- "$TARGET"
cp "$here/sglang_support/"*.py "$TARGET/"
cp "$legacy/engram_transport.py" "$TARGET/sglang_runtime.py"
cp "$legacy/row264_kernels.py" "$TARGET/"
cp "$component/"{read_group,read_lanes,dedup_rows}.py "$TARGET/"
