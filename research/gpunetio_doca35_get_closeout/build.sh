#!/usr/bin/env bash
set -euo pipefail
# Build on a CUDA/DOCA host after building the exact NIXL review core/plugin.
# NIXL_REPO: nixl-get-closeout-20261117, NIXL_LIBDIR: matching libnixl directory,
# DOCA_INCLUDE: directory containing DOCA headers. OUT_DIR defaults to ./build.
: "${NIXL_REPO:?set NIXL_REPO}" "${NIXL_LIBDIR:?set NIXL_LIBDIR}" "${DOCA_INCLUDE:?set DOCA_INCLUDE}"
expected=e56e4afa985ee2997f78675ae897c964d1d3568f
here=$(cd "$(dirname "$0")" && pwd)
if git -C "$NIXL_REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  [[ $(git -C "$NIXL_REPO" rev-parse HEAD) == "$expected" ]] || { echo 'unexpected NIXL HEAD' >&2; exit 1; }
  [[ -z $(git -C "$NIXL_REPO" status --porcelain -- src) ]] || { echo 'NIXL sources modified' >&2; exit 1; }
else
  [[ $(cat "$NIXL_REPO/SOURCE_COMMIT") == "$expected" ]] || { echo 'unexpected exported source identity' >&2; exit 1; }
fi
(cd "$NIXL_REPO" && sha256sum -c "$here/NIXL_SRC_SHA256SUMS" >/dev/null)
out=${OUT_DIR:-"$here/build"}
mkdir -p "$out"
nvcc=${NVCC:-nvcc}
"$nvcc" -std=c++20 -O2 -DNDEBUG -DNIXL_ENABLE_GPUNETIO_DEVICE_API=1 -DNIXL_GPUNETIO_PUBLIC_FIFO_ONLY=1 -arch=sm_90 -shared -Xcompiler=-fPIC -I"$NIXL_REPO/src/utils" -I"$NIXL_REPO/src/api/device" -I"$NIXL_REPO/src/api/cpp" -I"$DOCA_INCLUDE" "$here/src/engram_multirail_fifo_adaptive_base.cu" -L"$NIXL_LIBDIR" -lnixl -ldl -o "$out/libengram_multirail_base.so"
"$nvcc" -std=c++20 -O2 -DNDEBUG -arch=sm_90 -shared -Xcompiler=-fPIC "$here/src/engram_abi3_multirail_dedup_auto.cu" -ldl -o "$out/libengram_abi3_multirail.so"
shasum -a 256 "$here"/src/*.cu "$out"/*.so
"$nvcc" --version
# Check the complete delivered C ABI is present in the direct candidate.
for symbol in engram_last_error engram_abi_version engram_create engram_register engram_metadata engram_connect engram_fetch engram_fetch_row264 engram_fetch_row264_pending engram_set_cost_trace engram_set_trace engram_read_group_terminal engram_read_group_pipeline engram_read_group_stagger engram_read_group_create engram_read_group_fetch engram_read_group_fetch_mapped engram_read_group_fetch_compacted engram_read_group_destroy engram_destroy engram_record_consumed; do
  nm -D --defined-only "$out/libengram_multirail_base.so" | awk '{print $3}' | grep -Fxq "$symbol" || { echo "missing ABI symbol: $symbol" >&2; exit 1; }
done
