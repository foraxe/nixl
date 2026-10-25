#!/usr/bin/env bash
set -euo pipefail
tag=${1:?unique tag}; shift
[[ $tag =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
here=$(cd "$(dirname "$0")" && pwd)
out="$here/evidence/$tag"
[[ ! -e $out ]] || exit 2
mkdir -p "$out"
tools_root=/workspace/gpunetio/kernel-copy-tools
build="$tools_root/primitive${PRIMITIVE_BUILD_SUFFIX:-}"
export CUDA_VISIBLE_DEVICES=0
export DIRECT_COPY_LIBRARY="$build/liblocal_copy.so"
if [[ -n ${PRIMITIVE_TEST_TIMEOUT:-} ]] && ! nm -D "$DIRECT_COPY_LIBRARY" | rg ' T lc_test_fail_reset$' > /dev/null; then
  echo "Timeout injection requires a LOCAL_COPY_TESTING library" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$build:$tools_root/e/install/lib/aarch64-linux-gnu/plugins:$tools_root/e/install/lib/aarch64-linux-gnu:$tools_root/doca35/root/opt/mellanox/doca/lib/aarch64-linux-gnu:$tools_root/doca35/root/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/workspace/gpunetio/nccl-core-workload/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
sha256sum "$build/run" "$DIRECT_COPY_LIBRARY" "$here/run.cu" "$here/in_kernel.cu" "$here/example.cu" "$here/abi.cuh" "$here/tma_control.cuh" "$here/common.cuh" "$here/../../workdir/local_copy.cuh" "$here/../../workdir/kernel_copy.cuh" > "$out/hashes.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv > "$out/gpus.csv"
env | sort | rg '^(CUDA_|PRIMITIVE_|DIRECT_COPY_|LD_LIBRARY_PATH=)' > "$out/environment.txt"
cd "$out"
timeout --kill-after=10s "${PRIMITIVE_TIMEOUT:-900}" taskset -c 0-7 "$build/run" "$@" > results.jsonl 2> errors.log
tail -1 results.jsonl
