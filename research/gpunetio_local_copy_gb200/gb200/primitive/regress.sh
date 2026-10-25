#!/usr/bin/env bash
set -euo pipefail
tag=${1:?unique regression tag}
[[ $tag =~ ^[a-zA-Z0-9_.-]+$ ]] || exit 2
here=$(cd "$(dirname "$0")" && pwd)
out="$here/evidence/$tag"
[[ ! -e $out ]] || exit 2
mkdir -p "$out"
tools_root=/workspace/gpunetio/kernel-copy-tools
normal="$tools_root/primitive${PRIMITIVE_BUILD_SUFFIX:--prreview}"
testing="$tools_root/primitive${PRIMITIVE_TESTING_SUFFIX:--prreview-testing}"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="$normal:$tools_root/e/install/lib/aarch64-linux-gnu/plugins:$tools_root/e/install/lib/aarch64-linux-gnu:$tools_root/doca35/root/opt/mellanox/doca/lib/aarch64-linux-gnu:$tools_root/doca35/root/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
timeout --kill-after=10s 240 python3 "$here/../../workdir/test_copy.py" \
  --gpu 0 --nic mlx5_0 --gid 3 --library "$normal/liblocal_copy.so" > "$out/core.json"
timeout --kill-after=10s 60 python3 "$here/../../workdir/test_quarantine.py" \
  --gpu 0 --nic mlx5_0 --gid 3 --library "$testing/liblocal_copy.so" > "$out/quarantine.json"
sha256sum "$normal/liblocal_copy.so" "$testing/liblocal_copy.so" > "$out/hashes.sha256"
echo "PASS: core regressions and reset-failure quarantine"
