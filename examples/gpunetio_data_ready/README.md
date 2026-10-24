<!-- SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Optional remote-attention consumer harness

Builds on the public native-owner API. The source directly PUTs an immutable
32 KiB FP16 K/V page, then publishes ready. The receiver system-acquires and
validates the full page/guards/route, runs the unchanged `raAttention`, and
ACKs before source-buffer reuse. Both arms keep identical QP, payload, WQEs,
commits and CQ/status checks; `cqwait` adds only the data-CQ poll before ready.

This is a same-host H20 consumer-readiness harness. It is not serving, a
production operator, or cross-node validation.

The epoch is the first word of the 64-byte ready WRITE. The tested H20
campaign is positive evidence; it does not establish atomic visibility of
all 64 bytes on other platforms. Confirm the flag/body visibility contract
before extending this layout, or evaluate a separate final flag WQE with
fresh matched controls. Full-record/payload validation fails closed.

## Build

Compile/link only this caller against the matching configured NIXL build and
DOCA 3.5 headers/libraries. The measured profile used CUDA 13.0.48, SM90, and
the public-owner API defines below.

```sh
SOURCE_ROOT=/path/to/nixl-source
BUILD_DIR=/path/to/configured-nixl-build
DOCA_ROOT=/path/to/doca-3.5
CUDA_ROOT=/usr/local/cuda
LIBNL_INCLUDE_DIR=/usr/include/libnl3
OUT_DIR=/path/to/output
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES=
"$CUDA_ROOT/bin/nvcc" -std=c++20 -O3 \
  -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 \
  -rdc=true -DNIXL_TRACE_ENABLED -DNIXL_ENABLE_GPUNETIO_DEVICE_API=1 \
  -DNIXL_GPUNETIO_DEVICE_API_PUBLIC=1 -I"$DOCA_ROOT/include" \
  -I"$LIBNL_INCLUDE_DIR" -I"$CUDA_ROOT/targets/x86_64-linux/include" \
  -I"$SOURCE_ROOT/src/api/cpp" -I"$SOURCE_ROOT/src/api/device" \
  -I"$SOURCE_ROOT/src/core" -I"$SOURCE_ROOT/src/infra" \
  -c examples/gpunetio_data_ready/remote_attention_consumer.cu \
  -o "$OUT_DIR/remote_attention_consumer.o"
"$CUDA_ROOT/bin/nvcc" -o "$OUT_DIR/remote_attention_consumer_stepmetric" \
  "$OUT_DIR/remote_attention_consumer.o" \
  -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 \
  -Xlinker="$BUILD_DIR/src/core/libnixl.so" \
  -Xlinker="$BUILD_DIR/src/infra/libnixl_build.so" \
  -Xlinker="$BUILD_DIR/src/utils/serdes/libserdes.so" \
  -Xlinker="$BUILD_DIR/src/utils/common/libnixl_common.so" \
  -Xlinker="$BUILD_DIR/src/utils/stream/libstream.so" \
  -Xlinker="$BUILD_DIR/src/utils/device/libnixl_device_allocator.so" \
  -cudart shared -Xlinker="$CUDA_ROOT/targets/x86_64-linux/lib/stubs/libcuda.so" \
  -Xlinker="$DOCA_ROOT/lib/x86_64-linux-gnu/libdoca_gpunetio.so"
```

## Run

Create a runtime directory from the same configured owner build; the core,
plugin and allocator must use matching headers and ABI:

```sh
ninja -C "$BUILD_DIR" src/utils/device/libnixl_device_allocator_cuda.so
SNAPSHOT_DIR=$(mktemp -d)
for library in src/core/libnixl.so src/infra/libnixl_build.so \
  src/utils/serdes/libserdes.so src/utils/common/libnixl_common.so \
  src/utils/stream/libstream.so src/utils/device/libnixl_device_allocator.so \
  src/utils/device/libnixl_device_allocator_cuda.so \
  src/plugins/gpunetio/libplugin_GPUNETIO.so; do
  ln -s "$(realpath "$BUILD_DIR/$library")" "$SNAPSHOT_DIR/"
done
BINARY="$OUT_DIR/remote_attention_consumer_stepmetric"
RESULT_DIR=/path/to/new-result-directory
```

This is a runtime link directory, not a frozen copy. The runner records binary
and library checksums and verifies them again after the campaign.
Keep the linked build outputs unchanged during a campaign.

`run_pair.sh SNAPSHOT_DIR BINARY RESULT_DIR` runs five alternating paired
campaigns (50 measured 20-step episodes/arm, 1,000 RTT samples and 950
post-start intervals/arm/run), then payload/ready/ACK negative injections.
Set `PAIRED_ONLY=1` to skip injections. Set `RDMA_DEVICE` for your machine. Defaults are GPUs 0/1, GID 0,
`eth0`, control port 6546, and OOB ports 6544/6545.
Override runtime paths and network settings through the script's environment.
It derives IPv4 from the OOB interface; if ambiguous, set `LOCAL_IP`.

```sh
export RDMA_DEVICE=your_rdma_device
export OOB_INTERFACE=your_oob_interface
export GID_INDEX=your_gid_index
export DOCA_LIB_DIR=/path/to/doca/lib
export CUDA_COMPAT_LIB=/path/to/cuda/compat/lib
PAIRED_ONLY=1 examples/gpunetio_data_ready/run_pair.sh \
  "$SNAPSHOT_DIR" "$BINARY" "$RESULT_DIR"
```

The snapshot must include `libnixl_device_allocator_cuda.so`. The runner saves
source/consumer logs and exit codes and invokes `analyze_pair_results.py`,
which checks all episode/reference gates, matched p50/p99, and raw GPU-step
interval arrays (computed on the host from existing device timestamps).

For the functional-only 32-bit epoch-wrap check, use `--epoch 4294967280
--steps 20 --arm paired --episodes 1 --warmups 0 --first-arm ordered` on both
endpoints; run consumer and source concurrently with the same run/coordination
directory. Verify both exit 0 and both episode validations PASS. This check is
not a performance sample.
