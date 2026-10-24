<!-- SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Bounded native-device owner example

This example uses two processes and two GPUs on one host. The sender stages
64 registered 4 KiB PUTs plus a marker, with outstanding windows of 1, 8 and
32. The receiver validates every payload byte, hole and guard on its GPU.
The control build changes only the DOCA sharing profile. Both builds retain
the same payloads, commits and completion checks.

The opt-in `native_device_owner=true` requires the native GPUNETIO Device API
and the public DOCA 3.5+ profile. A lease binds an increasing token to one
CUDA grid's CTA 0/thread 0. Keep the stream, views and registrations alive
until every accepted PUT completes; release the lease as the grid's last
NIXL API operation and synchronize before releasing host-side views.

## Build

Follow the repository's SDK/runtime setup, then configure the examples:

```bash
meson setup build -Denable_plugins=GPUNETIO -Dgpunetio_device_api=enabled \
  -Dgpunetio_device_api_profile=public -Dbuild_examples=true -Dbuild_tests=false
ninja -C build \
  examples/gpunetio_device_owner/owner-endpoint-exclusive \
  examples/gpunetio_device_owner/owner-endpoint-gpu-sharing-control \
  src/utils/device/libnixl_device_allocator_cuda.so
export NIXL_PLUGIN_DIR="$PWD/build/src/plugins/gpunetio"
export LD_LIBRARY_PATH="$PWD/build/src/core:$PWD/build/src/infra:$PWD/build/src/utils/common:$PWD/build/src/utils/serdes:$PWD/build/src/utils/stream:$PWD/build/src/utils/device${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Build the CUDA allocator sibling as well: it is loaded lazily and may not
appear in ordinary `ldd` output. Use the same configured SDK and CUDA runtime
for both profiles. The measured setup used CUDA 13.0.48, SM90 and DOCA 3.5.0098.
Core, plugin, allocator and application headers must come from this same
dependent checkout; mixing an installed core from a different plugin ABI fails.

## Run

Set the fabric configuration for your machine and use two available GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OWNER_REPRO_RDMA_DEVICE=your_rdma_device
export OWNER_REPRO_OOB_INTERFACE=your_oob_interface
export OWNER_REPRO_GID_INDEX=your_gid_index
for pair in 0 1 2 3 4; do
  profiles=(exclusive gpu-sharing-control)
  if ((pair % 2)); then profiles=(gpu-sharing-control exclusive); fi
  for profile in "${profiles[@]}"; do
    coord=$(mktemp -d)
    exe="$PWD/build/examples/gpunetio_device_owner/owner-endpoint-$profile"
    "$exe" receiver "$coord" run 1 32110 > "$coord/receiver.log" 2>&1 & receiver=$!
    "$exe" sender "$coord" run 0 32111 > "$coord/sender.log" 2>&1 & sender=$!
    wait "$sender" && wait "$receiver" || exit 1
    echo "$coord"
  done
done
```

Each run uses 200 warmup and 1,000 measured GPU-time samples per window.
Report all five pairs. Fixed-commit mode commits every PUT; set
`OWNER_REPRO_BATCH_COMMIT=1` for a separate correctness qualification that
stages multiple PUTs before one commit and verifies precommit IN_PROG status.
`OWNER_REPRO_PRECOMMIT_CHECK` defaults to `1`; leave it enabled for that
qualification. For a matched performance comparison, set it to `0` in both
fixed and grouped arms, so only commit grouping changes. Real postcommit CQ,
payload, guard and lease-release checks remain enabled.
The window-32 timing includes an intentionally rejected 33rd admission probe;
window 8 is the clean bounded-window performance comparison.

Run five alternating EXCLUSIVE-profile pairs with the same executable and
runtime configuration:

```bash
examples/gpunetio_device_owner/run_commit_comparison.sh build /path/to/new-results
```

The runner requires the fabric environment and matching core/plugin library
paths above. It validates both endpoint exit codes and invokes
`compare_commits.py`, which rejects mixed qualification/performance modes,
incomplete runs and unexpected status/count fields before producing per-window
paired p50/p99. There are 65 accepted WQEs per batch in either arm. At window
8, expected API commits are 65 versus 9, including the final marker; the CSV
reports this expectation rather than claiming measured MMIO transactions.

For an isolated stale-rkey CQE test, use `fault-receiver` and `fault-sender`
with a fresh coordination directory and run ID. These roles revoke only their
own test registration. They require a real completion error, rejection of
later posts and refusal of successful lease release; a timeout does not pass.
