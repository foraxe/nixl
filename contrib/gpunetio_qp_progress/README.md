# Optional independent-QP component replay

This is a separately retained tool snapshot, **not a change in #2222** and not
registered in Meson/GTest CI. The source matches the public-API fixture used for
the October 20 data-only qualification. It needs three distinct GPUs on one host:
one source and two targets. It checks GPU payloads and changing epochs outside
the timed post-to-API-completion window.

Build against the candidate's release installation (DOCA >=3.5 and compatible
CUDA/driver), with GoogleTest available for manual assertions:

```bash
: "${PREFIX:?candidate release installation}"
: "${NIXL_LIBDIR:?directory containing matching libnixl.so and libnixl_build.so}"
nvcc -std=c++20 -O3 -arch=sm_90 --cudart shared \
  -I"$PREFIX/include" qp_progress_replay.cu -L"$NIXL_LIBDIR" \
  -lnixl -lnixl_build -lgtest -Xcompiler=-pthread -o qp_progress_replay
export QP_TEST="$PWD/qp_progress_replay"
# Include the same NIXL, DOCA, CUDA and RDMA provider libraries for both arms.
export LD_LIBRARY_PATH="$NIXL_LIBDIR:${LD_LIBRARY_PATH:-}"
```

Build main and candidate plugins from the **same plugin API version**, with the
same SDK, compiler and release flags. Keep the core, executable and loaded
dependencies unchanged while switching only the plugin. These measurements
isolate the backend delta; they are not a full-stack main-versus-candidate A/B.

Set these environment variables to your own available resources; no site values
are embedded in the tool:

- `MAIN_PLUGIN`, `CANDIDATE_PLUGIN`: plugin directories.
- `SOURCE_GPU`: one free GPU; `TARGET_GPUS`: two other comma-separated GPUs.
- `SOURCE_CPU`, `TARGET_CPU`: distinct physical CPU cores.
- `NIXL_QP_PROGRESS_NETWORK_DEVICE`, `NIXL_QP_PROGRESS_OOB_INTERFACE`,
  `NIXL_QP_PROGRESS_GID_INDEX`: optional explicit backend configuration.
- `RESULTS_DIR`: optional fresh output directory.

Run all five alternating pairs for each shape:

```bash
bash run_five_pairs.sh
```

The runner uses LAZY loading, one data QP per peer, 1200 warmup rounds, 10000
measured 4 KiB rounds, and 1000 measured 2 MiB/mixed rounds. It preserves both
worker logs and per-run JSON. Both endpoints must pass, not skip. Do not run
profilers or competing workloads during the comparison. Match GPU/NIC locality
and CPU affinity across arms. This script does not start MPS or change NICs.

Report p50 and p99 for every pair, medians of run-level metrics, median paired
percentage changes, and win counts. Include the large-transfer metrics; an
isolation win is not automatically an overall throughput win. Control JSON
contains 8 epoch bytes beyond the requested 4 KiB/2 MiB payload size.

For data-only correctness, use `OutstandingDescriptorsAndAttachedStreamOrder`,
`AttachedReadOrdersFollowingConsumer`,
`RemoteDeregisterReturnsBackendError` (target role `target-fault`), and
`CpuFatalLatchRejectsQueuedTransfer` with the same paired environment.
`CpuFatalLatchRejectsQueuedNotification` tests the separate notification
publication-kernel path and is **not a data-only/main gate**. It is retained in
the identical source, not claimed as passed for the data-only PR.

No serving, cross-node or async-notification benefit is established by these
component timings. This snapshot does not modify the backend under test.
