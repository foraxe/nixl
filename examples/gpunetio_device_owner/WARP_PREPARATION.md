<!-- SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Matched owner preparation comparison

Use the matching dependent checkout, public DOCA profile, build targets and
runtime library setup in [README.md](README.md). The new API requires all lanes
of CTA0/warp0, one owner token and a common 32-entry shared/global status array.
This experiment uses shared statuses for the collective arm.

The three explicit preparation modes use the same EXCLUSIVE executable:

| Mode | Preparation | Status storage |
| --- | --- | --- |
| 0 | Scalar | Thread-local |
| 1 | Scalar | Shared |
| 2 | Warp cooperative | Shared |

All timed arms set `OWNER_REPRO_BATCH_COMMIT=1` and
`OWNER_REPRO_PRECOMMIT_CHECK=0`. Compare mode2 versus mode1 to control status
storage location, and mode1 versus mode0 to expose that storage effect. Inspect
mode2 against both scalar controls before citing a gain. Each batch has 64
data PUTs plus a marker: 65 accepted WQEs with all completion/payload/guard/release
checks retained. Expected API commits are 65/9/3 at windows 1/8/32. WQE and commit
counts are caller-contract expectations, not measured NIC/MMIO counters.

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OWNER_REPRO_RDMA_DEVICE=your_rdma_device
export OWNER_REPRO_OOB_INTERFACE=your_oob_interface
export OWNER_REPRO_GID_INDEX=your_gid_index
examples/gpunetio_device_owner/run_preparation_comparison.sh \
  build /path/to/new-preparation-results
```

The runner requires a new result directory. It rotates/reverses arm order over
five blocks, using fresh endpoint processes and the same binary/runtime for
modes 0/1/2. Logical GPU0 sends and GPU1 receives. Optional
`OWNER_REPRO_SOURCE_CPU` and `OWNER_REPRO_RECEIVER_CPU` pin endpoint CPUs.
It saves both endpoint logs/exits, block order and executable hashes, then
checks hashes again and writes `summary.json`. Keep core/plugin/allocator/SDK
libraries fixed throughout; executable hashing does not pin those libraries.

Each run validates 3,600 batches including warmups, with 200 warmup and 1,000
measured samples per window. Five three-arm blocks validate 54,000 batches.
`compare_preparation.py` requires explicit 15-column sender records for all
three modes, correct mode/probe/commit metadata, paired endpoint validation,
and one complete permutation of arms in each order manifest. It reports the
manifest orders, their provenance, run-level p50/p99, medians and
paired deltas; it does not pool samples across runs or declare a performance
win automatically. Negative deltas mean the named candidate was faster.

For an existing campaign, `pairN.order` contains one arm name per line in
execution order (`mode0`, `mode1`, `mode2`, and optionally `parent`), once each.
The analyzer accepts any permutation, including forward/reverse campaigns; it
does not infer execution order from filenames. Record whether manifests were
captured during execution or reconstructed from an executed script in
`order_provenance.txt`. The output preserves that statement; absent provenance
is reported as unspecified, not as a verified order.

Optionally include a separately retained, immutable scalar parent executable
that emits the original 14-column grouped/probes-OFF CSV. Supply its previously
recorded SHA256; the runner checks it before and after the campaign:

```bash
examples/gpunetio_device_owner/run_preparation_comparison.sh \
  build /path/to/new-results-with-parent \
  /path/to/immutable-parent-owner-endpoint-exclusive "$PARENT_SHA256"
```

Only this explicit parent arm accepts 14 columns, with PREP_MODE unset. It is
never inferred to be mode0. The parent must work with the same runtime libraries.
Use mode0 versus parent to check example-wrapper overhead, and retain mode2
versus parent alongside the new scalar controls. The four-arm campaign validates
72,000 batches. Reanalyze using:

```bash
python3 examples/gpunetio_device_owner/compare_preparation.py RESULTS_DIR
python3 examples/gpunetio_device_owner/compare_preparation.py \
  RESULTS_WITH_PARENT --include-parent
```

Run probes-ON qualification separately by setting
`OWNER_REPRO_BATCH_COMMIT=1 OWNER_REPRO_PRECOMMIT_CHECK=1` and each explicit
`OWNER_REPRO_PREP_MODE=0`, `1`, or `2` on the existing two-endpoint invocation
from README. Both processes must exit successfully. Those qualification logs
must not enter this performance analyzer. Existing scalar wrong-grid/fault
probes do not replace qualification of the new collective API itself.

The source GPU timer includes preparation/commit/completion/lease release after
source initialization. Receiver validation gates acceptance but is not entirely
inside that source timer. Window32 retains a timed rejected 33rd-admission probe;
window8 is the cleaner primary comparison, and window1 measures collective setup
at one active payload lane. Diagnostic phase times motivate the experiment;
they do not establish a formal speedup or isolate parallel WQE stores from
group-reservation/validation amortization. Preserve all per-run results and
the strongest qualified scalar control. Collected coordination metadata and
machine-specific paths/network values should remain private.

CPU parser checks (no GPU required):

```bash
cd examples/gpunetio_device_owner
python3 -B -m unittest test_compare_preparation test_compare_commits
```
