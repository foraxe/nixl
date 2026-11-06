# Native GET relaxed-ordering qualification on GB200

The opt-in `native_relaxed_ordering=true` meets the >=10% lower-latency target
against tuned NCCL GDAKI on this local NIC-mediated consumer pipeline.
It is not enabled by default and is not a model/serving throughput claim.

## Measured result

Median graph-replay pipeline latency, 2026-11-06:

| Scattered reads | Tuned GDAKI | NIXL relaxed GET | NIXL lower latency |
| --- | ---: | ---: | ---: |
| 384 x 264B | 0.589 ms | 0.331 ms | 43.8% |
| 4096 x 264B | 6.814 ms | 3.253 ms | 52.3% |

Eager results agree: 0.591/0.332 ms and 6.824/3.256 ms, respectively.
Every one of three brackets cleared 10% on GPU-event and synchronized wall
time, for both sizes and both schedules. NIXL won all 300 measured pairs in
each primary comparison. No samples were removed.

The comparator is GDAKI with a 2048-entry queue and
`NCCL_GIN_RESOURCE_SHARING_CTA`, not its slower stock defaults. Both paths
use one 128-thread CTA and one requested context/data QP. NIXL uses 512-row
windows and checks every READ CQE. Neither backend's completion policy was
changed for this optimization.

Three fresh, unmodified GET-branch controls with the exact same current
adapter and harness gave NIXL graph medians 0.785/6.590 ms. The patched
plugin with the new option disabled gave 0.788/6.488 ms; do not substitute
older historical 0.814/8.4-ms runs for these matched controls. The final
gain claim compares NIXL and GDAKI in the same relaxed-arm process.

## Fix and contract

GDAKI normally registers data MRs with PCIe relaxed ordering. NIXL's native
GET registrations were strict. An independent GDAKI control with relaxed
ordering disabled reproduced the slow handoff to the checksum consumer.
The NIXL change adds a setup-only option for VRAM data MRs, not another
per-copy API, request handle or CPU posting step.

```cpp
nixl_b_params_t params{{"native_device_api", "true"},
                       {"native_relaxed_ordering", "true"}};
// Add the usual device/OOB parameters; the GPU GET API is unchanged.
```

Default remains strict. The option requires native mode. Host source MRs
and internal notification/control MRs remain strict. DMA-BUF and PeerDirect
registration retry strict flags if the optional relaxed attempt fails.
Registration logs report the accepted ordering.

Only opt in for non-overlapping in-flight destination ranges. Wait/acquire
before consuming or reusing them. Relaxed ordering changes write-after-write
ordering but not completion visibility; see the
[rdma-core contract](https://github.com/linux-rdma/rdma-core/blob/master/libibverbs/man/ibv_reg_mr.3).
CQ error checks, acquire fences, MCST handling and retirement are unchanged.

## Evidence and bounds

- GB200 ARM64, physical GPU0 owner / GPU1 reader, both NUMA0; mlx5_0/GID3,
  reported 400 Gb/s, same-NIC RDMA loopback. LSA size 1 and P2P disabled.
  Runtime asserts GDAKI and GPU doorbell handler 2; no host-proxy GIN fallback.
- HOST_NUMA VMM source with 8192 packed 264B rows, HBM destination.
  GPU-generated scattered IDs change each sample. Timed pipeline includes ID
  generation, GET, checksum consumer and API retirement where required.
  Every payload byte, checksum and destination guard is checked outside timing.
  Distinct rank source patterns reject accidental local reads.
- CUDA 13.1, sm_100, NVML driver 580.105.08. NCCL 2.32.3 commit
  `3e46c089f4ae4732722a419eb6096aead7e47a30`, bundled GPUNetIO 4.1.0.
  NIXL GET base `bdf954038aa76cec4b63d2fb556debc0a62ea696`, DOCA 3.5.0098.
  This is a working-stack comparison, not an isolated SDK comparison.
- Three pre/relaxed/post brackets, 20 warmups and 100 samples per mode/size.
  7200 measured pipelines and 1440 warmups passed; all 18 ranks closed cleanly.
  Strict pre/post controls drifted less than 5% in every bracket. Three extra
  unmodified-base launches add 2400 measured pipelines and 480 warmups.
- CPU policy test covers strict default, relaxed flags, strict retry and total
  failure using the callback shared by both registration paths. GPU tests pass
  strict/relaxed x pinned Host/VRAM, scalar/batch, immediate in-kernel consumers,
  poison-and-reuse, 130 x 16 operations across SQ/CQ wrap, guards and invalid
  options/descriptors. Hardware DMA-BUF strict retry is not separately proven;
  that retry path has callback-policy coverage.
- Warm repeated component data, not cold bulk bandwidth, a physical-limit
  claim, full model serving, or sustained-compute interference. Hardware
  requiring MCST remains unvalidated; that inherited path is unchanged.

Raw numeric samples, source/binary identities and selected validation records
are in `evidence/`. Selected records are explicitly not full machine logs;
their full-log hashes are retained. Full logs and diagnostic experiments stay
in the ignored local `results/` directory. `evidence/correctness.json` retains
correctness markers. `evidence/summary.json` is the auditable gate output.

Terminal-CQE gating and a different acquire-fence experiment did not produce
a gain; neither is in the library patch. The external profiler stalled its
diagnostic capture, which was excluded. No profiler measurements enter the
primary performance result.

## Reproduce

On the original GB200 workspace, private dependencies and builds are under
`/workspace/gpunetio/kernel-copy-tools`. The tested plugin is in
`get-perf/build/src/plugins/gpunetio`; the unchanged core/framework is from
the pinned `gdaki-compare/install` prefix. The adapter starts from reproduction
commit `5df059592a98d75c711ed2c1a461ec965b8dc20d`; its only active behavioral
addition is forwarding the new backend option at setup.

Build the plugin from this source with Meson release, GPUNETIO enabled,
`gpunetio_device_api=enabled`, and CUDA arch 100; do not rebuild the frozen
benchmark/plugin while a run is active. `build.sh` builds a fresh GDAKI-only
harness; `build_adapter.sh` builds the adapter against this source's headers.
Set `NCCL_ROOT`, `CUDA_ARCH=sm_100`, `GIN_SHARING=CTA` and a fresh `BENCH_BINARY`
for the harness, and `OUTPUT` for the adapter. The local scripts document
the pinned dependency locations rather than modifying system packages.

```bash
# In this directory after building the frozen gate binaries:
bash validate_local.sh UNIQUE_CORRECTNESS_TAG
bash run_gate.sh UNIQUE_GATE_PREFIX
python3 analyze_gate.py results UNIQUE_GATE_PREFIX

# Audit the committed measurements without a GPU or external logs:
python3 analyze_gate.py evidence ro-gate-v1
```

Do not mix other queue depths, GPU-sharing modes, SDK builds, topologies or
missing/error runs into the reported comparison. The fork PR is a human-review
checkpoint, not authority to update an upstream PR.
