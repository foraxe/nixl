# GB200 handoff: kernel-initiated tensor copy and native GIN comparison

## User intent and next task

Make registered Host DRAM -> owner HBM movement callable inside an application
kernel, close to Copy Engine / TGET ergonomics: prepare tensors/resources once,
copy_async, independent work, wait, consume. No per-copy CPU descriptor/request
workflow and no unnecessary API layers. Prefer reusing an existing native primitive
if it satisfies the interface and performance goal.

On GB200, first qualify and measure native NCCL GIN/GDAKI against direct Copy,
modern CUDA batched/graph copy, and strong mapped-host SM gather. Keep allocations,
indices, memory placement and consumer-ready boundary equivalent. Initialization
and registration are cold costs, not per-operation costs. Do not assume GIN is
faster from source inspection or compare it against an intentionally weak CE arm.

## Package map

- workdir/: complete standalone code, reports, journals, small raw evidence.
- workdir/README.md: scalar/warp API and implementation boundaries.
- workdir/engram_scenario/README.md: latest large-workload experiment and exact replay.
- workdir/challenge/GIN_COMPARISON.md: source-level GIN comparison, prerequisites,
  window allocation requirements, exact call sites and DeepInfer scenario evidence.
- dependencies/nixl-e/: exact clean E source, including low-level QP/MR classes.
- dependencies/nccl/: exact clean NCCL source, including bundled GPUNetIO 4.1.0.
- context/original-architecture/: original six input documents, historical context.
  Their host-API-first plan was superseded by the user's kernel-first instruction.
- SOURCE_PINS.json: dependency refs and URLs. PACKAGE_SHA256SUMS: archive file hashes.

## Current evidence, not a promise for GB200

Latest H20 test: 512B BF16 rows, 512 tokens x24 entries =12,288 reads/6MiB per
layer, two layers. 256MiB Host table/layer, GPU indices, real dense BF16 cuBLAS
projection [512,6144]x[6144,8192], 0/1/4 calls/layer. SERIAL versus LOOKAHEAD DAG;
full payload/guards and actual projection outputs validated after every request.
Two complete matrices:840 case-blocks,5,040 requests; about37.16M direct NIC READs.

Best tested H20 results, two layers:
- Copy+light consumer: CUDA-copy graph32.956ms, NIC8 contexts/QPs2.462ms,
  SM78-CTA gather0.286ms.
- Four projections/layer with lookahead: CE34.961ms, NIC4.921ms, SM3.311ms.
- NIC8/G4 lookahead regressed at8 CUDA connections: one actor queued until GEMM
  ended. Same binary/seed/placement,32 connections removed that tail; SERIAL
  stayed5.642ms,LOOKAHEAD4.901ms instead of5.954ms. Actual node traces support
  this scheduling attribution; remaining compute interference was not isolated.

Thus NIC beats the tested old-driver CE paths, but does NOT win against strong
SM gather. Do not discard those negative controls. Modern memcpy-batch returned
error36 on that old driver. Native GIN was NOT measured: DMA-BUF support0 and
legacy nv_peer_mem without nvidia_peermem; the pinned NCCL explicitly rejects it.
These are single-process H2D subscenarios, not full DeepInfer/multi-rank results.

## GB200-specific port gates

1. Inspect real GPU/NIC identity, free ownership, driver/toolkit/DOCA/NCCL versions,
   actual loaded libraries, topology and NUMA before changing anything. Retain
   other workloads; never reset GPUs or change host modules/drivers casually.
2. GB200 is ARM64/SM100. Do NOT run H20 x86 binaries. Current scripts hardcode
   sm_90, x86_64-linux-gnu, an Abseil source version, GPU6/7, NIC3/GID3 and
   CPU80-83/NUMA1. Rebuild and adapt these with evidence. Current engram scenario
   hardcodes NIC mlx5_bond_3 in lc_open; make it an explicit validated option.
3. Grace Host DRAM may be reached over NVLink-C2C. This changes mapped-host/SM
   performance, CE baselines, cache/coherency and NIC reachability. Identify the
   allocation's actual backing and route; do not relabel it as PCIe H20 H2D.
4. The direct runtime links E's exported internal C++ qp/mr classes. Its headers,
   plugin library and SDK must match. Build dependencies/nixl-e on the target;
   never mix its DOCA3.5-exported QP layout with newer SDK device headers.
5. Root build.sh expects E_ROOT/source and E_ROOT/install; reconstruct that build
   layout or adapt it. Inspect meson_options.txt; enable_plugins=GPUNETIO exists.
   External Meson wrap/development dependencies are not vendored by this package.
   Adapt libdir and include paths to ARM64; locate actual CUDA/cuBLAS/DOCA prefixes.
6. Native GIN public windows require retainable cuMem/VMM handles. Use supported
   HOST_NUMA VMM source and NCCL/VMM HBM destination, and run all controls on the
   SAME allocations. Ordinary cudaMallocHost/cudaMalloc tensors are not a proven
   supported drop-in for those windows. Qualify direct Copy on these allocations
   too; do not reuse ordinary-pinned H20 evidence blindly.
7. Pinned GIN2.32.3 bundles GPUNetIO4.1.0 and real AggregateRequests DB suppression.
   Non-final requests may aggregate; a final non-aggregate GET must actually submit.
   Non-MCST flushAsync only snapshots reservations: do not assume it rings the DB.
   There is no public WarpGet flag in this specific source pin; DeepEP's README
   requires a build providing that extension, so discover the actual package/API.
8. Start with one rank/self peer and one exclusive context, then reasonable
   context/sharing/handler configurations. Confirm native GDAKI selection and
   actual handler; CPU-proxy results cannot stand in for native GET performance.
9. The new driver may support CUDA batch APIs: implement/test that arm rather
   than replaying the old unavailable baseline. engram_scenario intentionally
   aborts if batch support appears until a matched arm is added. CUDA13 signatures
   differ from the12.8 calls in these sources; verify installed headers.
10. Run small correctness pilot before large matrices. Source rows encode full
    identity; wrong+31-row negative must be rejected. Keep first-fault preservation,
    every-CQE checking, MCST, admission, request ownership and safe/quarantined
    teardown. Do not obtain speed by deleting required visibility or error work.

## Minimal execution order

- Read the three README/comparison docs above; inspect installed SDK/API and port
  architecture, identity and build paths in an isolated branch/work directory.
- Build the pinned low-level dependency and direct runtime; run smoke/example,
  test_copy.py, and the disposable reset-failure test as appropriate to the platform.
- Run challenge/gin_preflight.py (read-only registration check, NOT full qualification).
- Build a native GIN GET baseline on supported common allocations. Record selected
  backend and library hashes. Preserve user-visible consumer-ready semantics.
- Port/run the small engram scenario first, then512B x12,288 x2 layers, real projection,
  lookahead and strong CE/SM controls. Use fresh processes for8/32 connection controls.
- Save hashes, actual commands/config, raw small JSONL, summaries and one node trace.
  Separate correctness, timing, integration and production qualification conclusions.

The full matrix's prebuilt graphs reached about24GiB GPU /16GiB Host RSS in one
H20 observation. Filters reduce cold graph footprint. All timing excludes cold
setup, reference computation and validation. Static indices are deliberately
known to CE before graph construction; this favors CE and is not a dynamic-input
end-to-end serving measurement.

## Recovery and out of scope

Large complete artifacts remain on the H20 source environment, indexed in the
reports: kernel-local-copy-20261023, kernel-local-copy-challenge-20261023,
kernel-local-copy-engram-20261024 under /tmp. No binaries/SDK/PyTorch/large traces
are in this tar. Recover remotely only if needed; regenerate seeded data.

Do not modify unrelated E WRITE work, NIXL #2222/PD branches or other sessions.
No PR/push/public replies, messaging or deployment has been authorized by this
handoff. No GB200 execution or speedup is claimed by this package.

## Suggested skills

- nixl-server-workflow: use the currently verified target access lane; H20 kh20
  commands in old reports are not GB200 routing instructions.
- root-cause-first-debugging: isolate build/runtime/performance causes.
- phd-level-systems-research: preregister discriminating controls and claim bounds.
- ponytail: minimal port, no new wrapper/framework without need.
- professor-feedback: independent challenge after a concrete result exists.

Keep one task journal and source-backed results. Local disk is constrained: source,
reports, compact evidence and recovery indexes locally; complete large artifacts
remotely. Never delete unique source/evidence to make room.
