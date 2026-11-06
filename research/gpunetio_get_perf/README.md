# Backend-owned native GET QP pool

This change replaces many prepared NIXL engines with one backend-owned QP
pool. The one-QP default, device-view ABI and GPU get/readSession signatures
are unchanged. It builds on the separate RO checkpoint(PR #6), which remains
frozen. This work targets the fork's GET branch for human review only.

## API and ownership

Set `native_device_api=true` and `native_qp_count=N`(1..32, default1) on both
peers. Register each buffer once. Prepare ordinary NIXL view pairs once with
`hint.customParam="native_channel=N"`; omitted/empty selects0. Kernel callers
keep using the same get/status/readSession calls. There are no new per-copy
CPU descriptors, posting requests, futures or connection objects.

Each channel has its own phase, generation, SQ and CQ, backed by the engine's
shared PD/registrations. Different channel context pointers reject mixed
local/remote/scratch view pairs. Preparing another view of a bound channel
never resets its generations or credits. Failed/busy channel views retain
pins until safe teardown. Native mode still binds one peer per engine.

Channels are independently ordered EVEN with RO disabled. Synchronize before
reusing overlapping destinations/scratch across channels. RO remains separate,
opt-in and default false. Notifications/control registrations remain strict.
Legacy WRITE `data_qp_count` remains capped at4, with its old stream constraints.
Native pools do not allocate the unused legacy CPU-mapped WRITE counter.

The existing versioned QPN exchange is reused. Pools larger than one retain
an unused dedicated QP for wire compatibility; the additional QP set provides
N active READ channels. The one-QP wire path is unchanged. Both peers must
configure the same count; mismatches are rejected, not silently negotiated.
No route restart/recovery or multi-peer native support is added.

## Performance against batched GDAKI

GB200 same-NIC loopback, HOST_NUMA source -> HBM, scattered264B rows,
identical vector checksum consumer. Both use relaxed VRAM data registration;
NIXL explicitly enables the separate `native_relaxed_ordering=true` option.
GDAKI uses GPU doorbells, CTA sharing,
queue2048, aggregate-request tiles512, and the same active QP count. No
availability/error checks were removed from NIXL. Graph median wall times:

| Case | Reads | NIXL us | Batched GDAKI us |
| --- | ---: | ---: | ---: |
| One backend/one QP | 384 | 130.37 | 125.11 |
| One backend/32QPs/32CTAs | 384 | 50.32 | 44.00 |
| One backend/32QPs/fourCTAs | 384 | 50.11 | 42.34 |
| One backend/one QP | 4096 | 1121.11 | 1024.05 |
| One backend/32QPs/32CTAs | 4096 | 163.12 | 158.88 |
| One backend/32QPs/fourCTAs | 4096 | 181.81 | 158.08 |

The pool makes32QP throughput practical with ONE agent/backend/registration
per rank. At4096rows,32CTAs provide6.87x lower wall latency than one QP;
fourCTAs provide6.17x. The32CTA transfer-ready interval is96.72us versus
GDAKI92.16us; full wall is2.7% slower than GDAKI. FourCTAs use fewer blocks
but are about15% slower end-to-end than GDAKI's32CTAs. This is a resource /
latency tradeoff, not a NIXL-superiority claim or measured compute-interference
win. Eager measurements and all samples are retained in `pool_evidence/`.

Matched32-independent-engine/four-CTA NIXL is184.31us versus181.81us for the
pool. Do NOT advertise that1.4% difference as a win: its GDAKI control also
varied about5.4%. The supported conclusion is similar throughput with32->1
NIXL agents/backends/registrations and one retirement event instead of32.

Three fresh launches per case,20warmups and100samples per mode/size; four
rotating eager/graph x GDAKI/NIXL modes, two sizes, four configurations:
9600timed pipelines plus1920warmups. Every byte/checksum/guard and accepted
READ count is validated outside timing. Both ranks close normally in all12
runs. GPU events separately record ID/GET/ready and consume/retirement time.
No profiler or trimmed samples enter these results.

## Correctness

- CPU option tests with C++20 -Wall -Wextra -Werror: counts1..32, all channel
  indices/default0, malformed/signed/out-of-range/binary options. Independent
  C++17 parser review passed as well.
- Default1 and pool4 x strict/RO x pinned Host/VRAM; extra pool32/RO/Host.
  Distinct device lane/QP addresses, mixed-channel rejection, immediate scalar
  and batch GPU consumers,130x16 SQ/CQ wrap per channel, repeated poisoned
  buffers, metadata reload and generation-preserving re-preparation.
- Mismatched1/4,4/1 and2/4 peers reject during MD/OOB setup. Disposable
  processes retain setup resources until exit; no recovery claim is made.
- A real invalid-rkey READ on the last pool channel returns a CQ error, then
  refuses reuse; a healthy channel continues to GET/consume correctly.
  Releasing idle and failed views preserves the failed view/MR lease. Memory
  checking of that retained view reports zero errors. Failed resources are
  retained until disposable-process exit, not reclaimed speculatively.
- The frozen pre-pool one-QP test binary runs against this plugin, validating
  the unchanged default/device ABI path. Feature-disabled host translation
  units compile with Werror; this is not a full feature-disabled GPU run.
- Release plugin build, shell/Python/whitespace/SPDX checks and an independent
  code/evidence review. CPU partition verification checks47840 representative
  indexing/wave-count configurations; it is not runtime CUDA qualification
  of all arbitrary shapes.

Hardware requiring MCST remains unvalidated. Real scalar-channel CQ failure
is tested; real NIC-error/recovery of the cooperative multi-channel benchmark
helper is not separately qualified. Completion fences/CQE checking in the
public device API are unchanged. GPU performance is not model/serving E2E,
cross-host/cable bandwidth, sustained compute interference, or proof of an
absolute physical limit. The earlier43.8%/52.3% RO headline was against
per-request-posting GDAKI, not this stronger batched control.

## Reproduction and provenance

Source base fbe8ccd3(GET bdf954038aa76cec4b63d2fb556debc0a62ea696 plus PR #6).
CUDA13.1 sm100; NCCL2.32.3 commit3e46c089f4ae4732722a419eb6096aead7e47a30,
bundled GPUNetIO4.1.0; NIXL DOCA3.5.0098. Different SDKs mean this is not an
isolated framework-overhead comparison. Physical GPUs0/1, NUMA0, mlx5_0/GID3,
LSA1/P2P-disabled, reported400Gb/s, same-NIC loopback, not external network.

Build the plugin release with GPUNETIO/device API enabled, CUDA arch100.
The pinned local build is `/workspace/gpunetio/kernel-copy-tools/get-pool/build`;
it links the unchanged framework from the `gdaki-compare/install` prefix.
`build_adapter.sh` uses this source's device headers. Build the harness using
`build.sh`, GIN_SHARING=CTA, GIN_AGGREGATE=512, VECTOR_CONSUMER=1 and the desired
BENCH_QPS. BACKEND_POOL=1(default) creates one pooled backend;0 is the matched
independent-engine control. The benchmark facade's read call is unchanged.

```bash
# Hardware checks with the frozen local binaries; choose fresh tags.
bash validate_pool.sh UNIQUE_CORRECTNESS_TAG
bash run_pool_perf.sh UNIQUE_PERF_PREFIX
python3 analyze_pool.py results UNIQUE_PERF_PREFIX

# Recalculate committed measurements without GPU/private logs.
python3 analyze_pool.py pool_evidence
```

Data files retain exact samples, build/source identities, selected markers
and configuration. Selected records are not full logs; their full-log hashes
are retained. Full local logs are ignored under `results/`. The inherited
`evidence/` directory is the historical PR #6 checkpoint; its `analyze_gate.py`
source-identity audit must be run from fbe8ccd3, not this modified harness.
Do not rebuild frozen binaries used by an active experiment.
