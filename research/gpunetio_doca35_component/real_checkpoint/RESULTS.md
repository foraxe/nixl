# Real-checkpoint vLLM provider comparison

On one H20 host, an owner and reader on separate GPUs used the vLLM Engram
owner/provider/runtime from commit `eaf7a18899a7a8da5ca89eccb8bcf7dffbbb964d`
in an existing vLLM source overlay. Both Engram layers came from an actual
188.83 GiB pinned-Host checkpoint. Every case used 384 or 6144 distinct,
deterministic IDs, three warmups and seven CUDA-event samples of the provider
lookup (GET plus scatter). Independently pread checkpoint rows matched before
and after timing; every timed request reported the expected completions. This
is a provider component test, not whole-model serving or general correctness.

Pre/new/post means 3.1 / 3.5 / 3.1 in separate processes. The 3.1 arm used an
existing row264 runtime binary SHA256
`9c158ae71ce75ca68cd45e3d4a02890d59f44cfa52b31f7e4dcb5954c737d46a`;
its exact source/build provenance has not been established. The 3.5 adapter is
`db51790fdca8832e53a15bcba8674d7c9a1e77d73086ab3e03cc17c810c6d00f`.
DOCA 3.1.0105 and 3.5.0098 were verified at runtime. The comparison changes
both implementation and SDK and is not an SDK-only comparison.

| Layer | Queries | 3.1 pre µs | 3.5 µs | 3.1 post µs | Change vs midpoint | Baseline drift | 5% gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 384 | 184.448 | 194.560 | 182.560 | +6.02% | 1.03% | FAIL |
| 1 | 6144 | 1620.992 | 1652.320 | 1621.216 | +1.93% | 0.01% | PASS |
| 14 | 384 | 178.368 | 189.600 | 177.984 | +6.41% | 0.22% | FAIL |
| 14 | 6144 | 1612.896 | 1655.520 | 1614.112 | +2.60% | 0.08% | PASS |

[`comparison/`](comparison/) retains every sample and validation result;
[`vllm_real_rows_benchmark.py`](vllm_real_rows_benchmark.py) is the measurement
source. A separately instrumented pre/new/post diagnostic is in
[`diagnostic/`](diagnostic/), with source
[`vllm_real_rows_diagnostic.py`](vllm_real_rows_diagnostic.py). Tracing changes
execution and its results do not replace the uninstrumented comparison.

An additional same-compiler 3.5 checked / unchecked / checked diagnostic is in
[`prepare_ablation/`](prepare_ablation/). The unchecked build deliberately
bypassed per-WQE safety checks **for this diagnostic only**. It reduced median
WQE-builder time by about 1.3–1.4 µs for 384 rows and 19 µs for 6144 rows,
but 384-row provider-event times did not improve in that diagnostic. This
isolates a cost of the checks, not a safe fix or a complete explanation of the
uninstrumented 384-row gap. The unsafe variant is not shipped as an API.

## Timing and reader-side follow-up

A later 384-row diagnostic alternated the ordinary measurement with a GPU delay
queued *before* the start event, giving the CPU time to enqueue the lookup. The
delay is outside the event interval, and each queued sample asserted that the
start event remained pending after CPU submission. In separate 3.1-pre / 3.5 /
3.1-post real-checkpoint processes, queued event increases were +6.23% and
+6.10% for layers 1 and 14; GET-kernel medians were about 8 µs higher in the
3.5 arm. Ordinary-event increases in that bracket were +1.76% and +3.88%.
The original +6% ordinary result is one noisy bracket, not a stable general
regression. Samples: [`underfeed/`](underfeed/) and source
[`vllm_underfeed_diagnostic.py`](vllm_underfeed_diagnostic.py).

To reduce owner/load variation, a single 3.1 owner then served six alternating
3.1/3.5 reader processes in order old1/new1/new2/old2/old3/new3. This mixed-SDK
setup first passed a small fixture but is not a general interoperability claim.
Queued-event changes in the three pairs were +2.80%, +2.19%, +4.15% for layer 1
and +2.81%, +2.76%, +4.28% for layer 14. GET-kernel medians were about 3–4 µs
higher for the 3.5 reader in all pairs. Ordinary-event pair changes were
+2.80%, -0.56%, +6.04% and +1.94%, +0.11%, +1.97%, respectively. Every process
used 15 samples per timing mode, and all data/completion checks and six normal
lease releases passed. Samples: [`same_owner/`](same_owner/). These results use
one host, one owner and two implementations/SDKs; they are not serving parity.

Two more diagnostics used a single 3.5 owner. The checked CUDA 12.8 / checked
CUDA 13 / checked CUDA 12.8 bracket had similar queued kernel medians; samples
are in [`compiler_bracket/`](compiler_bracket/). A checked / unsafe-unchecked /
checked comparison reduced queued GET-kernel time about 1.5–1.7 µs when checks
were bypassed, with little event-time change; samples are in
[`prepare_underfeed/`](prepare_underfeed/). The unchecked build is a test-only
ablation and is not a safe fix. The remaining device-path cost and full-model
performance remain unproven.

## Source-pinned 3.1 adapter comparison

The earlier 3.1 baseline binary's exact source was unknown. A 3.1 adapter was
rebuilt with CUDA 13 and DOCA 3.1.0105 headers from sources byte-identical in the
used paths to public NIXL research snapshot
[`b5322e46`](https://github.com/foraxe/nixl/tree/b5322e46f6e686ff16bdb3afb8a1a940a64e1163).
Its SHA256 was `7b0a39417f70c78e4af0b8a86be963b24d7f0c47194964ec4563758bd736907c`.
It linked against the retained 3.1 core/plugin binaries, whose exact source
provenance is not established; this is not a source rebuild of the full stack.

A same-owner, three-pair actual-checkpoint comparison between that build and
the earlier 3.1 binary found queued-event median differences between -0.19%
and +0.35%, with data and completion checks passing. This supports its use as
a bounded comparison baseline, but does not prove binary source identity.
Raw results are in [`legacy_source_check/`](legacy_source_check/).

Using the source-built 3.1 reader versus the 3.5 reader, with one fixed 3.1
owner and three balanced-order pairs, 384-row queued-event median changes were
+3.02%, +2.80%, +4.47% for layer 1 and +3.42%, +2.66%, +3.88% for layer 14.
All are below the 5% component threshold in this experiment. The 3.5 native
GET kernel remained about 3–4 µs slower, while ordinary event changes ranged
from -1.14% to +3.35%. All six reader processes passed byte/completion checks
and released their leases. Results are in [`source_baseline/`](source_baseline/).
This mixed-SDK, same-host result does not establish general inter-version
compatibility or full-model serving performance; the implementations, SDKs and
linked core/plugin stacks still differ.
