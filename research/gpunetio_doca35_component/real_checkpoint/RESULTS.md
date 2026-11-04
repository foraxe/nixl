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
