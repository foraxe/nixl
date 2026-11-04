# H20 synthetic component comparison

One same-host H20 run, DOCA 3.1.0105 control versus DOCA 3.5.0098 candidate,
seven GPU-event samples per arm, with the predeclared gates of candidate median
no more than 5% above the midpoint of the control medians and control drift no
more than 5%. The control is a separately built legacy research snapshot, so
this compares implementations and SDKs together. All timed cases checked
output bytes and completion counts. These are component observations, not
end-to-end inference or a claim of a speedup.

| Component | Control pre (µs) | 3.5 (µs) | Control post (µs) | Candidate vs midpoint | Control drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grouped graph, 16 QPs | 318.656 | 322.688 | 313.568 | +2.08% | 1.61% |
| Grouped graph, 1 QP | 552.320 | 551.136 | 556.576 | −0.60% | 0.77% |
| vLLM Context, 384 rows | 152.768 | 155.840 | 153.184 | +1.87% | 0.27% |
| vLLM Context, 6144 rows | 2137.632 | 2157.952 | 2124.704 | +1.26% | 0.61% |

Grouped cases use `make_ids.py`: 6144 synthetic queries, 1561 owned,
262144-row table, 264 bytes per row, window 512, depth 2, parallel polling,
no deduplication, CUDA graph, one GPU. Each arm ran grouped 16 QPs, grouped 1 QP, then the vLLM Context cases.
The Context cases used a synthetic 8192-row table, 3 warmups, 7 samples, one
QP and no graph. They use the same vLLM runtime wrapper as the
recorded experiment, but do not execute a whole model. The generated query
content is synthetic; no model weights or private request traces are included.

The sanitized per-sample measurements and validation records are in
[`component_samples.json`](component_samples.json). Run
`python3 summarize_results.py` from this directory to recompute the medians,
pre/post drift and 5% gate directly from those samples. The summary script
checks recorded adapter hashes, sample counts and the available validation
fields; it cannot independently rerun the original hardware checks.

The 3.1 control adapter SHA256 is
`ad6086bda139657787cf382e1e6209b6c4134a2e39428877f207bf5b879dea5a`;
the 3.5 adapter is
`db51790fdca8832e53a15bcba8674d7c9a1e77d73086ab3e03cc17c810c6d00f`;
the 3.5 plugin is
`1a09d29337b7cd7e04db88e47dc74787eef9691800232de7cbbc4c65a2cc4d64`.
The 3.5 candidate source is NIXL commit
`0630a0859751440316f3ee7143b44d4b09d39f88`, based on main
`c656976fcd6d3b10e34ecae01bfbc399b275f574`. The tested 3.5 batch completion path uses the SDK internal
non-advancing CQ helper. The measured H20 did not require MCST; the MCST-required
path has not been qualified. The result does not establish peer-restart
recovery, independent checkpoint quality, or full-serving performance.
