# Thin GPU-callable GPUNetIO copy

Human-review checkpoint for foraxe/nixl, based on the GB200 handoff branch.
This does not change NIXL dispatch or claim an upstream-ready transport API.

## API

Prepare the existing registration/connection context once. In the application
kernel, use ThreadCopy from CTA0/thread0 or WarpCopy collectively from all
32 lanes of CTA0/warp0. WarpCopy submits one request per active lane.

```cpp
local_copy::WarpCopy copy(prepared);
copy.copy_async(dst_for_this_lane, src_for_this_lane, bytes);
auto value = independent_work();
if (copy.wait() == 0)
    consume(dst_for_this_lane, value);
```

The facade hides cumulative tickets, not errors. Every successful call submits
immediately. A later refused call does not abandon earlier work: wait drains
admitted requests before reporting the error. Buffers/context must remain live;
destruction is not an implicit wait. CTA-wide consumers synchronize after the
issuing thread/warp waits. See the actual compiled examples in example.cu.

## Change

The original wait applied a NIC acquire fence and CQ consumer-index update for
every CQE. The new paths inspect every CQE but publish retirement once per batch.
WarpCopy also inspects the bounded batch cooperatively across the warp. No
intermediate error, ownership, range, cookie or admission check is skipped.
Collective entry rendezvous prevents arrival skew from tripping the backend's
full-active-mask guard. The original wait remains as a measurement control.

Existing warp-batched WQE submission is reused, not claimed as a new algorithm.
Scope is pinned DOCA 3.5.0098, regular CQ/SQ128, one exclusive issuer and a
32-WQE software window. With MCST, scalar transfers use two credits and a warp
batch may have at most 31 active lanes. Backpressure remains explicit -EAGAIN.

## Evidence

The named GB200 case is 32 scattered 512-byte reads, a full-payload transform,
and 128 rounds inside one application-kernel launch. Legacy warp completion
takes about 45.4 us/round; optimized completion takes about 33.2 us/round,
approximately 27% lower latency. This is the whole primitive round, not only
CQ polling and not model/training throughput.

Two seeds, seven randomized blocks and three measured repeats cover 450 cases:
18,900 measured kernels plus 6,300 validated warmups. The named GPU and wall-time
comparison is favorable in every block of both seeds. Source/binary manifests
match across seeds. No samples are trimmed. Independent integer work is consumed
with the copied payload; serialized and async schedules are both tested.
SM and TMA are controls, not replacements for the NIC primitive.

Correctness includes full output/payload/guards; negative validation; reuse;
admission refusal followed by drain; mismatched/stale tickets; low-level partial
warp rejection; forced MCST; real intermediate invalid-rkey CQ errors; injected
absent-CQE timeout; and reset-failure quarantine. Existing low-level tests remain
in workdir/test_copy.py and test_quarantine.py. Test-only timeout injection
requires a LOCAL_COPY_TESTING library.

## Reproduce

This is a host-specific research package. It requires the pinned E GPUNETIO
plugin, privately extracted DOCA/RDMA userspace, CUDA 13.1 and the clean pinned
NCCL build used by the handoff. These private binary dependencies are not added
to this PR. Default roots are /workspace/gpunetio/kernel-copy-tools and
/workspace/gpunetio/nccl-core-workload. Use an exclusive GPU0 window.

From this directory, with those prerequisites:

```bash
PRIMITIVE_BUILD_SUFFIX=-prreview bash build.sh
PRIMITIVE_BUILD_SUFFIX=-prreview bash run.sh NEW_PILOT pilot 20261029 1 1 8
PRIMITIVE_BUILD_SUFFIX=-prreview bash run.sh NEW_A confirm 20261029 7 3 128
PRIMITIVE_BUILD_SUFFIX=-prreview bash run.sh NEW_B confirm 20261030 7 3 128
python3 analyze.py evidence/NEW_A/results.jsonl
python3 compare.py  # Validates the retained canonical confirmation tags.

PRIMITIVE_BUILD_SUFFIX=-prreview-testing LOCAL_COPY_TEST_FLAGS=-DLOCAL_COPY_TESTING bash build.sh
PRIMITIVE_BUILD_SUFFIX=-prreview-testing PRIMITIVE_TEST_TIMEOUT=1 \
  bash run.sh NEW_CONTRACT pilot 20261030 1 1 8
bash regress.sh NEW_REGRESSION
```

Run tags refuse overwrite. Retained raw data is gzip-compressed; analyze.py
accepts .jsonl.gz directly or finds it when the corresponding .jsonl is absent.
Hardware UUIDs, local binaries, generated summaries and unrelated GIN/TMA
experiments are excluded. Further optimization is a separate working branch;
this PR is a frozen human-review checkpoint, not permission to submit upstream.
