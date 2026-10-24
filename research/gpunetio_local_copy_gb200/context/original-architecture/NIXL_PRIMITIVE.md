# Optional local HBM ↔ host DRAM copy through NIXL

Architecture investigation only. No code changes, builds, benchmarks, or PR review were performed.

Repository snapshot: ai-dynamo/nixl main, commit `d64231659c8194760b60d163b514a3eba1ac22ee`, committed 2026-10-22 16:02:42 UTC and inspected 2026-10-22. All repository links below are pinned to this snapshot.

## Executive recommendation

Treat local NIC-assisted copies as an optional execution lane for NIXL's existing one-sided READ/WRITE request abstraction. Keep the application-visible operation generic: registered source and destination ranges, direction, asynchronous completion, and an explicit backend choice. Do not require an Engram-specific gather operation, NCCL communicator, or a new transfer framework.

Start with **A: host-submitted NIXL local GPUNETIO**, not **B: a new GPU-callable NIXL backend**. A can evaluate the intended NIC DMA lane and its issue/completion resource footprint without also redesigning the device API. It still has host preparation, a host `postXferReq`, CUDA launches, and host polling. B is justified separately when the producer must issue directly from an already-running GPU kernel or produce descriptors on the GPU. Neither lane should replace the CE default until equivalent end-to-end completion semantics and workload benefit are established.

This is a proposal for a thinner NIXL integration, not a claim that same-NIC local RDMA is new hardware functionality or that GPUNetIO and NCCL GIN are inherently different physical transports. In particular, comparison against an already-batched GIN path must isolate library integration, issue resource use, and completion behavior rather than claim a novel advantage merely from batching.

## 1. What current NIXL actually provides

### Existing generic local transfer seam

The host API already accepts a self-agent as the target, and already separates registration, descriptor preparation, request preparation, posting, polling, and release. Its documentation explicitly warns that repeated `createXferReq` repeats validation/preprocessing, while `prepXferDlist` plus `makeXferReq` can reuse common descriptors. This is the appropriate abstraction to preserve, not a model-specific replacement. [Host transfer API](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/cpp/nixl.h#L238-L389)

Core calls `connect(self)` when a backend advertises local support. Registering memory with a local-capable backend also loads corresponding self-target metadata. The existing UCX implementation provides `connect(self)` and `loadLocalMD`, then uses its normal read/write and flush machinery. [Core local setup](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/core/nixl_agent.cpp#L393-L419), [registration](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/core/nixl_agent.cpp#L466-L514), [UCX self-connect](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/ucx/ucx_backend.cpp#L169-L181), [local metadata](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/ucx/ucx_backend.cpp#L305-L313)

### GPUNETIO is not currently a local backend

The current GPUNETIO plugin supports DRAM and VRAM, but `supportsLocal()` returns false. Its `connect()` is presently a no-op because remote connection creation occurs during remote connection metadata loading. `prepXfer` expects a peer QP and imported-style target metadata. Merely changing the boolean would therefore not implement local copies. [Backend capabilities](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.h#L56-L78), [connection behavior](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1420-L1455), [request preparation](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1629-L1702)

GPUNETIO currently accepts requests from the host, copies prepared descriptors into a CPU-accessible request ring, and launches internal GPU RDMA kernels on either an attached CUDA stream or a stream pool. Registration is outside the per-transfer issue path: GPU memory tries DMA-BUF registration with a legacy PeerDirect fallback; host memory uses `ibv_reg_mr`. MR ownership remains with the registration metadata. [Registration and metadata](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1511-L1617), [MR lifecycle](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/verbs/verbs.cpp#L588-L684), [ring publication](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1724-L1808)

GDRCopy here is not a third payload transport equivalent to CE or NIC DMA. GPUNETIO uses it to make request/control storage CPU-accessible, falling back to mapped host storage; the separate device utilities use GDRCopy for host-published control memory. [GPUNETIO control storage](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L423-L445), [host-published device memory](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/utils/device/device_ops_cuda_gdrcopy.cpp#L106-L181)

### Separate GPU device API

The public device API exposes `put`, `atomicAdd`, `getXferStatus`, and `getPtr`. Its current dispatch arms are `UCX_DIRECT` and `PROXY`. The proxy is GPU-initiated but uses CPU workers to perform transport submissions. GPUNETIO does not currently override either `prepMemView` method or have a device-dispatch arm. Thus host-stream GPUNETIO must not be presented as already providing a generic GPU-callable NIXL GPUNetIO API. [Public API](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/device/gpu/nixl_device.cuh#L31-L146), [execution modes](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/device/gpu/device_types.cuh#L33-L88), [dispatch](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/device/gpu/impl/device_dispatch.cuh#L109-L181), [proxy contract](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/utils/device/proxy/README.md#L20-L54)

### CE is a mover, not an existing NIXL backend name

This source tree has no standalone CE/CUDA-memory-copy plugin. UCX supports local transfers, but selecting UCX alone does not establish which transport or physical copy mechanism performed them. Its data path delegates read/write to UCX and flushes for completion. The device utility's synchronous `cudaMemcpy` is a setup/control-memory helper, not the NIXL payload backend. Use the actual existing CUDA copy route as the CE baseline, or pin and verify UCX's CUDA-copy transport before calling it a CE measurement. If a guaranteed CE lane must be exposed entirely inside NIXL, a narrow local CUDA-copy plugin could be added within the existing backend interface, but it is not present today. [Plugin tree](https://github.com/ai-dynamo/nixl/tree/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins), [UCX submission/flush](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/ucx/ucx_backend.cpp#L524-L598), [CUDA utility copy](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/utils/device/device_ops_cuda.cpp#L99-L121)

## 2. Recommended first scope: A, local host-API GPUNETIO

All items in this section are proposed, not current behavior.

1. **Bound the topology:** one NIXL agent/process, one communication GPU, one NIC, ordinary registered HBM and host DRAM. Keep peer-owned GPU buffers, VMM/multicast allocations, cross-process local peers, and multi-NIC policy outside the first contract. Logical same-host communication between two agents is not identical to a same-agent transfer.
2. **Implement real local initialization:** create a same-NIC paired/loopback QP arrangement appropriate to the supported NIC/SDK, bind the self peer, and retain it for the backend lifetime. Do not run the remote TCP metadata handshake against oneself. Advertise `supportsLocal()` only when initialization and topology/registration requirements are met.
3. **Implement `loadLocalMD`:** derive target address/rkey information from the existing registered MR. Do not re-register, serialize/parse, or exchange windows on every transfer. Keep registration ownership distinct from the self-target view, with a lifetime relationship that prevents the target view outliving its MR.
4. **Reuse host READ/WRITE:** for HBM→DRAM, local HBM is the WRITE source and self DRAM is the target; for DRAM→HBM, local HBM is the READ destination and self DRAM is the source. Use existing registered descriptor lists, prepared requests, `postXferReq`, `getXferStatus`, and release. No semantic dependence on tokens, embeddings, KV blocks, or layer boundaries.
5. **Keep posting and completion internal:** reuse GPUNETIO registration/QP/WQE mechanisms, with a deliberately small posting footprint. Optimize descriptor publication or launch aggregation only where it is actually on the critical path. Do not require a persistent application kernel to try the lane.

This scope can realize NIC-assisted data movement and a different issue-resource tradeoff without B. The gain is a hypothesis about relevant transfer shapes and concurrent work, not a universal bandwidth claim. Relative to CE, NIC use spends NIC queue capacity, PCIe bandwidth, GPU issue/poll resources, and setup/storage state. Relative to already-batched GIN, even the “fewer submissions” claim must be demonstrated rather than assumed.

## 3. Optional second scope: B, true GPU-callable GPUNETIO

B is a distinct extension, not a prerequisite for A. It would reuse `prepMemView`/`releaseMemView` and tagged device dispatch rather than create a parallel public framework.

- Add a GPUNETIO execution mode and a device-side implementation. Preparation must produce bounds, keys, owner/context identity, queue/channel references, and stable completion state usable without host request construction.
- Retain the existing public `put`/status convention where possible. A same-agent host-source PUT into HBM is conceptually possible with registered host addresses in WQEs, but must be an explicit capability, not assumed from today's GPU-oriented API text. If the desired operation is specifically GPU-issued pull, adding a true device `get` is extra public scope because NIXL has no current public device `get` operation.
- Do not introduce atomics merely to support copying. Unsupported `atomicAdd`, `getPtr`, execution levels, defer flags, or host-memory roles should fail explicitly rather than silently choose a CPU proxy.
- Validate both views belong to compatible backend contexts and communication GPU; the current dispatch uses the destination view to select an implementation. A new mode must not reinterpret another backend's source handle.
- Device status must distinguish acceptance from completion, preserve request-generation identity across slot reuse, and define source lifetime and destination visibility. A GPU-visible CQ ticket alone is not a complete cross-backend copy contract.

The existing source already distinguishes successful submission from completed operation in the UCX device adapter, returning `NIXL_IN_PROG` for accepted submissions and reserving success for completion polling. Preserve that semantic distinction. [UCX device status mapping](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/device/gpu/impl/ucx/device_ucx.cuh#L65-L117)

## 4. Selection and capability contract

Prefer the existing `optionalArgs.backends` restriction for an explicit GPUNETIO/UCX choice and `queryXferBackend` to confirm the selected backend. Do not overload `customParam` with an incompatible selector: GPUNETIO currently interprets its preparation-time payload as a raw `cudaStream_t`. [Backend restriction and custom payload](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/cpp/nixl_types.h#L164-L240), [query chosen backend](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/cpp/nixl.h#L369-L379)

The policy is conceptually `auto`, `force_ce`, or `force_gpunetio`, but those are proposed policy values, not existing NIXL flags. `force_gpunetio` must fail when unavailable, not silently fall back. `auto` may choose a fallback only before submission, with an observable reason. Once any operation is posted, retrying the same copy on another lane is unsafe without first establishing that the first lane has quiesced. Initially prefer explicit selection; do not pretend that a calibrated automatic crossover exists.

Minimal capability information, scoped to the instantiated backend and registered pair:

- Same-agent DRAM↔VRAM support and permitted READ/WRITE roles.
- Host-stream submission versus true device submission; do not conflate them.
- Required host pinning/MR eligibility, GPU/NIC/context identity, and supported allocation kinds.
- Consumer visibility scope: host-visible completion, owner-GPU visibility, peer-GPU visibility, and stream-ordered completion support.
- Descriptor/byte limits, queue capacity/backpressure, overlap restrictions, and supported channels/execution levels.

These can start as a small internal capability record, with query exposure only where selection needs it. `getSupportedMems()` and `supportsLocal()` alone are insufficient to express a valid source/destination pair and completion promise. No general-purpose topology scheduler or new cost-model framework is required.

## 5. Completion and ownership are the non-negotiable contract

Define three separate boundaries: **source ready for NIC/CE**, **transport no longer accessing source and destination**, and **destination visible to the named consumer**. A completion flag must state which boundary it establishes. For a generic copy primitive, successful completion should establish the promised consumer visibility, or the API must expose a required visibility step unambiguously.

Current GPUNETIO READ waits for a final CQE inside its posting CTA (and conditionally issues an SDK-specific extra operation); WRITE's posting CTA submits and publishes bookkeeping but does not wait for completion. A separate persistent kernel polls completions and posts notifications; CPU `checkXfer` observes the completion list. Therefore an event recorded immediately after a WRITE posting kernel must not be treated as completed payload DMA. Attached-stream submission orders the producer before posting, but is not by itself a general destination-ready stream primitive. [READ and WRITE kernels](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_kernels.cu#L28-L215), [GPU completion processing](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_kernels.cu#L257-L343), [host polling](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1993-L2030)

For inbound GPU writes, select the supported SDK/CUDA visibility protocol for the actual platform and consumer. Do not substitute a generic fence or host boolean for this proof. CUDA exposes native RDMA-write ordering scope and flush capabilities; its flush API may be unnecessary when native ordering already covers the consumer, and separate hardware mapping paths can be distinct ordering domains. The proposal must include visibility work in both its semantic promise and its cost. [CUDA RDMA ordering](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html#synchronization-and-memory-ordering), [flush and scope contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html), [capability enums](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TYPES.html)

Prepared descriptors/views, MRs, queues, completion slots, source buffers, and destination buffers remain alive until their operations are quiescent. An error is not proof of quiescence. Shutdown must stop new submissions, settle/drain device work, stop progress, close/export-release QPs, then destroy dependent control/MR resources. Use bounded backpressure instead of overwriting live rings. Preserve device/context guards and reject incompatible attached streams. Do not expand the first feature to peer-GPU visibility simply because the existing remote backend can register some peer-owned memory.

## 6. Cost structure and architectural implications

Registration/window/endpoint setup is an allocation/backend-lifetime cost for a well-prepared implementation, not inherently a per-copy cost in either raw GPUNetIO or a GIN design. Separate it from per-request costs.

At this snapshot, NIXL GPUNETIO has 512-descriptor request chunks, 32 request-ring positions, and 128 completion entries. Host preparation validates lengths, resolves metadata, coalesces eligible READ ranges, optionally pairs WRITE SGEs, reserves ring positions, and publishes the prepared data. It still does real CPU work. Request-ring positions are held until release. Reusing registrations and descriptor lists does not magically eliminate preparation of new changing index lists. [Constants/layout](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend_aux.h#L98-L159), [preparation](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1629-L1812), [release](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L2060-L2090)

Single-QP READ and WRITE launch one CTA of 512 threads per chunk, so requests exceeding one chunk can require multiple host CUDA launches. Multi-QP WRITE uses one grid of 2–4 CTAs, each 512 threads, with chunks striped among QPs. A separate persistent progress grid has three one-thread CTAs. These are source facts about launched work, not measurements of achieved occupancy or resource contention. [Launch configuration](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_kernels.cu#L458-L559), [per-chunk host posting](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1900-L1920)

The defensible hypothesis is therefore a different balance of CPU submission/preparation, GPU issue/poll residency, DMA throughput, and dependency latency for registered local scatter/gather copies. It is not “GPUNetIO uses no SMs,” “registration disappears,” “all requests become one launch,” or “a CQE/stream event automatically means every consumer sees the destination.”

## 7. Concrete change seams for an engineering handoff

No implementation is included; these are the smallest relevant ownership boundaries.

- `src/plugins/gpunetio/gpunetio_backend.h`: conditional local support, a `loadLocalMD` override, and owned local-endpoint state. For B only, add the two `prepMemView` overrides and `releaseMemView`.
- `src/plugins/gpunetio/gpunetio_backend.cpp`: self-endpoint initialization in `connect(self)`, local target-metadata construction, local request eligibility/dispatch, lifecycle and quiescence. Preserve existing remote behavior and the attached-stream binary parameter convention. This is also where registration/device ownership, request reservation, and polling are already managed.
- `src/plugins/gpunetio/verbs/verbs.cpp` and `verbs/verbs.h`: the low-level same-NIC paired-QP setup/reuse only if the existing QP abstraction cannot express it; reuse existing MR registration rather than duplicate it.
- `src/plugins/gpunetio/gpunetio_kernels.cu` and `gpunetio_backend_aux.h`: only the local posting/completion and visibility state needed for A, plus any separately justified chunk-aggregation change. Do not automatically couple local support to a rewrite of existing completion, ring, or multi-QP policies.
- `src/plugins/gpunetio/gpunetio_plugin.cpp`: explicit opt-in initialization/configuration if needed, with resolved capability information. Selection itself continues through existing backend handles.
- For B only: `src/api/device/gpu/device_types.cuh`, `src/api/device/gpu/impl/device_dispatch.cuh`, a GPUNETIO device implementation header, and matching device-view allocation/release. Modify `nixl_device.cuh` only if the agreed scope adds a new public operation such as device GET; a backend-specific PUT implementation does not require a new public operation.

For A, `src/api/cpp/nixl.h`, `nixl_types.h`, and the core self-agent transfer path should need no public signature change. A guarantee-specific capability query or future stream completion option can be additive, but is not a reason to invent another transfer API. A guaranteed standalone CE plugin, if desired, is a separate narrow backend addition rather than a redefinition of UCX.

## Decision boundary

Approve the architecture as an **optional, explicit, same-agent NIC-assisted lane under existing host NIXL READ/WRITE semantics**, contingent on local endpoint/MR support and a precise destination-visibility contract. Keep true GPU-callable NIXL GPUNetIO as a separate extension. Keep CE as a supported alternative, make overrides observable, and do not elevate either lane to a universal default or claim superiority over an already-batched GIN design from source inspection alone.
