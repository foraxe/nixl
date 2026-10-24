# DeepInfer transfer-path trace: CE, NCCL GIN/GDAKI, and host memory

Inspected 2026-10-22. Read-only source research; no build, GPU execution, benchmark, remote access, or repository modification performed.

## Executive result

1. **DeepInfer provides a strong motivation for NIC-assisted fine-grained host→HBM copies, but not released code for that KV swap-in.** Its deployment Q&A describes 3FS→host shared-memory fetching, then per-layer H2D overlap inside an engine thread, and says an RDMA NIC is used because CUDA copy engines perform poorly on fine-grained KV-block copies. The transport is described as GDR; the document does **not** identify GPUNetIO, NIXL, NCCL GIN, the NIC posting API, or GPU-versus-CPU initiation. Its README explicitly excludes the complete KV-cache system from this release. [Q&A lines 155–157](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L155-L157), [release boundary](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md#L38-L42).

2. **The actual released GPU-initiated sparse host-memory path is Engram, via DeepEP and NCCL GIN/GDAKI, not an identified NIXL path.** The source selects CPU RDMA storage by default on NVIDIA and invokes `EngramBuffer.fetch`; the kernel issues `gin.get` from registered storage windows into a GPU receive window. This is sparse n-gram embedding gathering, not the omitted KV swap-in. [allocation](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/parallel_utils/userbuffer.py#L127-L181), [fetch integration](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/engram.py#L195-L219), [device gets](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh#L109-L129).

3. **Registration is already amortized over the buffer lifetime.** Engram construction creates communicator/device resources, memory mappings and windows. Ordinary fetches reuse them and do offset/window selection, request posting, batching, and completion. A thinner GPUNetIO wrapper could reduce collective/symmetric-window lifecycle constraints or software overhead, but this trace establishes neither a per-fetch registration bottleneck nor a GPUNetIO latency advantage. NIC registration cannot be assumed to disappear.

4. **“NIC moves the payload” is not “zero SM use.”** Released Engram launches an issue kernel sized to every SM and a separate per-layer wait kernel. The NIC performs the bulk transfer, but issue/addressing/batching/completion consume GPU execution resources. This must not be conflated with the Q&A's separately described KV path.

## Source pins and supported configurations

- DeepInfer: `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, latest main found during inspection, commit timestamp 2026-10-21 04:55:43 UTC.
- Its vendored DeepEP: `f6aa38fe06cd4a0489e7730365d4e0114970a2f3`, verified from the [submodule update commit](https://github.com/deepseek-ai/DeepInfer/commit/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9).
- DeepInfer advertises NVIDIA SM100 and Huawei Ascend 950; Linux, GCC≥13, Python 3.12, CUDA 13 or CANN 9.2. [README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md#L33-L52).
- NVIDIA dependency files pin PyTorch `2.13.0+cu132`; build.sh force-installs `nvidia-nccl-cu13==2.32.3`. [requirements](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/requirements-nvidia.txt#L1-L5), [build](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/build.sh#L13-L22).
- Pinned DeepEP lists Hopper or newer, CUDA Toolkit≥13.1 for runtime JIT, C++20/std::format, PyTorch≥2.10, NCCL≥2.32.3, NVLink intra-node and RDMA inter-node. Engram is experimental and requires an NCCL build providing `ncclGinOptFlagsWarpGet`. These are DeepEP requirements, not evidence DeepInfer supports all Hopper deployments. [DeepEP README](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/README.md#L37-L69).

## A. What is and is not established about KV swap-in

The public Q&A supports this sequence:

`3FS storage → host shared memory [separate fetching process] → per-layer HBM destinations [swap-in thread, overlapped with forward computation]`.

It attributes the fine-grained H2D transfer to an RDMA NIC and states that this saturates PCIe without consuming SMs. That is an author assertion about production behavior, not an accompanying isolated measurement: no exact block-size distribution, CE baseline API/batching, NIC model, topology, direction-specific bandwidth, launch model, registration lifetime, latency distribution, or reproducible swap-in microbenchmark is given there. The missing cache-management implementation is explicitly acknowledged in [Q&A line 21](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L21).

Consequently:

- Do not say “DeepInfer uses NIXL GIN for KV swap-in.” No such implementation was found.
- Do not say “DeepInfer's KV swap-in is GPU-initiated GPUNetIO.” The document does not specify that.
- Do say “DeepInfer publicly motivates NIC-assisted fine-grained host-shared-memory→HBM transfers; the closest released GPU-initiated example is its separate Engram path.”

Indexed GitHub searches of DeepInfer returned no `nixl` or `gpunetio` matches. This is a bounded search finding, not an exhaustive proof about private infrastructure or every transitive dependency.

## B. Actual released Engram call path

### 1. DeepInfer allocation and reuse

`_UserBufferManager.allocate_engram` is intended to run once and retains `self.engram`. It calculates GPU receive-buffer size, per-rank storage size, QPs and depth; uses `use_cpu_rdma_storage = not DEEPINFER_ENGRAM_FORCE_GPU_STORAGE`; enables hybrid mode; and calls `set_config`. [userbuffer.py lines 127–181](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/parallel_utils/userbuffer.py#L127-L181).

The force-GPU variable evaluates false when unset on NVIDIA; Ascend forces it true. [boolean default](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/environ.py#L12-L13), [Engram setting](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/environ.py#L146-L147).

`EngramRuntime.initialize` writes layer embedding tables once, releases original storage references, and `fetch(hashes)` subsequently returns one hook per layer. [engram.py lines 195–219](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/engram.py#L195-L219).

### 2. Engram constructor: the cold path

- Estimates registration address-space requirements and may set `NCCL_WIN_STRIDE`.
- In hybrid mode creates CPU/GPU storage shareable handles and all-gathers them.
- Temporarily selects `NCCL_GIN_TYPE=3`; the source comment says selecting only GDAKI accelerates initialization.
- Calls `get_nccl_comm_handle(group, force_new_comm=True)`, so this specific Engram path does not reuse the PyTorch communicator even though generic DeepEP supports that.
- Creates the runtime, then synchronizes GPU/group/GPU.

Sources: [Engram constructor lines 61–104](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/engram.py#L61-L104), [communicator behavior lines 44–77](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/comm/handle.py#L44-L77).

The C++ `Context` explicitly asserts GDAKI capability and sets `reqs.ginType = NCCL_GIN_TYPE_GDAKI`, exclusive GIN contexts, QP count, queue depth, signals, and full/rail/custom-stride connectivity before `ncclDevCommCreate`. This is not merely an optional GIN environment hint. [context.cpp lines 121–164](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/context.cpp#L121-L164).

### 3. Host/GPU memory scope and windows

Memory is purpose-allocated, not accepted as arbitrary pageable host pointers:

- CUDA Driver VMM allocation properties choose pinned device memory or `CU_MEM_LOCATION_TYPE_HOST_NUMA`; GPU allocations require GPUDirect RDMA/VMM support. [symmetric.hpp lines 21–74](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/symmetric.hpp#L21-L74).
- GPU receive/workspace and CPU/GPU backing storage occupy adjacent virtual-address regions. In hybrid mode, storage from intra-node ranks is exported/imported as POSIX FD handles and mapped together. [allocation](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/symmetric.hpp#L137-L176), [hybrid sharing](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/symmetric.hpp#L192-L260).
- CPU-backed allocations enable `NCCL_ELASTIC_BUFFER_REGISTER`. [lines 325–328](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/symmetric.hpp#L325-L328).
- A main window is registered with `NCCL_WIN_STRICT_ORDERING`; each storage shard window uses `NCCL_WIN_GIN_ONLY`. Storage-window handles are copied to GPU memory once. The source explicitly notes window registration is collective. [context.cpp lines 195–222](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/context.cpp#L195-L222).
- Deregistration and device-communicator destruction happen in finalization, not each get. [lines 244–265](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/context.cpp#L244-L265).

The layout accommodates local-node storage and multi-node peers. The get loop is organized by GIN peer and registered shard, not a generic `(src_pointer, dst_pointer, byte_count)` copy API. Whether a chosen machine's same-node/self-NIC connectivity is admitted and performant remains a platform/runtime check; this code trace alone does not demonstrate it on hardware.

### 4. Steady-state fetch and completion

Python `fetch` accepts an int32 `[layers, tokens, entries_per_token]` index tensor, issues gets for all layers, and returns layer wait hooks. [Python API](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/engram.py#L157-L169).

The C++ wrapper builds views into retained receive memory; optionally allocates FP8 scale-factor output; allocates a request-tracking tensor; launches the issue kernel; and constructs hooks that launch a wait kernel per layer. There is no window-registration call in this method. [engram.hpp lines 157–248](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/buffers/engram.hpp#L157-L248).

Device processing:

- Reads embedding indices, partitions requests by peer, packs pending requests into warp-sized batches.
- Computes local shard selection and source/destination byte offsets.
- Calls `gin.get<ncclTeamTagWorld, ncclCoopThread, ncclGin_SegmentMixed>` with retained remote/local windows.
- Uses aggregate-request options until a threshold or final batch rings the doorbell.
- Performs one `flush_async` per active peer/context/layer after coordinating issue warps; wait hooks use `gin.wait`.

Sources: [issue/batching and gets](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh#L80-L205), [wait](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/engram/engram_fetch_wait.cuh#L9-L27).

The DeepEP wrapper's `get_sym_offset` is a pointer subtraction; `get` forwards retained windows/offsets into NCCL. [offset](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/comm/handle.cuh#L56-L61), [get wrapper](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/comm/handle.cuh#L128-L146). This does not prove NCCL's deeper address/key lookup is free; its cost must be traced/profiled separately.

**SM footprint:** issue launch uses `num_sms = device.get_num_sms()` with 32 warps/CTA, and one CTA per SM in the grid. The wait launch is one 1024-thread CTA per layer. These launch dimensions do not establish sustained occupancy for the full DMA interval: work-dependent early exits and NIC asynchrony matter. They do establish nonzero GPU issue/wait work. [launches](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/engram/engram.hpp#L35-L118).

For FP8, scale factors use a separate direct-load/store path through symmetric GPU pointers, so not every byte in Engram processing is necessarily moved by the NIC. [scale-factor path](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh#L208-L252).

## C. CE remains present and useful in the release

- DeepInfer state/block cloning loops call `cuda_copy_async`, whose implementation is `cudaMemcpyAsync(..., cudaMemcpyDefault, stream)`. Both source/destination derive from the same state or block tensor. These are cache-cloning paths, not the unpublished host swap-in. [scheduler](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp#L1264-L1299), [CUDA wrapper](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/cuda_utils.cpp#L91-L94).
- Engram initial `write` uses `cudaMemcpyAsync` per layer to populate storage. GIN is its fetch path, not a universal replacement for all initialization copies. [write](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/buffers/engram.hpp#L109-L154).
- DeepEP Bucket all-gather uses CE for pure NVLink, SM-issued RDMA plus CE NVLink forwarding in hybrid mode, and an SM kernel for pure RDMA. Do not read the high-level `num_sms=0` interface as proof every transport is SM-free. [dispatch branches](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/bucket/all_gather.hpp#L36-L114).
- The CE helper actually uses `cudaMemcpyBatchAsync` with `cudaMemcpyFlagPreferOverlapWithCompute`, plus batched CUDA stream wait/write operations. It is a relevant modern CE comparator, rather than only a CPU loop of individual memcpys. [driver helper](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/driver/driver.cpp#L41-L86).

## D. Performance evidence and limits

### What exists

DeepEP's Engram test constructs and populates the buffer before timing. It reports issue-kernel and wait-kernel durations, derived GB/s, and messages/s for a workload of specified layers/tokens/entries/hidden size. Its default hidden=256 and BF16 imply 512 bytes per requested embedding; `--use-cpu-rdma-storage` is an explicit test option, so a default test is not a CPU-storage test. [test setup/options and metric calculation](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/tests/engram/test_engram.py#L14-L38), [timing/options](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/tests/engram/test_engram.py#L99-L146).

Although the test labels fetch+wait “end-to-end,” its timing is the sum of profiled CUDA kernel averages, not a complete host-visible request wall clock. `bench_kineto` profiles CUDA activities and returns average per-kernel durations. Registration/communicator initialization, host dispatch, and complete application critical-path effects are not demonstrated by that metric. [timing helper](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/utils/testing.py#L113-L138), [CUDA profiling/averaging](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/utils/testing.py#L153-L204).

### Model constants are not measurements

- Engram QP sizing uses a hard-coded 10 µs read RTT and queue-capacity logic. [C++ estimator](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/buffers/engram.hpp#L273-L298).
- Python supplies either 90 or 45 MPPS based on a capability check. [Python estimator](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/engram.py#L129-L144).
- Hybrid all-gather's chunk-selection model assumes 10 µs CE overhead. [model](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/bucket/all_gather.hpp#L120-L134).

These constants should not be presented as measured general-purpose latency, platform guarantees, or a GPUNetIO-vs-CE crossover.

## E. Implication for a reusable NIXL local-copy primitive

Treat the proposed primitive as a distinct API/design problem: a local registered-DRAM/HBM transfer list, reusable registration/queue state, and explicit completion/stream ordering. Engram demonstrates batching and retained-registration patterns but adds embedding-index interpretation, communicator/rank topology, all-layer scheduling, and scale-factor semantics that a generic copy API may not need.

Three separable claims need separate evidence:

1. **NIC vs CE:** Does the same local small-block workload move faster or interfere less with compute using NIC DMA than modern batched CE?
2. **GPU vs CPU initiation:** Does on-device descriptor generation/issue eliminate a real CPU synchronization or launch bottleneck, after charging issue/wait SM cost?
3. **Direct GPUNetIO vs NCCL GIN/GDAKI abstraction:** Does bypassing collective/symmetric-window machinery improve cold setup, dynamic registration churn, hot per-operation instruction cost, or only API fit? The released path explicitly uses GDAKI; it may share the same lower-level GPUNetIO NIC machinery, which must be compared at the exact NCCL/runtime version.

For claim 3, report cold setup separately from warm reuse. The current source already amortizes its window setup, so a win attributed to “avoiding registration on every copy” would be an unfair baseline. No measurements from this investigation establish any of these speedups.

