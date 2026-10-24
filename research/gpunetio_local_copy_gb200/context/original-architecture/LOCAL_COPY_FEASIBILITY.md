# Same-node GPUNetIO as an alternative copy engine

Assessment date: 2026-10-22. Read-only source analysis; no hardware tests or performance measurements were run.

## Decision in one paragraph

**A same-NIC, GPU-posted RDMA copy is a technically credible reusable NIXL primitive, especially for fragmented registered host-DRAM ↔ HBM transfers. It is not a general replacement for CUDA copies.** Current NVIDIA code supplies a useful implementation precedent: NCCL GDAKI creates local-rank main/responder GPUNetIO QPs on the same NIC and connects them to one another. Start with an explicit, opt-in local registered-buffer path and a bounded descriptor batch, preferably READ into the initiating GPU for the host→HBM case. Keep direct CUDA/SM paths for local HBM and NVLink/C2C-connected memory unless actual measurements show an application-level win. The case to establish is lower request-to-consumable-data time or lower interference, not a higher theoretical interconnect bandwidth.

## What is established, and what is not

### Same-NIC loopback has concrete NVIDIA precedents

- **GPUNetIO local-QP pair:** NCCL commit `3e46c089f4ae4732722a419eb6096aead7e47a30` creates `self_gqp_list`, supplies a separate responder QP for the local rank, and mutually connects the main and responder QPs. Its connection setup enables remote READ and WRITE. This is stronger evidence than a same-host two-NIC test, and means two physical NICs are not a prerequisite for the design. It is a pair of local QPs, not literally one QP whose destination QPN equals itself. [NCCL local QP construction](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport/net_ib/gdaki/gin_host_gdaki.cc#L874-L945), [permissions](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport/net_ib/gdaki/gin_host_gdaki.cc#L466-L479).
- **Literally self-connected QP:** NCCL's ordinary IB flush path sets the destination QPN to the same QP's own QPN and uses local port addressing. It issues a one-byte RDMA READ from GPU memory into a registered host buffer. This establishes another real same-NIC loopback use, but its purpose is visibility, not sustained bulk copy. [QP setup](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport/net_ib/connect.cc#L1334-L1388), [RDMA READ](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport/net_ib/p2p.cc#L568-L603).
- DOCA 3.5.0 Perftest separately describes a LOCAL traffic partition between two NICs on one host. Do not use that alone as proof of the single-NIC case, and do not treat its GPU WRITE-only benchmark restrictions as universal GPUNetIO API restrictions. [Perftest](https://networking-docs.nvidia.com/doca/archive/3-5-0/doca-perftest).

**Remaining unknown:** the exact target NIC, firmware, virtualization/port configuration, DOCA version, queue type and memory-registration combination. The production precedents establish feasibility, not a universal support or performance guarantee. Prefer the demonstrated paired-QP construction until the target's single-QP-to-itself behavior is explicitly validated. Internal loopback versus a port/fabric path must also be established for the selected configuration; a successful operation alone does not identify its physical route.

### Initiation API, payload mover and route are separate questions

1. A CUDA memcpy call does not universally promise CE execution. CUDA's guide explicitly says the runtime may use other machinery. A comparison labelled “CE” must establish that the baseline really uses CE. [CUDA unified memory discussion](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html#exploiting-asynchronous-access-to-system-memory).
2. An SM copy/gather issues load/store traffic to locally or peer-mapped addresses. UVA identifies addresses; it does not itself select NIC RDMA. Peer accessibility depends on the actual PCIe/NVLink topology. [CUDA peer access](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#multi-device-peer-to-peer-transfers-and-memory-access).
3. GPUNetIO verbs prepare READ/WRITE WQEs and submit them to the NIC. For non-inline payloads, the intended copy is NIC DMA from registered source to registered destination, not an SM loop copying the bytes. However, `NIC_HANDLER_AUTO` can use a CPU-proxy mode; the existence of a device function does not prove CPU-free doorbell progress. [Pinned GPUNetIO v4.0.1 submission dispatch](https://github.com/NVIDIA-DOCA/gpunetio/blob/717ede55908bcd3a1da1a705485339a255efd21b/include/device/doca_gpunetio_dev_verbs_qp.cuh#L633-L684), [READ WQE preparation](https://github.com/NVIDIA-DOCA/gpunetio/blob/717ede55908bcd3a1da1a705485339a255efd21b/include/device/doca_gpunetio_dev_verbs_qp.cuh#L895-L947).
4. GPUNetIO also exposes a different DOCA DMA API. Its CPU↔GPU sample expressly uses a DPU DMA engine. That DPU requirement must **not** be transferred to ordinary ConnectX/BlueField RDMA verbs, and the DMA API must not be called the GPU CE. [DOCA 3.2.0 GPU DMA sample](https://networking-docs.nvidia.com/doca/archive/3-2-0/doca-gpunetio#gpu-dma-copy).

## Plausible win regions

These are architectural hypotheses, not measured outcomes.

| Memory pair | Judgment | Reason to consider NIC path |
|---|---|---|
| Registered host DRAM → HBM, many independent small fragments | Best first candidate | GPU-produced descriptors can be issued in a batch without returning to the CPU; NIC moves the payload while SMs do little per byte. Enough outstanding operations can amortize WQE/doorbell cost. |
| HBM → registered host DRAM | Credible secondary candidate | Similar resource separation, but producer release and CPU-visible completion must be explicit; direction-specific PCIe behavior may differ. |
| HBM → peer HBM without usable direct P2P | Conditional fallback | NIC may avoid a slower staged path if it can access both registrations. This does not imply all unsupported CUDA-P2P topologies support NIC P2P. |
| HBM → peer HBM with NVLink | Not a default replacement | CE and SM paths already exploit the direct inter-GPU fabric. NIC introduces an extra endpoint and a separate bandwidth ceiling. |
| HBM → HBM on the same GPU | Weakest default case | Direct copy stays in GPU memory; NIC copy exports the data out of and back into the GPU. Only a demonstrated CE/SM scheduling or contention benefit could compensate. |
| Grace-local DRAM ↔ HBM over NVLink-C2C | Do not generalize the PCIe-host case | Coherent, high-bandwidth CPU↔GPU access changes both the native baseline and NIC reachability. NVIDIA documents C2C as substantially wider than PCIe Gen5 x16. [Grace architecture](https://docs.nvidia.com/dccpu/grace-perf-tuning-guide/index.html). |

For ordinary NIC-mediated copy, each payload byte is read into the NIC and written out again. Those are **two DMA legs**, plus protocol, WQE and completion traffic. The upper bound is constrained by the source-read path, destination-write path, NIC internal throughput and every shared link. It is incorrect to automatically divide a PCIe full-duplex bandwidth figure by two: opposite directions can overlap. It is equally incorrect to assume NIC line rate or GPU HBM bandwidth is the achievable local-copy rate. If loopback is internal, external line rate need not be the limiter at all.

NIC routing does not acquire NVLink bandwidth merely because the GPU pair has NVLink. Conversely, on integrated/coherent platforms the NIC's actual route may include platform-specific C2C paths. Establish the real route instead of extrapolating discrete-GPU PCIe assumptions. DOCA recommends PIX/PXB GPU↔NIC placement, accepts some same-host-bridge configurations and warns about NODE/SYS performance. [DOCA topology guidance](https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-installation-and-setup).

## A practical minimal primitive

Keep the operation model-independent: an immutable prepared batch of registered source/destination ranges, offsets and lengths, followed by execution and a completion token. Reuse NIXL's registration, prepared-descriptor and completion machinery where it already provides this contract; do not force a model framework or persistent scheduling runtime into the abstraction.

For the initial host→HBM implementation:

1. Register a stable host window and a stable destination-HBM window against the chosen NIC/protection domain. Create a same-NIC local requester/responder QP pair once.
2. Represent each fragment as remote source address+rkey, local destination address+lkey and length. Here “remote” is an RDMA addressing role, not a statement that the bytes are on another machine.
3. A bounded GPU kernel consumes the prepared descriptor batch and posts RDMA READs, batching submission/doorbells with enough queue credits. The host supplies descriptors only if they originate on the host. If descriptors originate on the GPU, avoid a GPU→CPU→GPU control round trip.
4. Drain the relevant requests and apply the platform-required GPU visibility step before publishing completion. READ is a natural first choice because the initiating GPU owns the destination and the completion being consumed. WRITE remains a viable direction/policy choice; it is not inherently universally faster or slower.

Pinned GPUNetIO v4.0.1 provides thread and warp GET helpers, with the warp form preparing per-lane READs and one lane submitting the group. It also supports optional memory-consistency work. This makes a bounded batched implementation credible without a permanently resident communication kernel. [GET helpers](https://github.com/NVIDIA-DOCA/gpunetio/blob/717ede55908bcd3a1da1a705485339a255efd21b/include/device/doca_gpunetio_dev_verbs_onesided.cuh#L566-L769).

Treat kernel launch, GPU queue occupancy, polling and descriptor transfer as costs. One communication warp is not zero SM overhead, and a whole allocated CTA can reserve more resources than the active lanes suggest. Persistent progress may reduce launch latency but introduces residency, fairness and shutdown obligations. It should be an optional integration, not the initial reusable primitive's requirement.

## Correctness requirements that cannot be optimized away

- Registration/window lifetime must outlive all issued DMA, notifications and retries. Reusing an address is not proof that an old registration still names the same allocation. Do not assume a migratable managed allocation is a permanently stable RDMA window. Amortize registration over reusable buffers, but keep deregistration and reallocation invalidation correct. NVIDIA notes registration can be expensive and describes allocation-identity checks and drain-before-revocation requirements. [GPUDirect registration and ordering](https://docs.nvidia.com/cuda/archive/13.0.1/gpudirect-rdma/index.html#design-considerations).
- Establish producer→NIC visibility, NIC completion→consumer visibility and cross-thread publication separately. A CQE, a flag observed by one thread and arbitrary application data visible to another CUDA block are different events. Use the relevant library protocol, not an invented generic fence.
- The pinned implementation queries `CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING` and sets `need_mcst`. GET helpers can append a Dump WQE for platforms requiring it. Therefore the older generic GPUDirect warning about concurrently running kernels must not be misread as “GPU-side consumption is impossible”; supported GPUNetIO paths supply additional consistency mechanisms. Nor may those mechanisms be omitted just because a simple test happened to pass. [Capability query](https://github.com/NVIDIA-DOCA/gpunetio/blob/717ede55908bcd3a1da1a705485339a255efd21b/src/doca_gpunetio.cpp#L181-L193), [GET consistency](https://github.com/NVIDIA-DOCA/gpunetio/blob/717ede55908bcd3a1da1a705485339a255efd21b/include/device/doca_gpunetio_dev_verbs_onesided.cuh#L566-L644).
- No overlapping/conflicting ranges in an unordered batch. Maintain queue credits and wraparound invariants; never reuse WQEs or completion slots prematurely. Carry asynchronous errors to the NIXL completion result and provide a bounded failure/teardown path, rather than treating indefinite device polling as success.

## Three falsifiable adoption gates, not a benchmark matrix

1. **Path gate:** On one named platform/version tuple, the paired-QP local host→HBM batch must complete correctly with the intended NIC DMA route and intended GPU-doorbell mode. No silent CPU proxy, staging, unexpected external-port route, or per-transfer re-registration may be hidden in a claim of GPU-initiated local copy. Unsupported configurations fall back explicitly.
2. **Semantic gate:** A completed batch must be immediately safe for the real downstream consumer, including same-kernel/device consumers if that is the promised contract. Validate reuse and producer/consumer ordering, and make failure observable. If obtaining that guarantee needs a CPU round trip that erases the target benefit, reject that device-completion contract or narrow it honestly.
3. **Value gate:** For the one representative fragmented transfer shape motivating the primitive, beat the best applicable native baseline in request-ready→consumer-ready time or in throughput of the overlapping useful workload, with setup amortization, launch, synchronization and interference included. Compare against batched CUDA copies and a straightforward SM gather, not only a serial loop of tiny memcpy calls. CUDA 13.0.1 already documents `cudaMemcpyBatchAsync`; copies within a batch are unordered, so use equivalent independent descriptors. [Batched copy API](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY).

There is no universal fragment threshold to publish from this evidence. The break-even depends on per-batch launch/visibility latency, fragment issue rate, payload bandwidth, available concurrency, and contention. NIC copy trades CE/SM demand for NIC/PCIe/DRAM demand; it can harm simultaneous inter-node traffic even when its isolated latency wins. A new CUDA context or separate process also introduces scheduling/registration differences, so preserve the production context arrangement when evaluating the primitive.

## Version and scope cautions

GPUNetIO open-source `main` was also inspected at `b2b7a291ec70fdeeb1b27ba4b0ab8c8e260a729b` (2026-10-05, commit message identifies v5.0.0 rc1). It adds/changes capability and host APIs. The core READ, optional consistency and CPU-proxy-dispatch observations above were independently checked against the pinned v4.0.1 commit instead; none implies that the target NIXL plugin already uses these exact APIs. NIXL's actual linked DOCA/API version and its self-endpoint handling require separate code-level confirmation.

**Recommended architecture decision:** retain one general registered-copy contract with explicit backend/path capabilities; add a narrow local-NIC implementation behind it only after the three gates. The evidence is enough to justify a focused prototype, not enough to claim CE replacement or a performance win.
