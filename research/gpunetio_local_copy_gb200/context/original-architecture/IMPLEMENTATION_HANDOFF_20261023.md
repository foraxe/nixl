# NIXL local GPUNetIO: implementation handoff

2026-10-23. Design contract, not an implementation or performance result. No hardware execution, builds, tests, repository changes, outreach, or review of declined PR #2222. Public baseline: NIXL `d64231659c8194760b60d163b514a3eba1ac22ee`.

## Decision and advancement

The previously agreed scope remains unchanged: explicit opt-in, same-agent/process, one NIC and communication GPU, registered pinned Host DRAM → owner-GPU HBM through RDMA READ. Reuse NIXL host descriptors, request preparation, posting and status. READ-only was already decided in CTO_DECISION.md §8; it is not a new finding.

This handoff advances that decision into startup/rollback ordering, endpoint ownership, and request terminal-state requirements. The concrete source-level trap is that core checks `supportsLocal()` before invoking `connect(self)`. A capability that becomes true only after that connection would never initialize. Merely returning true is also insufficient.

## E branch reconciliation completed (2026-10-23)

Read-only inspection of the actual clean E checkout confirmed branch `experiment/gpunetio-byte-qp-20261022`, HEAD `e932593300366073dfd5f3333ef1d42f6abfbc0a`, base `7f9d5b530ad8ed01113a4662a2beb2fdb610d89f`. Reuse its MR registration, GPU READ and QP/CQ machinery and partial-post accounting. Self paired-QP, self-peer publication and non-owning loadLocalMD remain missing. No need to rewrite E byte-balanced WRITE work.

The prior offline/unknown-E blocker is resolved. See E_BRANCH_ALIGNMENT_20261023.md for exact files and line references. Historical local reports record 131 functional processes rather than the previously known 90; these were not rerun, and two-process endpoint smoke is not same-agent support. Current SDK deployment, visibility and fault-quiescence remain unverified.

The actual E also exposes the error-lifetime seam: its global CQ error can terminate progress and surface a negative status before all other accesses are proven stopped; core release/destructor and metadata-expiry bypass need a bounded cross-layer correction. Backend references alone cannot protect caller-owned memory from overwrite. Resolve this terminal-state contract before advertising local READ support.

## Endpoint lifecycle and atomic startup

Proposed states: DISABLED → INITIALIZING → READY → QUIESCING → CLOSED. Initialization failure transitions to FAILED/CLOSED after rollback; runtime failure moves READY → FAILED_QUIESCING → CLOSED. State names are internal design labels, not new public API values.

1. **DISABLED:** local opt-in is off; preserve current remote behavior and do not allocate self resources.
2. **INITIALIZING:** opt-in is captured during backend construction so `supportsLocal()` can declare the feature before core calls `connect(self)`. Complete prerequisites already owned by backend initialization, then create requester/responder QPs on the selected NIC/protection domain, connect both directions, and establish required queue/progress references. Do not use the remote TCP handshake against oneself.
3. **Commit READY atomically:** only after all self resources are valid, publish the self-peer entry used by `prepXfer`. Do not expose a half-connected entry. Core inserts a backend into its memory/backend selection lists after successful `connect(self)`; return failure before that publication if local setup fails. With explicit opt-in, fail backend creation rather than silently deliver a remote-only or alternate implementation.
4. **Rollback startup:** keep newly acquired resources under temporary ownership until commit. On failure, unpublish any provisional self state, stop/drain any work started by this attempt, release dependent GPU/export handles and QPs in dependency-safe order, and destroy only resources acquired by this attempt. Do not destroy shared registrations or other existing peers. Cleanup must be safe after partial construction and must not double free.
5. **READY registrations:** `loadLocalMD` creates a non-owning or explicitly shared-lifetime target view of an existing MR's address/rkey. It does not register again. Invalidate target views and prepared descriptors when registration is revoked; deregistration must not overtake in-flight access. One MR has one destruction owner.
6. **QUIESCING:** reject new preparation/posting, settle posted accesses, retain necessary progress while draining, then stop progress and destroy endpoint/control resources. Deregister dependent memory only after all possible access has stopped. Runtime error does not bypass this order. If quiescence cannot be established, retain/quarantine resources and report the blocked teardown rather than declaring them safely released.

An initialization failure and a runtime failure are different: the former must leave no usable backend/self endpoint; the latter may leave already-issued DMA and must retain its lifetime obligations.

## Dependency-ordered patch packages

### 1. Endpoint and metadata ownership

Files: `gpunetio_backend.h/.cpp`; `verbs/verbs.h/.cpp` only if paired-QP construction/ownership requires it; plugin configuration only for opt-in.

Implement the lifecycle above, self-peer lookup and local MR-derived metadata. Preserve remote setup. Avoid per-transfer connection or registration. Prefer the existing host signatures, but compatibility is conditional on the error/quiescence merge gate below. Do not promise an unchanged host contract before that gate closes. Core startup ordering is a dependency, not a reason to redesign core.

### 2. Narrow eligibility and request preparation

Depends on READY endpoint and valid local metadata. Accept only self READ with pinned DRAM source and owner-GPU HBM destination, compatible context/NIC/registration, matching lengths, valid bounds and nonconflicting destinations. Exclude overlap/memmove semantics. Reject local WRITE, peer GPU, cross-process, VMM/multicast, unsupported stream context and unsupported notification options before posting; do not silently ignore them.

Reuse existing descriptors, request chunks, rings, backpressure and READ posting path. Keep `optionalArgs.backends` selection and backend observability. Preserve the existing `customParam` CUDA-stream representation. Reject unsupported requests rather than silently choosing CPU proxy, staging or another backend. Remote requests retain their existing semantics.

### 3. Completion and terminal-state correctness

Depends on packages 1–2. Modify backend/kernel completion state only where needed to close the following contract. Reuse supported SDK consistency work; do not replace it with an invented fence. Every chunk, async error and partial submission participates in the aggregate result. Confirm the actual SDK visibility scope rather than treating the pinned READ implementation as universal proof.

### 4. Teardown and compatibility closure

Depends on package 3's ability to establish quiescence. Ensure request release, deregistration, endpoint destruction and backend shutdown cannot free live DMA resources; safe reverse-order cleanup applies to every partial startup stage. Preserve the non-opt-in and remote paths. Keep these acceptance requirements separate from performance optimization and do not bundle a ring rewrite or multi-QP work.

## Request acceptance and terminal states

These are internal semantic states, not proposed public enum additions.

- **PREPARED, not submitted:** all eligibility checks passed; request owns required metadata/resource references. Caller finishes and publishes source writes before posting. Source remains immutable and both buffers stay live while NIC access is possible.
- **REJECTED_BEFORE_SUBMIT:** invalid/unsupported request or resource shortage occurred before any WQE/launch could access payload. Release preparation-only reservations safely. Report the exact failure. An alternate path may be selected only now, under an explicit policy; forced GPUNETIO must report failure rather than fallback.
- **IN_FLIGHT / ACCEPTED:** at least some work may access memory. Acceptance/launch return is not consumer-ready completion and does not permit reuse, deregistration or fallback.
- **SUCCESS_QUIESCENT:** all fragments succeeded, all transport accesses finished, and the supported owner-GPU visibility protocol completed. Only now return successful transfer completion.
- **FAILED_DRAINING:** posting failed after an earlier chunk was issued, a CQ/visibility/runtime error occurred, or an uncertain post may have executed. Destination may contain partial data; stop further issue, preserve references and establish quiescence. Never report success because only the posted subset completed. No concurrent same-target CE retry.
- **FAILED_QUIESCENT:** all possible transport accesses have stopped, but the aggregate operation failed. Data remains invalid as a completed batch; safe resource release and an explicitly authorized retry policy are now possible.

**Mandatory A0 external-state rule:** FAILED_DRAINING remains externally `NIXL_IN_PROG`; record the failure internally and continue recovery/drain. Return a terminal negative transfer status only from REJECTED_BEFORE_SUBMIT or FAILED_QUIESCENT. This applies to both `postXferReq` and `getXferStatus`, including partial/uncertain submission. Internal failure telemetry may be emitted early, but it must not be confused with a terminal transfer result. There is no atomic rollback or transactional batch promise. A timeout alone is not quiescence. An uncertain submission is handled as potentially submitted, not as pre-submit rejection.

### Merge-blocking API/lifetime gate

Pinned core `getXferStatus` invokes `checkXfer` only while request status is `NIXL_IN_PROG`, and caches a negative backend status as terminal. Core `releaseXferReq` can delete the request after a negative status. Public release documentation permits cancellation of active requests or an error if abortion is impossible. The inspected APIs do not establish an explicit unsafe/poisoned-terminal-error contract requiring callers to preserve buffers. Therefore A0 cannot return an early terminal failure and rely on later polling or retained references to make it safe.

Retaining MRs or buffer references prevents some destruction but **does not prevent the caller overwriting/reusing an allocation**. That is why reference retention alone does not satisfy this gate.

Required implementation behavior:

1. While FAILED_DRAINING, keep the request externally in progress and retain its original internal failure cause. Recovery must continue even when normal completion progress encountered an error; an early global error return from existing `checkXfer` is not sufficient.
2. If `releaseXferReq` is attempted during draining, either establish quiescence before successful release or reject the release without deleting the live request. That release-attempt error is not terminal transfer completion; the request remains active, and caller must not reuse its buffers. Validate this exact behavior through the core/backend release interaction, not only a backend comment.
3. Produce terminal transfer error only after the actual SDK-supported QP abort/reset/drain sequence establishes that no late DMA can access either buffer. A CUDA stream synchronization or a failed CQ poll alone is not the proof. The exact supported sequence and its recovery bounds are not established by this inspection and are a merge gate.
4. Prevent core-side bypasses from falsely ending the lifetime contract: core can return an error when the request's metadata section expires before calling the backend. For A0, self registration/metadata removal must not create that situation during active DMA. Verify or add the required interlock; do not assume it from MR retention. Deregistration and teardown must follow the same drain rule.
5. If a bounded quiescence/recovery path cannot be established, do not ship A0 as a transparently compatible backend. A separately designed poisoned/unsafe status and explicit caller-side ownership protocol would require API/integration review and caller adoption. Merely documenting an existing generic negative status as unsafe is not an adequate substitute. An indefinite `IN_PROG` quarantine avoids unsafe reuse but does not satisfy the proposed bounded-failure acceptance gate.

This narrow source inspection establishes the integration hazard, not that current public NIXL promises reuse on every error or that the user's E implementation has this defect. The proposed terminal-error-is-quiescent behavior is an A0 requirement to verify/implement, not a claimed existing guarantee.

## Consumer-ready boundary and why READ first

Minimal promised consumer boundary: application observes successful `getXferStatus`, then submits the consuming kernel on the owner GPU. The backend must include whatever supported visibility work makes that promise true. Arbitrary already-running consumers, peer-GPU consumers and independent cross-stream/event consumers are not included.

READ-first avoids the particular WRITE issue/event ambiguity: the pinned READ posting CTA waits for its final CQE and conditionally performs consistency work; WRITE posting can finish before payload DMA. But READ does not make a CUDA event an aggregate success/error token. Keep event-based consumer-ready completion out of A0 until its visibility, all-chunk and error semantics are explicitly closed. Retain the actual linked SDK's supported consistency sequence and charge its cost.

## Do not build first

No public device GET or GPUNETIO device dispatch, persistent application kernel, automatic CE/NIC crossover, new CE plugin, Engram-specific gather API, reverse WRITE, multi-NIC or multi-QP tuning. Internal GPU posting does not constitute a public GPU-callable NIXL API. GIN/GDAKI itself uses GPUNetIO; this design claims an integration option, not superior DMA hardware.

## Pinned evidence

- [Core checks local support, then self-connects, then publishes backend selection](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/core/nixl_agent.cpp#L393-L425).
- [GPUNETIO current capabilities](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.h#L56-L78).
- [Request preparation and peer lookup](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1629-L1702).
- [READ completion/consistency and WRITE posting](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_kernels.cu#L28-L215).
- [Core status caching and metadata-expiry path](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/core/nixl_agent.cpp#L1319-L1355).
- [Core request release](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/core/nixl_agent.cpp#L1384-L1425).
- [Public post/status/release documentation](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/api/cpp/nixl.h#L332-L388).
- [Host aggregate completion polling](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_backend.cpp#L1993-L2030).
- [NCCL same-NIC paired-QP precedent](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport/net_ib/gdaki/gin_host_gdaki.cc#L874-L945).

Engineering approval: implement/reconcile packages 1–4 as one correctness-complete vertical slice, initially opt-in. Hardware qualification and workload value remain unestablished; neither is claimed by this handoff.
