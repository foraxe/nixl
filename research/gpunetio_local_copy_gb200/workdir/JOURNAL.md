# Kernel local copy

## 2026-10-23 — goal and scope

User correction supersedes ZIP's host-API-first implementation: allocate tensors,
prepare once, issue async copies directly inside the consuming kernel, wait there,
then consume. Keep the API thin; no per-operation CPU descriptor/request path.
Single process, one owner GPU/NIC, pinned host source and HBM destination, READ.
No PR publication or changes to E, #2222, PD, or existing device-owner branches.

Goal tool active in thread REDACTED_UUID.
Reuse E e932593300366073dfd5f3333ef1d42f6abfbc0a low-level QP/MR types;
evaluate SDK 3.5.0098 direct GET rather than extending the NIXL public device API.
Source/API inspection found SDK nonblocking GET and error-reporting CQ polling.
Cold initialization still needs registrations and mutually connected self QPs.

Experiment hypothesis: same-NIC paired QPs can pull registered host tensor slices
into owner HBM and make them consumable within the issuing kernel, using only
the SDK GET/consistency/CQ primitives. Baseline: cudaMemcpyAsync on identical
registered source/destination ranges, plus an independent computation control.
Metrics: full changing payload/guards, operation/window counts, in-kernel results,
request-to-consumer wall time. No assumed speedup. PASS requires exact payload,
correct rejection, bounded outstanding work, repeat/reuse and safe teardown;
FAIL on premature success/data mismatch; INVALID on ownership/config mismatch.
Kill only task-owned process on a bounded external timeout; never reset a GPU.
Build artifacts/raw logs stay remotely; retain source/reports/compact evidence.

Remote candidate: <H20_POD>. Live inventory: GPU7 free at 4MiB, no
compute process; GPU0–4 occupied. GPU5/6 used by adjacent Q1 work historically.
Recheck GPU7 immediately before each campaign, use a task-local lock.

## Implementation and causal checks

- Implemented standalone device Copy on existing exported E QP/MR classes and
  DOCA GET; no NIXL agent, OOB, host request or extra GPU progress kernel.
  Python preparation holds the actual pinned CPU/owner-GPU tensors.
- First runtime INVALID: system Torch2.11+cu130 requires a newer driver. Installed
  Torch2.7.1+cu126 with --target/--no-cache-dir in remote task root only; base unchanged.
  Direct CUDA smoke independently established the transport before Torch ran.
- INIT/RTR failure isolated to IB vs DOCA MTU enum bases (1 vs0); corrected the
  conversion. 4KiB smoke then passed all bytes and cleanup.
- Independent source reviewer identified missing ticket binding, first-fault
  preservation and unbounded/public-close lifetime. Added per-context cookie,
  admitted-range validation, sticky fault, and whole-cleanup retained worker.
  Close immediately invalidates Python device_handle; failed close retains tensors.
  Null-QP reset retry and recovery-pointer/cleanup data race were corrected.
- Actual CQ error now waits for cold controller to reset both QPs, then returns
  the original failure. Failed reset keeps GPU access quarantined; close refuses
  without freeing resources. This is not a portable bounded hardware-recovery claim.
- Concurrency fixture INVALID under first-use CUDA LAZY loading: diagnostic
  grid3/4, holder handshake0, competitor saw owner0 after timeout. EAGER alone
  produced actual overlap and -EBUSY. Preload both fixture functions with
  cudaFuncGetAttributes; final default-LAZY qualification passed without changing
  owner logic. Original diagnostic retained.

## Final evidence / completion

PASS: final full payload/guard/reuse matrix, 70,888 healthy GETs including
70,400 in one kernel across16-bit WQE wrap;24 bounds negatives; saturation,
stale/foreign tickets, wrong thread/grid, forced MCST, actual middle-rkey CQ fault,
same-kernel post-reset overwrite monitor and issue-without-wait close.
PASS: separate testing build simulates reset failure; close refuses in0.353ms,
tensor weakrefs remain live, no ordinary GPU completion, disposable process exits.
PASS: real float-tensor example, exact128-element transformed output. Nsight shows
one transform kernel,13,984ns, no extra posting/progress kernel or512B HtoD payload.

Performance current FAIL versus prebuilt CUDA-copy graph: median152.968 vs99.947us
for32x64B (+53.05%);691.000 vs639.069us for32x4KiB (+8.13%). Five alternating
paired blocks; full consuming validation included, not pure copy/serving/CE-engine
attribution. No optimization/adoption claim. Final normal library a328bc98..., exact
hashes and raw compact evidence in RESULTS.md/evidence and SOURCE_SHA256SUMS.

User goal delivered as a fixed-platform thin kernel-first prototype. Remaining
production/general-platform qualification and any performance optimization are
explicitly outside this demonstrated result. All task-owned test processes exited.
GPU5–7 subsequently acquired by other tasks; no further GPU work or cleanup.
Complete remote root /tmp/kernel-local-copy-20261023 on <H20_POD>;
local source/reports only. Original E checkout remains clean at e9325933.

## 2026-10-23 — user: challenge CE performance

New authorization: challenge the existing negative result and optimize if a causal
mechanism is proven. Existing goal was completed; this is its performance follow-up,
not a relabeling of the old microbenchmark. No new public API/PR scope.

Pre-registered hypotheses: (H1) timed byte-validation dominates prior absolute times;
(H2) per-operation system acquire reads of mapped-host Recovery::fault impose
PCIe round trips, with overhead scaling with fragment count; (H3) remaining
cost may be one GPU doorbell per fragment. First isolate H1/H2 before touching
production logic. Diagnostic metrics: request-to-minimal-consumer CUDA-event time,
issue/wait/total device phase times, isolated host-flag vs device-local load time.
Strong controls: both arms captured as CUDA graphs; prebuilt repeated-copy graph,
CUDA batch API if supported; full changing payload/guards checked outside timing.
Seven alternating paired blocks,100 requests/block after10 warmups. Candidate PASS
requires all correctness/error/ownership regressions plus >10% advantage and7/7
favorable pairs versus fastest supported CUDA baseline in a named regime. Negative
regions retained. No promised universal gain or altered failure semantics.

Live GPU7 free4MiB/no compute process at start. Use same task lock and recheck
occupancy; preserve prior source/binaries remotely before variants. New large
artifacts remain remote. Measurements invalid on ownership/config drift.

Professor challenge before promoting speedup: affine modulo13 source mapping can
be decomposed into pitched2D runs, so add both serial/parallel2D graphs. Preserve
pre-2D code/binary and provisional logs. Also retain every timed request's sampled
consumer output and verify it after timing; a later full verification alone is not
consumer-boundary evidence. Source role/credit/error checks remain unchanged.

Pre-register transfer check: compare that structured layout to a seeded Fisher-Yates
permutation, using a GPU index tensor for NIC/SM and prebuilt graph copies that
know the exact same indices in advance (favorable CE control). Apply the same
2D-run compression to both layouts. No changing layouts to erase a negative
result: report both, and constrain claims by index regularity. Strong SM control
is currently oneCTA to match NIC footprint; do not call it globally optimal SM.

## CE challenge delivery and NCCL GIN steering

Hardware first-fault control passed original full regression; keepingGPU sticky
state removes measured~1us/operation Host reads. Warp cooperative API added as
one thin entry: active lanes compute their own pointers, one reserve/SDKsubmit,
all READs signaled and everyCQEchecked, optional oneMCST pergroup. Independent
reviewfound no functional blocker; added sparse{2,7,31}/inactiveleader,zeroactive,
mixedscalar+warp and pending-credit rejection tests. Final scalar+warp regression
passed, including70,400warpGETs inonekernel, reuse/guards, realCQ errors and
resetfailurequarantine. Exactsource/binary identities in challenge/evidence.

Professor challenged provisional2.6x/2.2x vsparallel1DCE: affine13*jmodN is
compressible. Added pitched2D and then nonadjacent positive-stride packing, both
serial/parallel. Structured32/128x4KiB nowCE25.08/48.44us vswarp29.15/108.76us:
negative preserved,0/7wins. Preregisteredrandom seeds20261023/24/25 useGPUindex
tensor while CE knows same indices upfront. BesttestedCEencodingperblock gives
1.63–1.83x across32/128x4KiB,42/42pairedblockwins>10%. Tiny64B favorsSMgather;
scalarAPI didnotgainwarpresults. Full100x128consumedoutputsandallpayload/guards
validatedoutside timing. Conditionalblock-bootstrapCIsrecorded, no globaloptimalCE
orservingclaim. SameGPU/NIC/allocations andwarmregistration for allarms.

Nsightdefaultgraphaggregation wasINVALIDfornodeclassification; reranexplicit
--cuda-graph-trace=node. CEpayload20x32x4096 exactly2,621,440B,320HtoDactivities;
NICnoCUDApayloadcopy. Both20consumers+oneuntimedvalidator. Profiledtimeexcluded
from performance table. Traces/SQLite onlyremote, compact evidence local.

UserasksNCCLGINlikelyfaster. Sourcepin3e46c089/NCCL2.32.3 bundlesGPUNetIO4.1
withrealAggregateRequestsDBskip; defaultwrapperstillthreadGET, completion/MCST
protocol differs. FinalnonaggregateGET mustsubmit; nonMCSTflushonlysnapshots.
No nativespeedclaim. LiveGPU7DMABUF=0,onlylegacy nv_peer_mem; exactNCCLguard
rejects nativeGDAKI. ReusablepreflightreturnsBLOCKED3; checkedsame550libcuda
environment andUUID. Existing72podlibraryhashverified. No gatebypass/proxysubstitute/
hostmodulechanges. MatchedfutureGINbaselinealso requiresHOST_NUMA VMMsource and
VMMHBMtargetsharedbyallcontrols. Detailschallenge/GIN_COMPARISON.md.

Currentcomplete remote root /tmp/kernel-local-copy-challenge-20261023/warp,
initialbaseline retainedseparately; currentheaderandruntime mustmatch. Final
reportchallenge/RESULTS.md, summaries/identities underchallenge/evidence.
NoPR/push ormutationofE/2222/PD. Localonlysource/reports/compact evidence.

## 2026-10-24 DeepInfer scenario source check

CurrentDeepInfer9bd9cf24/vendorDeepEPf6aa38fe rechecked viaauthenticatedGitHub.
KV Q&A saysfinegrainedCEpoor→GDR/NIC,notexplicitGIN; releasedEngramGPUindexed
GET isseparate. TestdefaultBF16512B×512tokens×24entries=12,288reads/6MiBperlayer;
notproductionparams,CPUstorageexplicitflag. MultiQP/context,aggregateddoorbells,
layerwaits explainplausibleadvantage;noexplicitmatchedCEbenchmarkfoundinreviewed
Q&A/test. CEhelperalreadyusesMemcpyBatchAsync;Engramissue/waitcostSMs.
Addedsource-backedscenariosectiontochallenge/GIN_COMPARISON.md;noGPUexperiment
ornativeGINqualificationchanged. ExistinglocalDeepEPhybridbranchleftuntouched.

## 2026-10-24 user go — representative Engram-shaped H2D

Implement new harness only, preserve coreCopy7fc6ffd4 and priorCEresults. Public
DeepEPshape:512B BF16row,512tokens×24headindices=12,288reads/6MiB/layer,2layers.
Singleprocess H2Dsubscenario with524288Hosttablerows/layer; no fullmulti-rank
DeepInfer reproduction. GPUindex table-offset+random sampling, allCEgraphsknow
sameindicesupfront. Hypothesis: moreinflight across1/4/8privateinitiatorQPs
improvesmessage-rate untiltransport/PCIeorCQcostsaturate;SMgather maywinlatency
butinterferewithmatrixcompute. Null: NICdoesnotbeatbestCE/SMoroverlapdoesnot
reduceactualexposedwait. Do not infer overlapfromasyncAPI alone.

SERIAL DAGcopy0→projection0→copy1→projection1;LOOKAHEADcopy0→{projection0,copy1}
→projection1. RealBF16 cuBLAS projectionconsumesgathered[512,6144] andonehot
W[6144,8192],FP32outputallchecked;0/1/4calls/layer initially. Copy-only,
perlayerdataready/compute/exposedwait,andend-to-end reportedseparately. Every
requesttarget/outputpoisonedbeforetiming;fullpayload/guards/projectionoutputs
validatedaftertiming.7alternatingblocks withwarmup;freshsourceepochperblock.
StrongCEserial/parallel1Dandpositivepair2Dgraphs;scalableconstruction rather
thanoldcubicpacking. SM1/8/SMcountCTAs;record actualfootprint. NoGINnumbers:live
registered-prerequisitegate stillapplies. No skippedfailurechecks or backendfallback.

PASS:exactchangingdata/projectionchecks,validownershipandmatchinggraphs,
reproduciblepairedbenefit ifclaimed. FAIL:wrongdata/ornoadvantage. INVALID:
GPUownership/config/mempolicy drift or unsupportedbaseline. Stop onlytask-owned
workers onexternaltimeout,neverGPUreset/driver/modulechanges. Largebuild/graphs
sourceRAMandtracesstayremote;source/reports/compactevidence local.
Topologyinspection:GPU6↔mlx5_bond_3 PIX,NUMA1;GPU7↔NIC0 SYS(oldtest).
Newmatrix usesGPU6/NIC3/CPU80–83 iffree; do NOT transferoldabsolutetimes.

Plan refinement BEFORE execution: replaceonehotWwithdeterministicdenseBF16
integerW[-8,8]nonzero. GoldAisindependentlygeneratedfromindices/epoch onGPU,
GoldC=cuBLAS(GoldA,W)computedoutsidetiming. MeasuredprojectionconsumesNIC/CE
GatherA. RepeatedGEMMsaccumulatebeta1fromzero,soearlybadconsumptioncannotbe
erasedbylaterpasses;comparefullCtoG*GoldC. IntegerBF16inputsandboundedG<=4
makeFP32sums exact(<2^24). Goldisnotamemorytransportfallback. G0usesminimal
consumerwhosevaluesarechecked,notunconsumedcopycompletion. Two layersremain
two layers;G counts controlledprojectionwork,notadditionalmodel layers.

## 2026-10-24 Engram-shaped execution complete

Newharnessengram_scenario/scenario.cu,coreCopy7fc6ffd4unchanged. GPU6/NIC3PIX,
CPU80–83+membind1, BF16512B×12288reads/layer×2layers,256MiBHosttable/layer.
DenseintegerBF16W[6144,8192],actualprojectionconsumesgatherdata;beta1repeat
propagatesearlywrongconsumption,GoldA/GoldCindependenttransport;fullvalidation
everyrequest. Fixed31-rowidentitycollisionBEFOREfullmatrix;wrong+31negprobe
detected~4.16Merrors. Explicit32MiBcuBLASworkspace removedcapturedalloc/free
childgraphunsupportednode10/11. Pilotall60casespassedafterfixes.

Twofullmatrices840case-blocks,5040validatedrequests;~37.16MdirectNICREADs.
Seed24/default8connections5reps/block;seed25/explicit32compute+8copy3reps/block.
Secondconfigbest G0 CE32.956ms,NIC8 2.462ms,SM78 .286ms;G4CE34.961ms,NIC8
look4.921ms,SM78look3.311ms. CurrentprototypebeatstestedCEbutSMisbetter.
Only1/4/8context+QPscalingtested,notisolatedQPvariableornativehardwarepeak.

NIC8G4lookaheaddefault8regressed5.64→5.95ms7/7. Samebinary/seed/placement
8→32connectioncontrolpreservedSERIAL5.642msbutLOOK4.901ms7/7benefit. Node
traces:defaultoneactorstartedaftercompute0(~2.07msdelay),32allstartwithin
~21us;compute0still~2.07vsserial1.59ms,remainingresourcecauseunseparated.
Do not extrapolateindependentactor-gridqueueproblemtoGINoneissuegrid. SMcopy1
nodeoverlapsactualGEMM0for169–204us;CEpathtraceexact36MiB3framepayloadHtoD,
noSMpayloadkernel. Profilertimeexcluded. NativeGIN/GDAKIGPU6gate0DMABUF+legacy
moduleconfirmed;nogaterelaxationorproxybenchmark. NotDeepInfer/fullrankrepro.

READMEandevidenceengram_scenario/;completeremote
/tmp/kernel-local-copy-engram-20261024;largensys/SQLite/buildonlyremote. Initial
fullNUMAsnapshot overwrittenbypilotprofile,documented;subsequentrunsuniquecwd
preserveownmaps. UnrelatedmodelGPU0–3wereactive,nevertouched. NoE/2222/PD/
NCCLsourcechanges,noPR/publication,nohostsdriver/moduleschanges.
