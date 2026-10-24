# GPUNetIO 本机数据搬运：CTO 方向与最小架构

2026-10-22。状态：源码调研后的候选设计，不是性能结果或实现承诺。该方向与 950 DRAM 池、SuperMem 独立推进；未执行设备测试、修改用户代码或发起 PR。

## 结论

值得推进，但准确定位是 **NIXL 中的可选本机 NIC 搬运路径**，优先解决 Host DRAM → GPU HBM 的细粒度、非连续、多块搬运。不将它描述为普遍优于 Copy Engine，也不将“本机”自动解释为“无需网络硬件”。

接口应先复用 NIXL 既有异步传输与 backend 选择，而不是先造另一套 memcpy 框架。第一阶段让现有 GPUNETIO backend 获得受支持的 local endpoint 能力；只有业务需要直接从已运行的 GPU kernel 发起请求时，再接通真正的 NIXL GPU device API。

## 1. 机会依据与关键更正

DeepInfer 官方 Q&A 表示，细粒度 KV block 的 Host 共享内存 → HBM swap-in 选择 GDR/RDMA 网卡搬运，因为 CE 在该类复制上表现不佳。但完整 KV 系统未包含在开源版本，因此其具体 swap-in 实现、是否采用 GIN、性能分界不能由公开代码确认。

公开可追的另一条路径是 Engram：DeepInfer → vendored DeepEP EngramBuffer → NCCL GIN get。固定版本显式要求 GDAKI。NCCL 的 GDAKI 使用 GPUNetIO，并已有同 NIC、本地 rank 的成对 QP loopback 实现。因此“直接 GPUNetIO vs NCCL GIN/GDAKI”主要比较集成层、注册模型、提交/进展与资源占用，不是两种截然不同的 DMA 硬件。

注册及 window 创建位于初始化阶段，fetch 复用它们。直接 GPUNetIO 仍须注册 Host/GPU 内存、管理权限与生命周期；不能把移除 NCCL 的 window 对象等同于免注册，也不能将初始化成本按每次访问累加制造优势。

来源：
- [DeepInfer KV swap-in 说明](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L155-L157)
- [DeepEP GIN get](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh#L109-L129)
- [DeepEP 指定 GDAKI](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/comm/context.cpp#L124-L145)
- [NCCL 同 NIC 的 GPUNetIO loopback QP](https://github.com/NVIDIA/nccl/blob/3e46c089f4ae4732722a419eb6096aead7e47a30/src/transport/net_ib/gdaki/gin_host_gdaki.cc#L874-L945)

## 2. 第一个产品场景

选择长期注册的 Host DRAM 中多个非连续 KV block，将其搬入一个或多个 HBM 目标范围。每次请求由源/目标位置和长度列表描述；数据内容与模型无关。明确 source/destination memory type、设备、NUMA/PCIe 路由、registration epoch、操作方向及完成条件。

为什么先选它：它同时匹配用户的 KV 经验、DeepInfer 的明确需求和 NIC 批量 WQE 发起的潜在价值。它不要求立即实现 Engram 的完整索引/分布式表语义，也不要求改 attention kernel。

首版不承诺同卡连续 HBM→HBM 更快，不把 NVLink/C2C 路径存在的机器与普通 PCIe 机器混为一谈。reverse direction 与 peer HBM 是可单独资格验证的组合，不能自动继承 Host→HBM 结论。

## 3. 两个阶段，不混淆“GPU 发起”

### A：NIXL host API 下的 local GPUNETIO 路径

沿用 host prepare/post，由一到少量 GPU posting kernel 批量发 RDMA 请求。CPU 仍负责现有控制与可能的轮询。需要 local metadata、connect(self)/本地 endpoint 状态、源目标注册关联和 completion/visibility 合同；不能只把 supportsLocal 改成 true。

当前上游 NIXL GPUNETIO supportsLocal=false。当前上游 READ 和单 QP WRITE 每批最多 512 描述符，使用单 CTA、512 threads；较大请求可能引发多次 CPU kernel launch。READ 的 posting CTA 等完成，另有进展 kernel；因此有可优化的资源/调度面，但不能说完全不占 SM 或 CPU。

当前 NIXL 并没有独立 CE plugin；UCX 的本机 self-endpoint 也不等于必然由 CE 执行。沿用 READ/WRITE、self-agent 与 optionalArgs.backends 选择路径，不另造模型专用 gather API。如果产品要求 NIXL 内显式可保证的 CUDA-copy 选项，需要另行评审窄本机 backend，不能当成现成功能。

此阶段即可验证“用户在 CE 不合适时选择另一条搬运路径”的主要产品价值。使用既有显式 backend 选择，初期不做未经数据校准的自动切路。

### B：真正 GPU-callable 的 NIXL GPUNETIO device API

应用的已有 GPU kernel 直接发请求，不经每批 host launch。当前 NIXL device API 公开操作有 put、atomicAdd、status、getPtr，没有公开 device get；如果要求 pull 语义，新增 get 是单独的公开 API 设计范围，不能只改一个分发分支。当前上游 NIXL device API 分发为 UCX_DIRECT/PROXY；新增 GPUNETIO 分支及 memory-view 准备属于额外工作，不能把阶段 A 当作已完成阶段 B。

只有阶段 A 或具体 GPU-resident 场景证实提交路径值得消除后，才决定 B 的优先级。若计算中动态生成索引，B 与 Engram 类稀疏读取更相关；若传输表本来由 CPU 生成，先增加整套 device API 不一定有收益。

来源：[NIXL pinned main](https://github.com/ai-dynamo/nixl/tree/d64231659c8194760b60d163b514a3eba1ac22ee)、[GPUNETIO kernels](https://github.com/ai-dynamo/nixl/blob/d64231659c8194760b60d163b514a3eba1ac22ee/src/plugins/gpunetio/gpunetio_kernels.cu#L458-L559)。这些是上游基线，不代表用户自己的 E 路线工作树状态。

## 4. 真正可能产生优势的地方

1. 多个碎片合成少量发起工作，降低每块 CPU/CUDA 提交成本。
2. 在足够吞吐下减少 SM 发起/等待占用，给推理计算留资源。
3. 为本机、不需要 NCCL 集体初始化/对称窗口约束的用法提供较轻的独立注册与 endpoint 管理。
4. 当 CE 或相应链路已有负载时，使用可用的 NIC 搬运资源改善并发执行。

以上都是待证假设。DeepEP 已经批量 get、聚合 doorbell，其 CE 工具还使用 cudaMemcpyBatchAsync 与 PreferOverlapWithCompute，不能把它当逐块无优化基线；它的 Engram 发起 kernel 按 SM 布置 CTA，因此资源占用可比较，但更少 CTA 也可能降低搬运吞吐。

性能拆开记：初始化/注册；每次 prepare/post；实际传输；目标可消费完成；与计算并发后的整体收益。一次性初始化收益不能伪装稳态收益，通信 kernel 更短也不能自动推导推理更快。

## 5. 硬件与正确性合同

- 使用选定 NIC、GPU、驱动、GPUNetIO 版本下受支持的 local QP 路径。NCCL 存在该设计是可行性依据，不证明 NIXL 当前插件对所有设备已经支持。
- 内存范围、访问权限、registration key/epoch 必须有效，源与目标保持存活到真实终态。初版排除重叠源目标，不把 memcpy 语义扩成 memmove。
- 区分 accepted、CQ completion 和目标 GPU 可消费。遵循平台要求的 ordering/flush/visibility，不以 CQE 单独替代所有可见性条件。
- 超时/取消不释放仍可能被 DMA 访问的 buffer。显式失败不能自动在同一目标上并发执行 CE fallback；先关闭旧访问，或采用独立安全目标且明确结果发布者。
- 声明 SM/进展 kernel、CPU proxy 和 NIC 资源。NIC_HANDLER_AUTO 可能选 CPU proxy，调用 device API 本身不是 CPU-free 证据。
- DOCA 的 GPU-triggered DPU DMA API 是另一项能力，不把其 DPU 要求套到本路线的 GPUNetIO RDMA，也不混称 GPU CE。

## 6. 交给专项团队的判别问题，而非重复派实验

最小合格比较是同一 topology、内存分配、块分布、目标可见性和注册复用条件下：
- 合理批处理/graph 或可用优化的 CUDA-copy 基线，而非只比较朴素逐块提交。若声称比较的是硬件 CE，须确认实际执行引擎；cudaMemcpy API 名称本身不构成证明。
- 现有可用 SM copy/gather 路径，用来区分省 CPU 提交和真换搬运资源。
- direct GPUNetIO local 路径，单独记录发起与进展资源。
- 比 GIN 时使用已热身、长期注册的 GIN/GDAKI，且相同数据布局和完成边界；若无法构造等价语义，不给出直接速度排名。

不要求先跑全排列。先确定一个业务代表性的碎片 Host→HBM 场景，以及它与计算并发时的结果。总耗时、CPU 提交、SM 占用、NIC 吞吐及最终计算影响都应能解释。裸带宽不是唯一目标。

## 7. 推进与停止条件

值得进入实现评审：local path 受支持、接口/生命周期闭合，且对代表性场景有清晰的可检验收益来源。

值得作为 NIXL 通用能力交付：收益不能只来自删掉安全同步、把注册从计时中单边排除，或使用明显较弱的 CE 基线；需要至少一个可重复、可解释的优势区域及不会误用的能力边界。

若仅赢初始化、不赢稳态，则定位为集成/初始化成本优化；若仅赢特定 NIC/PCIe/块分布，则做受限能力，不宣传通用替代；若输给合理 CE/SM 路径，则保留负结果，不为原语概念强造需求。

当前建议立项边界：先评审阶段 A 的 local endpoint 与批量提交改动，阶段 B 保留为独立后续决策。CE、SM 和 NIC 是可选资源路径，不是必须淘汰其中一种。

## 8. 最小实现评审边界

阶段 A 优先单 agent/process、一个通信 GPU 与 NIC、长期注册的普通 Host DRAM → 本 GPU HBM、RDMA READ。仅在真实支持及 visibility 合同成立时再开放反向 WRITE。peer GPU、跨进程、VMM/multicast 和多 NIC 不自动纳入首版。

代码接点是 GPUNETIO backend 的 supportsLocal/connect(self)/loadLocalMD、verbs 层本地 paired-QP，以及现有请求准备和完成管理。复用 MR，不重复注册 self-target view；先把局部能力接通，不把 ring 重写、multi-QP 和全部既有优化捆成一个补丁。

保持现有 NIXL READ/WRITE + self-agent + backend 选择语义；显式请求不支持的 local GPUNETIO 返回不支持，不偷偷走 CPU proxy 或另一条路径。可选 fallback 只在未提交前选择；已提交后跨路径重试须先满足访问终止与目标安全条件。

研究依据与逐文件改动边界见同包 DEEPINFER_TRACE.md、LOCAL_COPY_FEASIBILITY.md、NIXL_PRIMITIVE.md。本结论没有重新审阅用户暂缓的 #2222，也未检查用户 E 路线当前工作树；实施前应由实际负责人对齐已有实现，避免重复开发。


## 2026-10-23 实现对齐更新

此前“E 分支未核实”的状态已解除：实际干净 checkout 为 experiment/gpunetio-byte-qp-20261022，HEAD e932593300366073dfd5f3333ef1d42f6abfbc0a。MR、GPU READ、QP/CQ 及部分提交跟踪可复用；self paired-QP、self-peer、非拥有型 loadLocalMD 仍缺。实现顺序先解决失败排空与安全终态，再开放本机 READ。具体交接与逐文件证据见 IMPLEMENTATION_HANDOFF_20261023.md、E_BRANCH_ALIGNMENT_20261023.md。当前仍是静态设计/审阅，不是新代码或硬件验证；不重开 #2222。
