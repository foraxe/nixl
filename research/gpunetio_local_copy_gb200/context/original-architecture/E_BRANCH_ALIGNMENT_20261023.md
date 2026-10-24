# GPUNetIO 独立 E → 本机 Host DRAM READ 对齐（2026-10-23）

结论：找到了真实 E checkout；可复用 MR、GPU verbs READ、QP/CQ 管理，但本机同 agent READ 尚未接通。最重要的缺口是安全终态：当前 E 错误返回与 DMA 停止没有绑定；公版 core 的元数据过期和 release 析构路径也不能由 backend 内部引用计数补救。

## 身份与证据边界

- 实际目录：`<LOCAL_HOME>/projects/wip_and_temp/h20_test/gpunetio_wip/nixl-byte-qp-20261022`（下文 E 文件均相对它）。
- 当前分支 `experiment/gpunetio-byte-qp-20261022`，HEAD `e932593300366073dfd5f3333ef1d42f6abfbc0a`；本地别名分支 `gpunetio-byte-balanced-write` 同 SHA。开始、结束 `git status` 均干净，无未提交修改。
- E 基线 `7f9d5b530ad8ed01113a4662a2beb2fdb610d89f`，实现 `3cead9a573f2860e3c3ec9f31c86598e3b5ad742`；HEAD 只追加复现文档。base→HEAD 7 文件 +149/-8，未改 core、CUDA kernel 或完成协议。
- 本地对象库已有公版 `d64231659c8194760b60d163b514a3eba1ac22ee`，通过 `git show` 只读核对；不是 checkout 或重开公共 main。它与 E 的共同祖先为上述 7f9 基线。7f9→d642 的 GPUNetIO 目录无差异；本文涉及的 backend_engine.h、transfer_request.h、nixl_memory_section.cpp 也与 E 相同。core 行号变动单独标明。
- 历史证据已更新：E RESULTS.md:70–85 记载 131 功能进程、8 种 READ/WRITE×DRAM/VRAM endpoint smoke、zero-final 负例对齐；:61–66 有 same-splits/QP0 控制，:27–45 有匹配性能，:112 仍为 cross-node 未核实。这是本地报告记载，未回读远端原始产物、未复现实测，也不证明同 agent DRAM→owner HBM READ。

## reuse / adjust / missing / unknown

| 项目 | 判定 | 当前实现证据与最小差距 |
| --- | --- | --- |
| 同 agent requester/responder paired QP、self-peer 注册 | **missing** | backend.h:63 返回 supportsLocal=false；backend.cpp:1428 connect 是 no-op。:925–1027 仅按 remote_agent 建一套 data/notif（及 WRITE striping）QP；aux.h:235–261 无 self responder 字段。:1441–1511 依赖 OOB remote handshake。没有本地配对互连和 self map 发布，不能把已有两进程 loopback 当成同 agent 支持。 |
| 初始化顺序 | **reuse + adjust** | E core nixl_agent.cpp:409–417、公版 d642 同行，先查询 supportsLocal，再 connect(self)。必须让能力在 connect 前稳定可知，不能用 self QP 已存在决定 supportsLocal，否则永远不进初始化。connect(self) 应先完成成对 QP / self route，再发布成功；失败完整回滚。现状不是已有循环依赖，而是新增能力时需避免的陷阱。 |
| loadLocalMD / MR 所有权 | **missing + reuse** | GPUNETIO 无 override，backend_engine.h:209–211 默认失败。可复用 registerMem 的单次真实 MR 和 imported mr(addr,size,rkey)：backend.cpp:1544–1563、1612–1621；verbs.cpp:665–681 的 imported wrapper 不持 ibmr、不注销真实 MR。应新建独立 public MD 包装同一注册区域，避免再 ibv_reg_mr，且真实 MR 只由 private MD 销毁。不要把 private 指针直接当 public 或复制其 unique_ptr。 |
| 元数据注销 | **adjust** | infra/nixl_memory_section.cpp:286–304 先 registerMem 后 loadLocalMD，失败回滚；:567–590 无条件 unloadMD(self wrapper)，core:561–568 随后 deregisterMem(private)。因此 self wrapper 必须独立且非拥有真实注册。仍需在已提交 DMA 结束前阻止真实 MR 注销；单有包装不构成 in-flight 保护。 |
| DRAM→owner HBM READ 资格门 | **reuse + missing** | backend.cpp:595 支持 DRAM/VRAM；:1518–1551 已有 Host ibv_reg_mr、owner VRAM dmabuf / peermem fallback。prepXfer:1652–1666 仅验 op/count/length；:1694–1699 仅限制 foreign-GPU attached stream，未限定 self READ 的 source=DRAM、destination=owner VRAM、同 agent、正确 devId/存储类型。private MD aux.h:197–201 仅有 devId、MR、字符串，需保存足以作资格判定的类型/owner 信息。owner HBM 对应 READ 的 local destination，Host 对应 remote/self source。 |
| 实际 SDK / doorbell | **reuse / unknown** | meson.build:16 要求 doca-gpunetio>=3.5。backend.cpp:951–957 指定 `DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB`；verbs.cpp:336–343 按该模式 export；kernels.cu:56–84 使用 prepare_read→mark_wqes_ready→submit→poll_cq_at。不是 CPU doorbell 或新 device API。历史 E RESULTS.md:27 记载 DOCA3.5.0098/CUDA12.8，但本次未检查加载中的库或 SDK 头实现，因此不能外推任意 3.5+ SDK 的精确完成/可见性语义。 |
| READ 正常完成 / GPU 可见性 | **reuse + unknown/adjust** | kernels.cu:69–90 对 need_mcst 追加 DUMP、poll final CQE，GPU release 后置 in_use；:283–338 progress 再 poll SQ，写 completed；backend.cpp:2066–2077 扫全部已提交位置后 retire。可复用链路，但 release fence 本身不能替代 NIC→GPU visibility 证明；需对锁定 SDK 核实 poll/DUMP 的 acquire/visibility 合约及 owner GPU consumer 的 stream 交接。READ 内 poll 失败只 printf（:84–86），仍发布 in_use；不能假定第二次 poll 一定重报已消费的错误。 |
| 部分提交 | **reuse + adjust** | backend.cpp:1957–1977 逐 chunk launch，记录 postedCount，若先前已有提交则返回 IN_PROG；checkXfer:2066–2077 等已提交完成再返回 postStatus。这部分保留了正常部分失败的等待。可是函数入口 :1933–1939 的 device/global-error 失败可直接返回负码；必须检查旧执行或其他已提交 DMA 是否还存活。 |
| 全局/CQE 错误终态 | **missing（安全门）** | kernels.cu:291–299 首个 CQE error 置 exit_flag 并退出 progress；completion_status.h:30–31 将标志直接映射负码；backend.cpp:2050–2052 在检查 request 的所有完成之前返回负码。多 chunk / 多 request 时存在“一个 CQ 错，另一些 DMA 未证已停，调用者却已收到终态”的静态可达时序。未实测该时序；代码没有安全排空证明。 |
| releaseXferReq | **adjust，core seam** | E core:1363–1392，d642:1384–1419：只在 IN_PROG 分支认真处理 releaseReqH 的返回；一旦 check 变负码，直接 delete。transfer_request.h:66–69 析构调用 releaseReqH 但忽略结果。backend.cpp:2110–2120 仅 dataQpCount>1/global-error 时同步内部 post streams，不包含 attached stream，且 stream 完成不是 NIC DMA 全部排空证明；:2132–2142 对非 IN_PROG 的 check 错误仍释放 request。不能把 release SUCCESS 当 safe-to-reuse。 |
| 元数据过期绕过 backend | **adjust，core seam** | E core getXferStatus:1304–1312、公版 d642:1324–1332，在 remoteSection.expired 时直接 NOT_FOUND，未 check/drain backend，且该分支不更新缓存 status；post/repost 的过期门 E:1193–1198（d642:1197–1202）也在检查既有 IN_PROG 之前。调用者看见错误可覆盖内存，backend refcount 挽救不了。这与“backend 负码被 core 缓存终态”是两条不同路径。self invalidate 被 E core:1793–1795 拒绝，但 self deregister 的 MR 生命周期仍须处理。 |

## 最小下一步代码边界（本次未实施）

1. 在 GPUNETIO backend.h/.cpp / aux.h 中增加专用 self paired-QP 建连与销毁、稳定 supportsLocal、self route 和非拥有真实 MR 的 loadLocalMD 包装。复用 verbs QP/MR 构造与 connect_verbs_qp（utils.cpp:296–305 已允许 remote READ）。不要走对自己的 TCP/OOB 握手，不重做 E 字节分片，也不接 A+C/B/device API 栈。
2. 先只开放 self、DRAM source→owner VRAM destination、READ、明确的内部 stream。资格拒绝必须发生在任何提交之前；MR 保持至该执行安全终止，且不得允许调用者此时覆写。
3. 把 backend 执行状态明确分成 RUNNING、FAILURE_DRAINING、SAFE_SUCCESS/SAFE_ERROR：部分提交、全局错误均进入排空；排空中仍返回 IN_PROG 或同步完成排空后才返回负终态。安全门须涵盖已 launch 尚未 posting 的 kernel、所有已提交 SQ/NIC 工作、CQ 错误与失败 QP，不能仅等 CUDA stream。无法证明 quiescent 时不得报告可释放/复用。
4. 保留一个有界 core 合约修正：getXferStatus/post 的 metadata-expiry 分支对已开始执行先走 backend 收尾；releaseXferReq 必须显式检查释放结果，不能依赖忽略返回值的析构。未发布过任何工作的 stale prepared handle 仍可立即拒绝。这是 buffer 安全的 cross-layer 接点，不能声称只加 backend 引用计数即可完成。
5. 在固定 SDK 上静态确认 READ CQ acquire、DUMP/need_mcst、Host CPU 发布前置条件和 owner GPU 消费交接后再确定 kernel 改动；不提前添加未经需要证明的 fence。后续验证计划属于主任务调度，本轮不启动。


## 验证范围

本轮为实际 E checkout 的静态只读审查；未修改用户代码、执行测试、访问远端或启动硬件任务。历史功能/性能数字未独立复现。SDK 实现和实际加载库、故障排空、GPU 消费可见性仍需专项证据。
