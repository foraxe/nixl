# Kernel local copy — 2026-10-23

**历史初版报告。**当前源码已包含GPU-local错误标志和warp组提交；最新性能、
强二维CE反例和NCCL GIN边界见 [challenge/RESULTS.md](challenge/RESULTS.md)。
下文数字只对应原始a328bc98构建，不赋给新源码。
初版源码清单保留在 [initial-SOURCE_SHA256SUMS](evidence/initial-SOURCE_SHA256SUMS)。

## 判定

**目标原型完成，正确性 PASS；当前性能优势 FAIL。**

实现用户修正后的 kernel-first 范围，不是 ZIP 原先的 host-request A0。
普通 tensor 一次准备；应用 kernel 直接 `copy_async(dst, src, bytes)`、等待并消费。
没有扩展 NIXL 公共 API，也没有修改 E、#2222、PD 或其他 device-owner 工作树。
没有提交/推送/发布 PR。

## 固定环境与实现

- Pod `<H20_POD>`，GPU7 H20，UUID `REDACTED_UUID`。
- 驱动 `550.127.08`；NIC `mlx5_bond_0`，port1/GID3，固件 `32.39.3920`。
- 编译 CUDA `12.8.93`，DOCA headers/libraries `3.5.0098`。
- PyTorch `2.7.1+cu126` 独立安装于远端任务目录，实际加载 libcudart `12.6.77`；
  系统 `2.11.0+cu130` 无法在当前驱动初始化，未覆盖系统安装。
- 复用 E `e932593300366073dfd5f3333ef1d42f6abfbc0a` 的低层 QP/MR 类型及构建，
  构造同一 NIC 的 requester/responder QP 并互连，不创建 NIXL agent 或 TCP 握手。
- 直接调用 SDK nonblocking GET 和逐 CQE error-reporting poll，显式 GPU SM doorbell。
  payload 不经 CPU staging；不是新增硬件/PTX 指令。
- 正常库 SHA256 `a328bc983ba5f92c9a482860f675a2e497845ecf9eae27d2961a4b570dc8c914`。
  E plugin SHA256 `0245b98e08fb8c3f3b7a40d427ec37de98984e4e65fce6fd5c78553b1f3f5b08`。
  [loader readback](evidence/loaded-libraries.json) 包含路径与全部相关库哈希；
  move-runtime 中实际 DOCA 库与 SDK 3.5.0098 原文件逐哈希相同。

## 实测验证

最终命令 `CUDA_MODULE_LOADING=LAZY python3 test_copy.py --gpu 7`，退出0。
[完整结果](evidence/qualification-delivery.log)。

| 检查 | 结果与界限 |
| --- | --- |
| 连续/碎片、changing host epoch、同一注册重复使用 | 70,888 次 healthy GET；全部 payload 与目标护栏在 GPU 内精确校验。连续 1B/4KiB/1MiB，37 碎片，窗口1/8/16/32。 |
| WQE 回绕 | 单次 kernel 中 32×2200=70,400 GET，posted=retired=70,400，跨过16位计数回绕。 |
| 独立计算与直接消费 | issue 后执行独立运算，wait 后消费；消费线程通过 CTA barrier 同步。未量化实际 DMA/计算重叠比例。 |
| 源/目标越界、零字节、极大长度 | 24 次拒绝，reservation 不变；没有提交无效范围。 |
| 满队列、stale/foreign ticket | 满32 WQE后拒绝且 reservation 不变，排空后可再用；已退役及不同 cookie ticket 拒绝。 |
| 错误线程/并发 grid | thread1 被拒；两个真正并发 grid 的非 owner 返回 -EBUSY，owner 成功。 |
| MCST | H20正常为mcst=0；强制mcst=1的额外路径 posted=retired=222，payload/护栏通过。它是额外路径测试，不是另一种硬件资格认证。 |
| 真实 CQ 错误 | 三个已发 READ 中间一个使用错误 rkey，返回实际 -EIO；同一失败原因保留，后续提交拒绝。两个 QP reset 后才允许 kernel 返回失败，随后同 kernel 覆写目标并监测，未见晚到写入。 |
| 发起后不 wait 就关闭 | 冷关闭完成 CUDA actor、停止 QP，然后释放注册；退出0。取消不保证 payload 完整。 |
| reset 失败隔离 | 测试专用构建模拟 reset 失败：close约0.353ms内拒绝，tensor仍保活，GPU不返回普通完成；一次性子进程退出。不是实际硬件 reset 失败复现。 |

Nsight [kernel summary](evidence/kernel-summary.csv)：tensor 示例只有 **1次 transform kernel**，
13,984ns；没有独立 posting/progress kernel。
trace HtoD memcpy仅为初始化4096/12288B队列、320B SDK descriptor及104B copy context；
没有512B tensor payload的CUDA HtoD copy。输出512B通过D2H读取并逐元素与期望比较。
完整 trace/SQLite留在远端 `example-trace.nsys-rep` / `example-trace.sqlite`。

## 并发测试的无效配置与修正

初次 LAZY 加载第二个 kernel 同步 context，owner 等待超时后才执行竞争者。
记录为 **INVALID 并发测试**，没有改 owner 实现来配合预期：
[diagnostic](evidence/owner-lazy-diagnostic.log) 中 grid3/4不同，但竞争者看到owner=0、
holder握手=0，证明没有实际竞争。只改EAGER加载后看到owner=85、竞争者grid86、
握手=1、结果-EBUSY。测试在launch前用cudaFuncGetAttributes预加载两个函数后，
默认LAZY完整回归也通过。

## 匹配微基准：当前没有优势

同一源码/注册/拓扑，32独立碎片、相同索引映射、相同独立计算和全量GPU消费校验；
NIC为一个消费kernel；CUDA基线将固定地址的cudaMemcpyAsync节点和相同消费kernel
提前录为graph，避免每次CPU逐块提交。初始化/graph建立不计入稳态。
5个配对block交替顺序，各10次warmup、100次计时；CUDA event涵盖搬运到校验完成，
另保留host wall time。全部校验通过。[原始行](evidence/benchmark-delivery.jsonl)。

| 每碎片大小 | NIC median us | CUDA graph median us | NIC耗时变化 |
| --- | ---: | ---: | ---: |
| 64B ×32 | 152.968 | 99.947 | +53.05% |
| 4KiB ×32 | 691.000 | 639.069 | +8.13% |

这是带逐字节校验的组件测试，不是优化后带宽、纯搬运延迟或业务收益。
CUDA graph使用固定已知地址；未比较动态GPU索引、batched CUDA API、SM gather或GIN。
未做CE执行引擎专项归因，不把cudaMemcpy API名称当作独占CE证明。
不据此宣称NIC普遍更慢，也不宣称发现了优势区间。

## 安全与推广边界

本轮证据证明上述版本/配置中的行为。错误返回前停止DMA依赖DOCA QP reset的控制面
语义；实际CQ错误及晚到写监测通过，但有限测试不是所有故障/固件的证明。
没有找到覆盖任意 exported-GPU-QP 的公开、有界 reset/abort 保证。
NVIDIA [RDMA手册](https://docs.nvidia.com/rdma-aware-networks-programming-user-manual-1-7.pdf)
和 [Linux UMR修正](https://github.com/torvalds/linux/commit/d97505baea64d93538b16baf14ce7b8c1fbad746)
说明错误处理和reset可能丢弃CQE的关系，不能外推为所有SDK故障的完成时限。
因此目前是固定平台实验原语，不是通过通用生产资格认证的透明替代品。
若停止失败，保留资源并拒绝close；不假装错误等于安全释放。

设备操作只资格验证了CTA0/thread0的单issuer和同CTA消费。
同一进程该GPU的其他stream可能影响冷关闭等待。硬件内部loopback还是外部端口路径
未通过物理计数器分类；没有跨节点、零CPU、普遍优于CE的结论。

## 恢复

完整远端：`<H20_POD>:/tmp/kernel-local-copy-20261023/`，`env.sh`记录有效依赖。
`src/`是完整源码/so/smoke/benchmark；`torch12/`是独立5.1GiB运行环境；
build日志、资格验证、benchmark JSONL、Nsight trace和SQLite均在任务根目录。
本地只保留源码、报告、哈希和紧凑证据。运行结束后未残留本任务的测试进程；
之后GPU5–7出现其他任务，停止GPU工作，没有清理/中断它们。
