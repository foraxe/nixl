# NCCL GIN：实现对照与实测边界

2026-10-23。用户判断GIN可能更快；这是合理候选，**目前没有原生GIN/GDAKI的匹配实测排名**。
不把此前GIN-proxy的PUT/信号实验当作本次Host→HBM GET结果。

## 固定源码

本地干净 `../../nccl-gin-upstream`：`3e46c089f4ae4732722a419eb6096aead7e47a30`，
NCCL2.32.3；bundled GPUNetIO **4.1.0**。本次直接Copy用DOCA GPUNetIO3.5.0098。
这里只描述这个pin，不把缺失的实验选项外推到其他NCCL包/后续版本。

| 机制 | 该pin的GIN/GDAKI实现 | 对当前Copy的意义 |
| --- | --- | --- |
| 本机self | 为本rank创建main/responder QP并互连 | 不是新DMA硬件；可直接作为同进程同NIC候选。 |
| 热路径 | kernel `gin.get`，复用window/key/context | 不应把每次注册、CPU提交成本算到GIN头上。 |
| 聚合 | AggregateRequests映射SKIP_DB_RINGING；bundled4.1确实跳过doorbell | SDK3.5对应路径只是未实现的扩展注释；新版本不能按旧行为评估。 |
| 实际提交 | 非MCST的flushAsync仅snapshot reservation；最后一个非aggregate GET应提交前面的请求 | 不能将所有GET都设Aggregate后假设flush一定ring。MCST flush另有实际WQE提交。 |
| Warp | wrapper在coop.thread_rank()==0发默认THREAD GET；该pin没有公开WarpGet选项 | CoopWarp不等于每lane分别GET；bundled SDK有warp primitive，不能混为wrapper已经用了它。 |
| 可见性 | 记录last_issued_get，需要时flush追加一次MCST | 注册复用和一致性工作已经存在，应计入相同完成边界。 |
| 完成 | wait到snapshot最后ticket并做acquire；可用timeout/abort路径 | 当前Copy逐个检查所有signaled CQE；这是协议/成本差异，不能单凭函数数量判GIN更快或不安全。 |
| 其他资源 | exclusive contexts、THREAD/CTA/GPU sharing及其他受支持NIC handler | 强基线应测试合理资源模式，不能故意选CPU proxy或差的配置。 |

精确接点（相对NCCL源码根）：

- `src/include/nccl_device/gin/gdaki/gin_gdaki.h:44–48`：聚合flag映射；
  `:261–283`：get；`:308–335`：flush；`:338–377`：wait。
- `src/transport/net_ib/gdaki/doca-gpunetio/include/common/doca_gpunetio_verbs_def.h:49–51`：4.1.0。
- 同目录 `include/device/doca_gpunetio_dev_verbs_qp.cuh:524–527`：真正跳过doorbell。
- `src/transport/net_ib/gdaki/gin_host_gdaki.cc:874–945`：self paired QP。

独立源码检查支持“GIN有更成熟的聚合/完成机制”，**没有证明它比当前warp版本快**。
warp版本同时改变并行准备、检查摊销和提交次数，不能将收益全部归因doorbell。

## 当前H20：BLOCKED，而非性能FAIL

[实查](evidence/gin-prerequisite.json) 在同一有效libcuda环境中：

- GPU7，`CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED=0`，查询返回CUDA_SUCCESS。
- 驱动API12040，实际库`/usr/lib/x86_64-linux-gnu/libcuda.so.550.127.08`。
- kernel `5.10.134-16.3.al8.x86_64`。
- 只有legacy `nv_peer_mem`，没有 `nvidia_peermem`。

可复用只读脚本`gin_preflight.py --gpu 7`也重新运行并输出
[BLOCKED及GPU UUID](evidence/gin-preflight.json)，按约定退出3；不会分配tensor或启动kernel。

`src/transport/net_ib/gin.cc:31–43`明确拒绝该组合；GDAKI不接受legacy nv_peer_mem。
因此当前没有启动原生GIN计时，也没有绕过检查或改动主机模块/驱动。
另一现有H20构建所在72pod也只见legacy模块，未把它臆测成可用替代环境。
[官方要求](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)
也区分GDAKI与CPU proxy的注册条件；GET不是可用proxy冒充的同口径结果。

现有可复用NCCL构建保留在：
`<H20_POD>:/tmp/nccl-gin.7DG12W/build/lib/libnccl.so.2.32.3`。
本次重新读取SHA256：
`ed3169a0f2d4fc1f4064f0dba463ff721cd8358c7c0f1f9de508ccb7845268b9`。
源码快照无.git；身份来自本地pin和实际库哈希，不伪称远端gitHEAD。

## 公平比较还需要统一分配

该pin的公开`ncclCommWindowRegister`要求可retain的cuMem/VMM allocation handle，
不是任意`cudaMallocHost`/`cudaMalloc` tensor都能直接注册。
`src/dev_runtime.cc:1181–1208`、`src/include/alloc.h:414`可见保留句柄及类型检查。
Host段使用`CU_MEM_LOCATION_TYPE_HOST_NUMA`；见
[官方Host buffer限制](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)。

获得合格环境后，原生对照应：

1. 使用一份HOST_NUMA VMM源和一份NCCL/VMM HBM目标；GIN、直接Copy、CUDA控制都用
   完全相同分配、NIC、索引和下游消费校验。先资格验证这些分配，不冒用普通tensor的旧证据。
2. 一次注册/建连；分别记录冷初始化。热循环用一个rank/self peer、一个exclusive context
   起步，再测试合理的协作/资源共享方案；不要为了对照而限制GIN最好的受支持路径。
3. 非末尾GET聚合，末尾GET实际提交，flushAsync建立完成边界，timeout-capable wait后消费。
   不能通过删掉可见性、错误语义或把CPU proxy混入得到排名。
4. 先记录GIN类型及实际doorbell/transport，再给时间；比较健康消费完成，也单列失败处理差异。

结论：**GIN应该保留为主要强基线；当前只能给实现分析及准确的资格阻塞，不能宣称谁更快。**

## 2026-10-24：DeepInfer具体场景

重新核对当前DeepInfer `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`，
vendor/DeepEP gitlink仍是 `f6aa38fe06cd4a0489e7730365d4e0114970a2f3`。

- [Q&A:155–157](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L155-L157)
  描述3FS→Host共享内存→逐层HBM，swap-in与forward重叠。它说CE对细粒度KV块表现不好、
  用GDR/RDMA NIC，没有明确该KV实现是GIN；完整cache manager未开放，不能杜撰CE基线。
- 明确公开的GIN路径是独立Engram：GPU index tensor→按peer/context聚合GET→分层wait hook。
  [fetch](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh)
  有32项pending批、doorbell阈值、多个QP/context，最后一批实际提交，之后flush_async。
- [公开测试默认shape](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/tests/engram/test_engram.py)：
  BF16 hidden256→每消息512B；512token×24entry→12,288次随机读取/层、6MiB/层；
  默认两层合计24,576次、12MiB。这是测试默认值，不是线上配置；须显式开CPU-storage才是H2D测试。
- 因而合理机制假设是：不可压缩的小随机块使CE命令成本难摊薄，GPU侧索引处理、批量提交和
  足够在途READ帮助NIC利用带宽，分层预取再隐藏剩余时间。源码证明调度结构，没有公开对照
  数字证明各因素分别贡献多少；不能泛称GIN总比CE快。
- 同仓库的[CE helper](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/csrc/kernels/driver/driver.cpp#L58-L64)
  已用cudaMemcpyBatchAsync和PreferOverlapWithCompute；不能假定其CE全部是朴素逐块调用。
  Engram仍启动GPU issue/wait kernel，不能把KV Q&A的“不占SM”套用到该公开实现。

对本项目的启发：32/128×4KiB、规则2D可压缩/固定graph只是边界控制，不是Engram复现。
后续若评价DeepInfer相关价值，应增加512B级、每层上万条GPU索引读取、层间issue/wait及计算重叠，
并在合格环境比较原生GIN、现代batched CE及SM gather；目前没有执行该规模的新实验。
