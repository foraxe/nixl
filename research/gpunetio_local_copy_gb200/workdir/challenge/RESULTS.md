# Kernel copy：挑战CE后的结果

2026-10-23。**找到有条件优势，并保留了会推翻优势的强基线。**
三组预先固定的随机索引，在32/128个4KiB块上，warp版本相对本轮最快的CUDA-copy
编码加速约1.63–1.83×，42/42配对block有利。规则索引可压缩为少量二维copy时，CE更快。
这不是普遍替代CE，也不是相对NCCL GIN的胜利；原生GIN当前为资格BLOCKED。

## 研究问题和因果路径

用户要求挑战先前的负结果。先前153/691us含逐字节校验，不能直接作为纯搬运时间。
本轮重新计时到真实、轻量的GPU消费者输出，完整验证放到计时外。

### 1. 热错误标志造成可以消除的开销

初版每次copy和wait均从mapped Host Recovery读取错误标志。
1024次依赖acquire-load的device计时，7样本中位：Host约1188ns，本地GPU约135ns。
仅将首错保存在GPU，冷失败时system-release通知原Host恢复线程：

- 单独对照中128×4KiB的scalar总时间约561→414us；issue约491→348us。
- 保存的issue时间随碎片数增长，符合每次操作去掉约1us Host往返的预测。
- CPU控制器原本只读fault、写stopped，没有丢掉已有Host取消功能。
- 首错CAS、QP停止确认、失败后拒绝新提交、隔离/拒绝close均保留。

这个因果对照保留在`baseline/`、`gpu-fault/`远端源码/二进制和
[baseline](evidence/baseline.jsonl) / [gpu-fault](evidence/gpu-fault.jsonl)。
该两份旧诊断harness彼此匹配，不能与后来更强基线直接拼成同一张绝对时间表。

### 2. 并行准备和组提交，而非删检查

SDK3.5的单个GET会调用一次submit，submit内部有两次doorbell写。
scalar逐项做WQE、检查、提交，仍是主要issue成本。
新增`copy_async_warp(dst,src,bytes,active)`：

1. Warp0各lane给出自己的地址/长度；先检查共同context、有效范围和总credit。
2. 一次保留全部槽位，各active lane并行准备signaled READ。
3. 需要时整组追加一个SDK MCST/DUMP，再统一SDK mark-ready/submit。
4. 返回共同ticket，lane0沿用原来的逐CQE wait和错误恢复。

提交在方法返回前发生，不延迟到wait；没有CPU pointer-table、descriptor/request流程。
收益归因于**并行准备、检查摊销及组提交的组合**，没有实验将全部收益单独归因doorbell。
scalar接口仍存在，但不是本轮获胜接口。

## 强基线如何推翻临时结论

初次warp32×4KiB约29us，对1D串行graph约91us、独立节点graph约78us。
独立professor审查指出源索引`13*j mod N`是结构化模式，能用二维pitch表达，
不应只和32/128个1D节点比较。

加入连续目标行的2D合并后，规则128×4KiB的CE降至约53us，击败NIC约107us。
又加入**非相邻目标行**的正stride二维packing：只合并源、目标都成正等差序列的行，
每行恰好覆盖一次，serial/parallel graph都测。没有CPU打包数据或更改布局。
它是确定性greedy，不声称最优cover。

最终每个case从所有可用CUDA编码中取最快者；并且每个配对block也对最快CUDA arm比较。
随机索引不是为抹掉负结果：先登记规则/随机两种布局和种子，再运行并保留两类结果。
完整索引列表写入每份JSONL。

## 最终测量方法

同一H20 GPU7、mlx5_bond_0/port1/GID3、驱动550.127.08、DOCA3.5.0098、CUDA编译12.8.93。
分配、注册、QP及graph预建均在稳态计时外；各arm用同一份Host/GPU分配和同一索引。
CPU的完整索引提前交给CE图构建；NIC/SM支付GPU index tensor读取成本，控制偏向CE。
每个块间改变Host payload epoch；源在执行期间不变。

每case：7个交替顺序block，各10次warmup、100次测量。所有arm都为预建graph：

表中耗时是7个block的每请求均值之中位数，不是每请求延迟分布的p50/p99。

- GPU-local sticky-fault的scalar NIC；warp NIC。
- 一维copy的serial/parallel graph。
- 二维相邻run的serial/parallel graph。
- 二维非相邻packing的serial/parallel graph。
- 一个CTA的mapped-host SM gather控制（与NIC均1CTA/128threads，不称全局最优SM实现）。

`cudaMemcpyBatchAsync`实际返回36：`API call is not supported in the installed CUDA driver`。
没有把不支持静默当成较慢结果；本轮不包含更新驱动后的batch API表现。

CUDA event覆盖graph请求到轻量消费kernel结束；另记录Host wall time、issue/wait/kernel阶段。
每个请求都保存128个实际消费结果，100×128个输出全部在计时后校验。
另一个不计时的GPU kernel检查完整payload、hole和guard；检查调用次数，不能靠上轮数据通过。
没有在计时区间做昂贵逐字节校验。

### 主种子结果（耗时中位数，us）

| 布局 | 形状 | Warp NIC | 最快测试CUDA-copy | 判定 |
| --- | --- | ---: | ---: | --- |
| 规则affine | 32×4KiB | 29.147 | 25.078 | CE胜，NIC0/7 |
| 规则affine | 128×4KiB | 108.757 | 48.442 | CE胜，NIC0/7 |
| 随机seed20261023 | 32×4KiB | 28.580 | 51.016 | NIC7/7，配对中位1.783× |
| 随机seed20261023 | 128×4KiB | 107.537 | 187.173 | NIC7/7，配对中位1.738× |

这四项最快CUDA编码均为二维packing的serial graph。
随机布局seed20261024/20261025复核32和128块，配对中位加速1.777/1.671×和1.832/1.626×，
四项也都7/7有利。因此三个种子、两个形状共42/42有利，全部超过10%预先门槛。
不是从所有可能索引中选出有利seed；仅代表这三个明确保存的排列。

[summary.json](evidence/summary.json)包含每个arm的中位数、每block最快CUDA对照和
固定seed bootstrap95%区间。主seed两项区间约[1.777,1.801]、[1.699,1.746]。
这是同一进程配对block的条件不确定性；未覆盖跨机器、固件、NUMA、热状态或部署差异。

负区域也保留：单块CUDA-copy通常更快；64B碎片时1CTA直接SM gather明显更快。
获胜于CE不等于获胜于所有本机资源路径。

## 真实引擎/数据路径验证

在相同随机32×4KiB、10warmup+10测量下，另做非性能排名用Nsight node trace。
必须指定`--cuda-graph-trace=node`；初次默认graph聚合trace看不到内部节点，
标为INVALID引擎分类，不拿“没看到kernel”当证据。

[node-trace-summary.json](evidence/node-trace-summary.json)：

- 最快CE图：320个payload HtoD活动，合计2,621,440B，恰好20×32×4096B；
  只有20个轻量消费者kernel和1个计时外validator，没有额外SM搬运kernel。
- Warp NIC：同样20消费者+1validator，HtoD活动仅33,648B初始化（含GPU索引和context），
  没有这些payload的CUDA copy活动；GET/提交/等待都在应用kernel内。

Trace用来分类路径，不能把被profiler扰动的时间替换未profile计时。
未用物理端口计数区分NIC内部loopback与外部路线，也未声称NIC零SM/零CPU资源。

## 正确性与独立检查

[完整回归](evidence/warp-final-qualification.log)通过：

- 原scalar70,888次healthy GET，以及原先的边界、满队列、ticket、错误线程/grid、
  MCST、真实中间rkey错误、首错保留、QP停止、晚到写检查和未wait就close。
- Warp的changing epoch/reuse额外70,622次GET（含单kernel70,400次跨16位回绕）；
  还包括基础partial组、forced-MCST（126CQE）、真实warp CQ错误及失败后拒绝。
- Warp32+MCST越容量、单lane越界、非完整warp、混合context均提交前拒绝。
- Sparse lanes{2,7,31}且lane0 inactive；zero-active拒绝；旧scalar在途时warp满组拒绝、
  较小warp组加入后一起安全完成。完整payload/guard验证。
- [独立测试构建](evidence/warp-quarantine.log)模拟reset失败：close拒绝，tensor保活，
  GPU不报告普通完成；一次性进程退出。约0.052ms拒绝不是成功硬件恢复证明。

源代码审查验证了warp lane排序、credit、MCST、统一context、owner与发布顺序；
professor审查实际找到了2D反例并执行，因此最终结论已收窄。
没有放宽原来的失败/生命周期约定；生产通用资格仍未建立。

## 可复用模型与边界

可以用两个数量指导选择：碎片数N，以及CE能够合并出的copy命令数K。

- CE大致支付K份命令开销和payload成本；规则pitch使K远小于N，CE仍有明显优势。
- 当前NIC以最多32个READ一组，支付组准备/提交、N份CQ检查和payload成本。
  随机布局的K仍大，组提交更容易摊薄开销。
- 极小块直接SM load/store可能同时胜过两者；需要计算干扰证据才能判断业务选择。

本轮新增的是这个固定平台上的因果/边界证据，不是新的DMA硬件、一般CE替代定理，
也不是相对GIN更快的结论。GIN详见 [GIN_COMPARISON.md](GIN_COMPARISON.md)。
当前NCCL支持路径被实查注册条件挡住；没有proxy冒充或gate bypass。

## 重现与精确产物

完整远端：`<H20_POD>:/tmp/kernel-local-copy-challenge-20261023/`。
`baseline/`、`gpu-fault/`为因果控制，`warp/`是当前完整源码/库/程序。
其环境复用原任务`/tmp/kernel-local-copy-20261023/env.sh`，独立Torch未复制。
远端warp目录已有flat的`perf_challenge.cu`/`build_perf.sh`；从本地目录新部署时，
把`challenge/perf_challenge.cu`和`challenge/build.sh`复制到同一运行根目录，后者命名为build_perf.sh。

```bash
source /tmp/kernel-local-copy-20261023/env.sh
export LOCAL_COPY_ROOT=/tmp/kernel-local-copy-challenge-20261023/warp
export LD_LIBRARY_PATH="$LOCAL_COPY_ROOT:$LD_LIBRARY_PATH"
cd "$LOCAL_COPY_ROOT"
bash build.sh
bash build_perf.sh
python3 test_copy.py --gpu 7
# gpu, paired blocks, repetitions, independent-compute iterations,
# count filter, bytes filter, arm filter, layout seed(0=affine)
./perf_challenge 7 7 100 0 0 0 all 0
./perf_challenge 7 7 100 0 0 0 all 20261023
./perf_challenge 7 7 100 0 32 4096 all 20261024
./perf_challenge 7 7 100 0 128 4096 all 20261024
./perf_challenge 7 7 100 0 32 4096 all 20261025
./perf_challenge 7 7 100 0 128 4096 all 20261025
```

调用前重新检查设备归属；不改/杀其他任务。计时文件与source/binary哈希见
[identities.json](evidence/identities.json)；normal库`7fc6ffd4…`，benchmark程序`97cb73ab…`。
本地`analyze.py`对六份最终JSONL生成summary，并校验排列、7配对block、零mismatch。

```bash
# 在本地local_copy/challenge目录，或同结构远端目录
python3 analyze.py evidence/{affine-packed,random-packed,seed2-32,seed2-128,seed3-32,seed3-128}.jsonl > evidence/summary.json
```

源/报告/紧凑证据留本地；二进制、完整trace/SQLite和build日志只留远端。
初版源/库完整保存在baseline及原任务src，不能把历史a328bc98库与新header混用。
