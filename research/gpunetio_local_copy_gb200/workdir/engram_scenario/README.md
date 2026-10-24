# Engram-shaped layered H2D experiment — 2026-10-24

**已执行两个完整矩阵：840个case-block、5,040次完整请求验证。**
在本H20环境的大量512B随机gather中，当前NIC原型明显胜过测试过的CE图；
但并发SM gather更快，并且与投影计算重叠后仍保持优势。
不能据此给NCCL GIN排名；原生GDAKI的注册条件仍不满足。

## 工作负载和验证

- 单进程、GPU6 H20 / NIC `mlx5_bond_3`，PIX，同NUMA1；CPU80–83、Host内存membind1。
  与旧GPU7/NIC0的SYS拓扑不同，不能直接比较之前的绝对时间。
- Driver550.127.08、Linux5.10.134、CUDA编译12.8.93、独立cuBLAS12.6.4.1；
  GPU UUID `REDACTED_UUID`。
- BF16每行256元素=512B；512token×24head=12,288条索引/层、6MiB/层，两层合计12MiB。
  每层Host表524,288行（256MiB）；按head分区随机采样，索引保存在GPU。
  这是公开Engram测试形状的单进程H2D子场景，不是DeepInfer全系统／四rank复现，
  也不代表KV swap-in线上块大小。GPU索引提前完整交给CE建图，控制偏向CE。
- BF16 dense投影 `[512,6144]×[6144,8192]→FP32[512,8192]` 直接消费gather结果。
  W为固定seed的稠密非零整数[-8,8]，每层重复0/1/4次。
  G=4是受控计算预算，不是4个额外模型层；不是线上权重／完整forward。
- 独立GoldA从索引、行身份、layer、epoch生成，GoldC在计时外通过cuBLAS计算。
  实际重复投影从零以beta=1累加，防止后一次计算覆盖前一次错误消费；完整C对比G×GoldC。
  值域保证最大整数累加界5,898,240<2^24，实际检查用精确相等。
- 每次请求前目标毒化为CD、输出清零；每次请求后检查完整payload、两端guard、
  全部投影输出。Host表epoch每block变化。源行前7个base31位编码完整行号，
  其余列为带epoch的hash值；故意错读+31行的负例检测到约416万错误。

核心Copy代码／运行库未改，仍是已资格验证的`7fc6ffd4…`。新程序是独立harness。
NIC的1/4/8个initiator QP通过独立Copy context实现，每层分别运行1/4/8个单warp actor。
它们同时改变PD/MR/controller/issuer数量，因此是context/QP组合扩展，不是孤立的QP变量。
所有CE/SM/NIC case均保持全部16个context冷资源，避免分配差别计入某个arm。
原始JSON的qps_per_layer字段仅对NIC arm有意义；CE/SM没有活动NIC数据路径。

## 受控DAG

SERIAL：`copy0 → projection0 → copy1 → projection1`

LOOKAHEAD：`copy0 → {projection0, copy1} → projection1`

projection1同时依赖copy1与projection0，不能靠并行执行两层计算得到假收益。
记录copy-start/ready、compute-start/done和总时间；这些事件含调度等待，
不把“时间窗口相交”直接当成NIC DMA逐字节重叠证明。
节点级Nsight另查真实actor/GEMM并行和排队。

CE比较serial/parallel 1D图和可扩展的正stride相邻run 2D图；没有执行旧的立方复杂度
非相邻packing，也不称全局最优CE。现代`cudaMemcpyBatchAsync`实查返回36，当前驱动不支持。
SM比较1/8/78个CTA，每CTA128线程，warp以16B/lane coalesced读取mapped Host行。
SM78是本轮最好测试配置，不称全局最优SM；它使用更多SM处理payload，且是独立kernel+graph控制，
并没有在此交付一个可由任意既有kernel直接调用的SM异步copy产品接口。

## 主要结果

每个矩阵60种配置（10arm×3计算预算×2DAG），7个block交替顺序，每case2次warmup。
主矩阵seed20261024每block测5次；复核矩阵seed20261025测3次。
时间为7个block每请求均值的中位数，不是请求p50/p99。
均包括两层搬运和实际消费，排除初始化、建图、Gold和验证。

### 拓展到真实消息数量后的结果

复核矩阵统一设置compute connections32、copy connections8，各家取最快测试配置：

| 两层工作 | CUDA-copy图 | NIC原型 | SM78 CTA |
| --- | ---: | ---: | ---: |
| 仅搬运＋轻消费 | 32.956ms | 2.462ms | 0.286ms |
| 每层1次dense投影＋lookahead | 33.689ms | 2.923ms | 0.942ms |
| 每层4次dense投影＋lookahead | 34.961ms | 4.921ms | 3.311ms |

主矩阵的排序相同：copy-only最好CE33.194ms，NIC8约2.485ms，SM78约0.286ms。
主矩阵SERIAL copy-only的NIC1/4/8为17.662/4.671/2.485ms；
SM1/8/78为9.907/1.349/0.289ms。测到8个context/QP仍在扩展，没有测到NIC硬件上限。
两完整矩阵中的NIC arm共约37,158,912次healthy READ，均通过当前验证。

CE 1D serial约64.980ms，2D约33.208ms；相同payload合并命令后时间近半。
结合SM的高吞吐，这支持当前512B CE路径主要受命令粒度开销限制，
不支持“Host→GPU物理带宽只有0.4GB/s”。
普通CE批处理在新驱动上的结果未知，不能外推到DeepInfer的CUDA13/NCCL栈。

### 异步接口不保证重叠收益：连接数反例

主矩阵默认连接配置下，NIC8/G4 LOOKAHEAD约5.96ms，SERIAL约5.64ms，7/7退化。
节点trace发现第二层一个actor直到第一层GEMM结束后才启动；不是所有actor都运行了3ms。

随后用相同binary、seed、放置和仅两种相同图，显式固定copy connections8，
在新进程中只改变 `CUDA_DEVICE_MAX_CONNECTIONS`：

| compute connections | SERIAL | LOOKAHEAD | 配对SERIAL/LOOKAHEAD |
| --- | ---: | ---: | ---: |
| 8 | 5.641ms | 5.954ms | 0.947×，0/7有利 |
| 32 | 5.642ms | 4.901ms | 1.152×，7/7有利 |

条件block-bootstrap95%区间分别约[0.946,0.950]和[1.146,1.154]。
SERIAL基本不变；第二层暴露等待从约1.06ms降到约3us。
Trace中最后一个actor的启动延迟从约2.07ms降至约21us，32连接时全部在compute0结束前启动。
这给出了连接配置／图调度导致排队长尾的因果证据，**不是简单宣称“一分支等于一硬件队列”**。
[CUDA12.8.1连接设置说明](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#cuda-environment-variables)。

32连接下compute0仍约2.07ms，高于SERIAL约1.59ms；剩余共享资源/缓存/fence影响未进一步拆开。
不能把这个每QP独立actor-grid的调度问题泛化成GIN／所有NIC实现的缺陷。
第二个完整矩阵同时换了seed和连接配置，它验证排序/配置复核，不是索引变量的单因素试验；
上表独立8→32对照才是连接变量实验。

## Trace和正确性质量

- CE2D/G0 trace的图内HtoD活动36,864次、37,748,736B，正好3请求×两层×6MiB；
  图内仅6个消费kernel，没有额外SM搬运kernel。
- SM78/G4图内6个SM copy和24个实际GEMM kernel，无图内CUDA memcpy。
  第二层SM copy与compute0 kernel实测相交约169–204us，支持真实kernel并行。
- NIC8/G4两种DAG、8/32连接均保存node trace；未用被profile扰动的时间替换正常计时。
  只证明actor并行/排队，不声称已测到每段NIC线上的DMA起止。
- 初次cuBLAS capture包含alloc/free节点，child graph返回unsupported。
  节点诊断[10,0,0,11]确认原因，显式32MiB workspace后子图只含kernel，预检60case通过。
- 首版数据每31行重复，被审查指出会放过row+31错误。完整矩阵前修复并执行有害负例；
  早期弱数据pilot不作为最终性能证据。

## 能推广什么

本轮支持：大量不规则512B任务让这个旧驱动的CE命令开销明显暴露；
更多独立issuer有帮助，但调度连接数会改变预取是否真正及时执行。
在本机现有实现/分配上，SM gather是更强的性能竞争者，不能只打败CE就宣布NIC最优。

本轮不支持：原生GIN速度排名、DeepInfer线上复现、模型端到端收益、零SM/零CPU、
所有GPU/驱动/NUMA拓扑的结论。CPU-source静态GPU索引在每次请求相同、cache被warmup；
没有计入实际业务的动态CPU调度、3FS、跨rank、量化scale或kernel内完整融合计算。
两个完整矩阵及配对控制的区间只描述本机条件波动，不覆盖跨机器/热状态误差。

原生GIN在GPU6也实查DMA-BUF=0、仅legacy nv_peer_mem，资格仍BLOCKED；
不能拿本原型的逐CQE/32-credit/多grid结果替代GIN的大窗口、聚合发起和分层完成结果。
后续优先在合格环境测原生Engram/GIN，再决定是否继续维护独立NIC路径。

## 复现、身份和资源

完整远端：`<H20_POD>:/tmp/kernel-local-copy-engram-20261024/`。
核心依赖：`/tmp/kernel-local-copy-challenge-20261023/warp`及原任务独立Torch/cuBLAS目录。
源码`scenario.cu`、`build.sh`、`analyze.py`、`trace_summary.py`。
local evidence只保留JSONL、摘要、哈希和小型CSV；完整nsys-rep/SQLite与build日志在远端。

证据入口：[主矩阵](evidence/full.jsonl) · [主摘要](evidence/summary.json) ·
[第二矩阵](evidence/seed2-connections32.jsonl) · [复核摘要](evidence/summary-seed2.json) ·
[连接对照](evidence/connection-comparison.json) · [8连接timeline](evidence/timeline8.json) ·
[32连接timeline](evidence/timeline32.json) · [数据路径trace](evidence/path-trace-summary.json) ·
[GIN资格](evidence/gin-preflight6.json) · [精确哈希](evidence/identities.sha256)。

```bash
source /tmp/kernel-local-copy-20261023/env.sh
export LOCAL_COPY_ROOT=/tmp/kernel-local-copy-challenge-20261023/warp
export CUBLAS_ROOT=/tmp/kernel-local-copy-20261023/torch12/nvidia/cublas
export LD_LIBRARY_PATH="$LOCAL_COPY_ROOT:$CUBLAS_ROOT/lib:$LD_LIBRARY_PATH"
cd /tmp/kernel-local-copy-engram-20261024/src
bash build.sh
# gpu,tokens,table_rows,projection_N,blocks,repeats,seed,arm_filter,G_filter,mode_filter
CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_MAX_COPY_CONNECTIONS=8 \
numactl --physcpubind=80-83 --membind=1 ./scenario 6 512 524288 8192 7 3 20261025 all -1 all
python3 analyze.py ../seed2-connections32.jsonl > ../summary-seed2.json
```

运行前检查归属并使用任务GPU锁，不中断其他进程。全矩阵缓存60图的一次观测约24GiB GPU、
16GiB Host RSS，主要是图缓存，非单次payload所需；可用过滤参数缩小初始化占用。
旧full运行的NUMA快照被后续profile覆盖，不能将profile地址映射冒充full快照；
命令的membind策略保留，后续runs/子目录各自保留独立快照。
没有修改核心Copy/E/2222/PD、驱动或模块，也没有发布PR。
