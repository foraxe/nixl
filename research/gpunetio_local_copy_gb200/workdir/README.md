# Kernel 内本机异步 copy

已在 H20 上实现并验证的窄原型：pinned CPU tensor → 本 GPU tensor，
由使用数据的 kernel 直接发起 NIC RDMA READ，等待后在同一 kernel 中消费。
CPU 一次完成注册和同 NIC paired-QP 初始化；每次 copy 不经过 NIXL
descriptor/request/post，也不启动额外的 posting/progress kernel。

```cpp
// 应用 kernel 内，当前原型由 CTA0/thread0 发起。
auto ticket = copy->copy_async(dst, src, bytes);
compute_independent_work();
int status = copy->wait(ticket);
// status == 0 后可消费；同 CTA 其他线程需经过 __syncthreads()。
```

多个不连续块可以由warp0协作，一次提交；每lane直接计算地址，无需描述符表：

```cpp
if (threadIdx.x < 32) {  // 整个warp0共同调用；block为一维
    auto t = copy->copy_async_warp(dst_for_lane, src_for_lane, bytes, active);
    if (threadIdx.x == 0) status = copy->wait(t);
}
__syncthreads();        // status==0时，同CTA线程再消费
```

**更新：**warp版本在三组随机4KiB块索引上比本轮最强测试CUDA-copy编码快1.63–1.83×；
可压成规则二维copy的布局仍是CE更快。标量调用没有自动获得warp收益。
[性能挑战和负对照](challenge/RESULTS.md) · [NCCL GIN实现与当前阻塞](challenge/GIN_COMPARISON.md)。

一次准备真实 tensor：

```python
src = torch.arange(128, dtype=torch.float32, pin_memory=True)
dst = torch.empty(128, dtype=torch.float32, device="cuda:7")
with LocalCopy(src, dst) as copy:
    # 将 copy.device_handle 传给自己的 CUDA kernel；src/dst 仍是普通 tensor。
    launch_my_kernel(copy.device_handle, src, dst)
```

完整可运行例子见 [example.py](example.py) 和 [example.cu](example.cu)。
它在一个 kernel 中执行 copy 和变换，校验全部 128 个 float。
接口的实质是 [local_copy.cuh](local_copy.cuh) 中的copy、warp copy和wait。
[runtime.cu](runtime.cu) 管一次性准备、注册和冷错误/关闭路径；
[local_copy.py](local_copy.py) 只负责 tensor 检查、保活和 C ABI 调用。

## 当前边界

- 单进程、单 owner GPU/NIC；源必须是 contiguous pinned CPU tensor，目标是
  contiguous owner-GPU device tensor。每个注册区域非空且不超过 UINT32_MAX。
- 每个 context 只有一个owner：CTA0/thread0；warp0可协作准备，由lane0提交/wait。
  同时不能由另一grid使用。同一context可连续复用，也可持有多个ticket。
- 最多 32 个未退役 WQE：普通 GET 可有 32 个在途，MCST GET 为 16 个。
  warp整组只需一个MCST/DUMP，因此启用MCST时一组最多31个active lane。
  队列满在提交前返回 `-EAGAIN`，不改 reservation；不在内部等待队列空位。
- warp入口要求完整warp0参与、相同context、1D block；各lane可用active谓词。
  sparse active也允许，lane0可不搬数据但仍负责控制；零active拒绝。
- 一次 GET 不超过 SDK 的 1 GiB WQE 上限。调用者保证各在途目标区间不冲突，
  源写入在 launch 前结束，且在相应 wait 完成前不修改源、不复用目标。
- `wait` 可等待较新的 ticket，一并检查所有此前 CQE；已退役或其他 context
  的 ticket 被拒绝。成功包括 SDK NIC acquire，以及需要时的 MCST/DUMP。
- CQ 错误/设备端等待超时会使 context 失效。错误恢复线程先停止两个 QP，
  kernel 才返回失败；数据无效，不自动重试。正常路径无 CPU 提交或 payload staging；
  仍有一个 CPU 线程约每毫秒检查冷错误标志，并非零 CPU 资源。
- 开始 `close` 后禁止再 launch，Python 会立刻撤下 device handle。冷关闭会
  等待本进程该 GPU 的 CUDA 工作，因此也可能等其他 stream。整个清理由保留的
  worker 执行；调用者约 5 秒内得到完成或拒绝。拒绝时保留 runtime/library/tensor，
  只能重试 close，不能使用旧 device handle。若 NIC 无法停下，GPU wait 不报告可复用。
- 目前依赖固定 E 构建导出的内部 QP/MR C++ 实现，尚非稳定 ABI/上游公共 API。
  尚未支持 peer GPU、跨进程、多 issuer、其他 tensor 分配类型或自动切路。

## 构建和运行

依赖已经编译好的 E QP/MR 底座、DOCA 3.5.0098、CUDA toolkit 和兼容驱动的 PyTorch。
原型直接复用 SDK device GET，没有为此扩展 NIXL 公共 device GET API。
以下路径由环境提供，`E_ROOT` 下应有 `source/` 和 `install/`：

```bash
export E_ROOT=/path/to/pinned-E-build
export DOCA_ROOT=/path/to/doca-prefix
export CUDA_ROOT=/path/to/cuda-toolkit
export CUDA_RUNTIME_LIB=/path/to/cuda-runtime-libs
export PYTHONPATH=/path/to/compatible-torch
export LD_LIBRARY_PATH="$PWD:$E_ROOT/install/lib/x86_64-linux-gnu/plugins:$E_ROOT/install/lib/x86_64-linux-gnu:$DOCA_ROOT/lib/x86_64-linux-gnu:$DOCA_ROOT/../../../usr/lib/x86_64-linux-gnu:$CUDA_RUNTIME_LIB:${LD_LIBRARY_PATH:-}"
bash build.sh
python3 example.py
python3 test_copy.py --gpu 7
./benchmark 7
```

`example.py` 当前固定使用 GPU7；在其他环境先改为自己拥有的空闲设备。
不要在未知归属的 GPU 上直接运行。测试自己的 kernel 时照常传当前 CUDA stream。

独立失败注入构建不会给正常库加入 reset-failure API：

```bash
LOCAL_COPY_TEST_FLAGS=-DLOCAL_COPY_TESTING \
LOCAL_COPY_LIBRARY=liblocal_copy_testing.so bash build.sh
python3 test_quarantine.py
```

该测试必须在一次性子进程内运行：故意保留等待中的 kernel，验证 close 拒绝和
tensor 保活，然后退出该子进程。它验证隔离行为，不宣称恢复了失败的 NIC。

## 结果与恢复入口

详见 [RESULTS.md](RESULTS.md)、[JOURNAL.md](JOURNAL.md) 及 [evidence](evidence)。
当前正确性PASS；性能优势受索引布局和协作方式限制，不做通用CE替代承诺。
完整 build、独立 PyTorch、二进制和 Nsight 报告保留远端，未下载大型副本。

初版负结果保留在RESULTS.md；最新warp结果和强CE对照见challenge/RESULTS.md。
当前远端完整入口：
`<H20_POD>:/tmp/kernel-local-copy-challenge-20261023/warp/`。
执行 `source /tmp/kernel-local-copy-20261023/env.sh`，并将当前目录放在LD_LIBRARY_PATH最前。
Header和runtime库须来自同一构建；原任务src中的旧库作为历史版本保留。
源码本地清单见 `SOURCE_SHA256SUMS`；装载库路径/哈希见
[loaded-libraries.json](evidence/loaded-libraries.json)。
最新运行身份另见[identities.json](challenge/evidence/identities.json)。
