# Experimental GPUNETIO GET reproducer

This is a **legacy research snapshot**, based on NIXL `8585eee0` and DOCA
3.1.0105. It is not NIXL main, a supported Device API, or a recommended
production deployment. The GPU GET path uses private backend structures;
NIXL [#2544](https://github.com/ai-dynamo/nixl/pull/2544) is a separate API
proposal and does not provide this implementation. Do not use this snapshot
alongside a different NIXL core or plugin.

The checked-in M1, shim and Python sources reproduce the research stack's
functionality. The new build described below has passed single-node synthetic
M1A and ABI-3 row-READ tests, and a cross-node ABI-3 synthetic row-READ test,
on H20. A fresh build was also used for a real H20 TP4/TP4 1P1D vLLM run:
two sets of eight greedy prompts matched the local-table control, and a
32-request C4 performance run completed. One sequential pair is insufficient
to claim a stable speedup or tail improvement.
Do not include production weights, endpoints, credentials, logs or metadata
in a public report.

## Build

Requirements: Linux x86-64, NVIDIA H20/SM90 with a working driver, CUDA 12.8,
DOCA GPUNetIO 3.1.0105 development packages, libibverbs/mlx5, C++20, CMake,
Ninja, pkg-config, Python 3 and Meson 1.9.0. Run in an isolated prefix.
The GPUNetIO device nodes and compatible RDMA device must be visible. The
configuration needs the dependencies used by NIXL, including asio and
tomlplusplus; Meson may download them through the checked-in wrap files.

```bash
export WORK="$(mktemp -d)"
curl -fL https://github.com/abseil/abseil-cpp/releases/download/20250814.1/abseil-cpp-20250814.1.tar.gz -o "$WORK/abseil.tar.gz"
echo '1692f77d1739bacf3f94337188b78583cf09bab7e420d2dc6c5605a4f86785a1  '"$WORK/abseil.tar.gz" | sha256sum -c -
tar xf "$WORK/abseil.tar.gz" -C "$WORK"
cmake -S "$WORK/abseil-cpp-20250814.1" -B "$WORK/abseil-build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_INSTALL_PREFIX="$WORK/absl" -DABSL_PROPAGATE_CXX_STD=ON \
  -DBUILD_SHARED_LIBS=ON
cmake --build "$WORK/abseil-build" -j4
cmake --install "$WORK/abseil-build"
export PKG_CONFIG_PATH="$WORK/absl/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$WORK/absl/lib:${LD_LIBRARY_PATH:-}"

# From the root of this research branch:
curl -fL https://codeload.github.com/taskflow/taskflow/tar.gz/refs/tags/v3.10.0 -o "$WORK/taskflow.tar.gz"
echo 'fe86765da417f6ceaa2d232ffac70c9afaeb3dc0816337d39a7c93e39c2dee0b  '"$WORK/taskflow.tar.gz" | sha256sum -c -
test ! -e subprojects/taskflow
mkdir -p subprojects/taskflow
tar xf "$WORK/taskflow.tar.gz" -C subprojects/taskflow --strip-components=1 taskflow-3.10.0/taskflow
cp subprojects/packagefiles/taskflow/meson.build subprojects/taskflow/
meson setup "$WORK/build" . --buildtype=release \
  -Denable_plugins=GPUNETIO -Dgpunetio_device_api=enabled \
  -Dgpunetio_device_api_profile=legacy-doca31 -Dnixl_cuda_arch_list=90 \
  -Dbuild_tests=false -Dbuild_examples=false -Dwith_trace=false
ninja -C "$WORK/build" -j4 src/core/libnixl.so \
  src/plugins/gpunetio/libplugin_GPUNETIO.so \
  src/utils/device/libnixl_device_allocator_cuda.so \
  research/engram_shared_rdma_m1/shared_engram_m1
python3 research/engram_shared_rdma_m2/build_against_m1.py --build "$WORK/build"
```

`meson setup` requires access to its pinned fallback sources. Taskflow's
v3.10.0 tarball SHA-256 is
`fe86765da417f6ceaa2d232ffac70c9afaeb3dc0816337d39a7c93e39c2dee0b`.
The validated build reused an existing checkout whose headers were compared
against that tarball; the optional Prometheus exporter was unavailable and
was not needed by this test.

## Single-node smoke

Use two assigned idle GPUs and set the device-specific settings for the
machine. The owner and first reader share GPU 0 or use distinct GPUs; the
example below uses owner 0 and two readers on 1. Select unused ports.

```bash
export CUDA_VISIBLE_DEVICES="<two-assigned-gpu-ids>"
export GPUNETIO_DEVICE_API_RDMA="<rdma-device>"
export GPUNETIO_DEVICE_API_GID_INDEX="<gid-index>"
export GPUNETIO_DEVICE_API_OOB_INTERFACE="<network-interface>"
export NIXL_PLUGIN_DIR="$WORK/build/src/plugins/gpunetio"
export LD_LIBRARY_PATH="$WORK/absl/lib:$WORK/build/src/core:$WORK/build/src/infra:$WORK/build/src/utils/common:$WORK/build/src/utils/device:${LD_LIBRARY_PATH:-}"
python3 research/engram_shared_rdma_m1/run_m1a.py \
  --binary "$WORK/build/research/engram_shared_rdma_m1/shared_engram_m1" \
  --out "$WORK/m1a" --owner-gpu 0 --reader-gpus 1 1 \
  --owner-port 56131 --reader-ports 56132 56133 56134 \
  --long-loops 12 --long-pause-ms 100
python3 research/vllm_engram_repro/rebuild_shim_smoke.py \
  --library "$WORK/build/research/engram_shared_rdma_m2/libengram_native.so" \
  --owner-port 56231 --reader-port 56232
```

The second test copies results to the CPU for byte verification and checks
the device completion count. It is correctness-only; do not include that
copy in performance measurements. The M1 harness validates its device results
and reader reconnect/lifecycle. Both use synthetic data.

## Cross-node ABI-3 smoke

Build the same source and dependencies on both hosts, set the same environment
variables as above, and select one assigned GPU on each host. Set `$SHIM` to
the locally built `libengram_native.so` on each host. Run the owner:

```bash
python3 research/vllm_engram_repro/rebuild_cross_node.py owner \
  --library "$SHIM" --port 56431 --metadata "$WORK/owner.private.json" \
  --stop "$WORK/owner.stop"
```

After `OWNER_READY`, transfer the metadata file over a trusted channel to the
reader host. Its contents include remote memory keys. On the reader:

```bash
python3 research/vllm_engram_repro/rebuild_cross_node.py reader \
  --library "$SHIM" --port 56432 --metadata "$WORK/owner.private.json"
```

After the reader exits successfully and has retired its work, create
`$WORK/owner.stop` on the owner and wait for its clean exit. This test uses
manual coordination, not the owner lease protocol; do not stop the owner
while a reader may still access its memory. It validates 32 packed rows and
completion count on synthetic data, with a host copy for checking results.

## Real model and 1P1D study

The tested setup had two H20 nodes, four GPUs each for P and D, and a fifth P
GPU for the owner. The owner backed both packed tables with one pinned Host
allocation (188.83 GiB for the tested checkpoint). Ensure sufficient Host RAM
and locked-memory allowance; use one fully matching checkpoint on both nodes.
Inputs, user output, RDMA keys and checkpoint bytes are not included here.

Use the vLLM integration at [fc092117f3](https://github.com/foraxe/vllm/tree/fc092117f3)
or its successor, with CUDA PyTorch, Triton and the dependencies for the
existing P/D KV connector installed on both endpoints. The tested workers
used vLLM `0.33.0rc1.dev185+g65faf4efa`, PyTorch 2.14.0+cu130, eager mode,
Marlin MXFP8 linear and Triton-unfused MXFP4 MoE. The owner used system
PyTorch 2.11.0+cu130. A compatible CUDA 12 runtime is required for this shim
even when the vLLM workers use CUDA 13. These are experimental, mixed-version
settings; other software/hardware combinations have not been qualified.

On **both** nodes, export the paths below from the corresponding local build;
use matching NIXL core and plugin libraries, and set the RDMA device, GID,
interface and `VLLM_HOST_IP` for the local topology. Unset any inherited
`ENGRAM_SHARED_*` variables. Do not put real addresses or keys in logs that
will be published.

```bash
export NIXL_SOURCE="$PWD"  # root of this research checkout
export SHIM="$WORK/build/research/engram_shared_rdma_m2/libengram_native.so"
export NIXL_PLUGIN_DIR="$WORK/build/src/plugins/gpunetio"
export PYTHONPATH="$NIXL_SOURCE/research/engram_shared_rdma_m2:$NIXL_SOURCE/research/vllm_engram_repro:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$WORK/absl/lib:$WORK/build/src/core:$WORK/build/src/infra:$WORK/build/src/utils/device:$WORK/build/src/utils/common:$WORK/build/src/utils/serdes:$WORK/build/src/utils/stream:${LD_LIBRARY_PATH:-}"
export GPUNETIO_DEVICE_API_RDMA="<rdma-device>"
export GPUNETIO_DEVICE_API_GID_INDEX="<gid-index>"
export GPUNETIO_DEVICE_API_OOB_INTERFACE="<network-interface>"
export VLLM_HOST_IP="<this-node-address>"
export NCCL_SOCKET_IFNAME="<network-interface>"
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
```

Start the owner on the P node using a dedicated assigned GPU. Use fresh paths
and an unused RDMA/control port, and set the advertised address to a trusted
network interface. The owner control protocol is not authenticated and must be
restricted to a trusted environment. Keep the generated metadata file private.

```bash
CUDA_VISIBLE_DEVICES="<owner-gpu-id>" python3 research/vllm_engram_repro/real_checkpoint_owner_mmap.py \
  --checkpoint "$MODEL" --library "$SHIM" --owner-gpu 0 \
  --rdma-port "$OWNER_RDMA_PORT" --control-host 0.0.0.0 \
  --advertise-host "$P_HOST" --control-port "$OWNER_CONTROL_PORT" \
  --ready-file "$WORK/owner.private.json" --stop-file "$WORK/owner.stop" \
  --max-pinned-gib 190 --storage-layout row264
```

After `owner ready`, securely copy its ready file to the D node. On each node,
activate the vLLM Python environment and run the corresponding command below
with a unique reader-port base; with one READ lane the four local GPU ranks use
base, base+4, base+8 and base+12. `$MODEL` must point to the same
checkpoint content on both nodes, and `$P_GPU_IDS` / `$D_GPU_IDS` each list
four assigned GPUs. The helper uses the vLLM example's existing P/D KV
connector; the Engram reads use this branch's GPUNetIO shim.

```bash
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024,pinned_max_cached_size_mb:1024
# P node
python3 research/vllm_engram_repro/launch_pd_role.py --role prefill --mode remote \
  --model "$MODEL" --host 0.0.0.0 --port "$P_HTTP_PORT" --devices "$P_GPU_IDS" --tp 4 \
  --ready "$WORK/owner.private.json" --library "$SHIM" \
  --reader-port-base "$P_READER_PORT_BASE" --rdma-device "$GPUNETIO_DEVICE_API_RDMA" \
  --extra=--disable-custom-all-reduce --extra=--kernel-config \
  --extra='{"moe_backend":"triton_unfused","linear_backend_per_quant":{"mxfp8":"marlin"}}'
# D node
python3 research/vllm_engram_repro/launch_pd_role.py --role decode --mode remote \
  --model "$MODEL" --host 0.0.0.0 --port "$D_HTTP_PORT" --devices "$D_GPU_IDS" --tp 4 \
  --ready "$WORK/owner.private.json" --library "$SHIM" \
  --reader-port-base "$D_READER_PORT_BASE" --rdma-device "$GPUNETIO_DEVICE_API_RDMA" \
  --extra=--disable-custom-all-reduce --extra=--kernel-config \
  --extra='{"moe_backend":"triton_unfused","linear_backend_per_quant":{"mxfp8":"marlin"}}'
```

The helper currently fixes GID 3 and OOB interface `eth0` when launching a
remote worker; change those explicit values for another topology. For a local
table control, replace `--mode remote` with `--mode native` and omit the ready,
library, reader-port and RDMA options. Keep every other model, P/D, allocator,
kernel and workload setting the same.

Start the vLLM checkout's P/D proxy using the endpoints and KV bootstrap port
appropriate for the deployment. For example, in the vLLM environment:

```bash
python "$VLLM_ROOT/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py" \
  --host 0.0.0.0 --port "$PROXY_PORT" \
  --prefill "http://$P_HOST:$P_HTTP_PORT" "$P_KV_BOOTSTRAP_PORT" \
  --decode "http://$D_HOST:$D_HTTP_PORT"
```

Against its `$PROXY_URL`, run the same two sets of synthetic greedy prompts on
each arm, saving the local and remote result files under different names:

```bash
python3 research/vllm_engram_repro/greedy_canary.py --url "$PROXY_URL" \
  --model engram-model --cache-salt-prefix run-a --out "$WORK/greedy-a.json"
python3 research/vllm_engram_repro/greedy_canary.py --url "$PROXY_URL" \
  --model engram-model --cache-salt-prefix run-b --out "$WORK/greedy-b.json"
vllm bench serve --backend openai --base-url "$PROXY_URL" --endpoint /v1/completions \
  --model engram-model --tokenizer "$MODEL" --dataset-name random \
  --random-input-len 512 --random-output-len 64 --num-prompts 32 \
  --max-concurrency 4 --num-warmups 4 --seed 2026 --ignore-eos \
  --save-result --save-detailed --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99
```

Compare output hashes and prompt/output counts for both local and remote
canaries, for example:

```bash
python3 - "$NATIVE_CANARY" "$REMOTE_CANARY" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    native = json.load(f)
with open(sys.argv[2]) as f:
    remote = json.load(f)
assert len(native) == len(remote) == 8
assert native == remote, "greedy outputs or token counts differ"
print("8/8 greedy outputs and token counts match")
PY
```

Also compare each arm's two repeat files. Retain complete benchmark samples rather than filtering warm or slow
requests. The detailed output includes generated text and must remain private.
Eight greedy prompts and 32 measured requests are only a smoke/performance
study, not full accuracy, production p99, or multi-run speedup evidence. A
565-token native-only logprob comparison still showed differences after
position 512 even when the tested greedy outputs matched.

Stop both vLLM roles gracefully with a nonzero shutdown timeout, confirm all
reader leases are released, then create `$WORK/owner.stop` and wait for owner
exit before freeing its backing. Forced termination, cancellation and recovery
remain open work. `cross_node_smoke.py` and `real_checkpoint_read_smoke.py`
provide additional synthetic and real-row checks.
