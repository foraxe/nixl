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
on H20. vLLM testing of a fresh build is still open. Earlier measurements used a separately built shim,
so these build results do not independently reproduce their performance.
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

## Further validation

`cross_node_smoke.py` contains separate synthetic owner and reader roles;
`real_checkpoint_owner_mmap.py`, `real_checkpoint_read_smoke.py`, the adapter
and `launch_pd_role.py` show the real-checkpoint/vLLM study path. They require
PyTorch with CUDA, Triton and the pinned vLLM integration branch in addition
to this runtime. Exchange the generated owner ready file securely and keep it
private: it is operational metadata. These are not yet a verified
fresh-build 1P1D reproduction. Validate resource retirement and
one owner generation before treating any result as equivalent to the study.
