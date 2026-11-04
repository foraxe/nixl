# GPUNETIO device GET component reproduction

This opt-in bundle compares the existing DOCA 3.1 Engram ABI-3 runtime with
an adapter using the NIXL DOCA 3.5 GPU GET API. It uses synthetic data only.
The 264-byte packed row and selection logic live here, on the application
side. NIXL receives registered views and generic offsets and lengths.

The 3.5 batch path checks each READ CQE in parallel and advances the CQ only
after the batch has been checked. It currently calls the internal DOCA
3.5.0098 `doca_priv_gpu_dev_verbs_poll_one_cq_device_at` helper; this is not a
public stability guarantee. A public advancing-poll alternative measured in a
separate 1-QP comparison was substantially slower. The current evidence is
for H20 where `need_mcst=false`; hardware requiring MCST is not qualified.

## Requirements

One assigned idle H20 (SM90), Linux x86-64, GPUNETIO-capable NIC and working
GPUDirect configuration, CUDA 12.8, Python with CUDA PyTorch, NumPy and
Triton. The example measures one host and one GPU; it does not prove
cross-node or application-serving performance. Use separate environments for
the two DOCA/NIXL builds; do not mix their core libraries or plugins in one
process.

Build the 3.1 ABI-3 control from the public research snapshot
[`b5322e46`](https://github.com/foraxe/nixl/tree/b5322e46f6e686ff16bdb3afb8a1a940a64e1163/research/vllm_engram_repro)
using its documented DOCA 3.1.0105 instructions. Build the NIXL 3.5 candidate
at commit `0630a0859751440316f3ee7143b44d4b09d39f88` with
`-Dgpunetio_device_api=enabled` and DOCA 3.5.0098, then build this
bundle's adapter against that matching candidate source/build:

```bash
export NIXL_SOURCE=/path/to/candidate/nixl
export NIXL_BUILD=/path/to/candidate/build
export DOCA_INCLUDE=/path/to/doca-3.5/include
export CUDA_HOME=/path/to/cuda-12.8
export OUTPUT=/path/to/libengram_native_nixl35.so
./build_engram_native_nixl35.sh
```

For an installed NIXL, set `NIXL_PREFIX` and `NIXL_LIBDIR` instead of
`NIXL_SOURCE` and `NIXL_BUILD`. Configure `NIXL_PLUGIN_DIR` and the dynamic
loader for the **matching** core/plugin/DOCA build in each arm. The test
requires 17 consecutive free OOB ports starting at `LANE_PORT` and two free
ports starting at `VLLM_PORT`.

## Connect the adapter to the vLLM Engram reader

The standalone vLLM research branch at
[`eaf7a188`](https://github.com/foraxe/vllm/tree/eaf7a18899a7a8da5ca89eccb8bcf7dffbbb964d)
contains an ABI-3 owner, reader and `verify_rows.py`. After building the
adapter above, use its path as `ENGRAM_NIXL_SHIM` for both owner and reader,
and follow that branch's
[owner and reader instructions](https://github.com/foraxe/vllm/blob/eaf7a18899a7a8da5ca89eccb8bcf7dffbbb964d/examples/disaggregated/engram_rdma/README.md).
Use the same DOCA 3.5 candidate core, GPUNETIO plugin and SDK libraries in each
process. The owner and reader may use separate hosts, GPUs and network devices;
provide their own ready-file paths and non-overlapping ports. Protect the ready
file because it contains access information.

```bash
export ENGRAM_NIXL_SHIM="$OUTPUT"
# Set NIXL_PLUGIN_DIRECTORY and NIXL_LIBRARY_DIRECTORY to the matching 3.5
# build, plus MODEL, OWNER_READY_FILE, and TEST_READER_PORT_BASE as described
# in the linked vLLM instructions. Start its owner first.
export NIXL_PLUGIN_DIR="$NIXL_PLUGIN_DIRECTORY"
export LD_LIBRARY_PATH="$NIXL_LIBRARY_DIRECTORY:${LD_LIBRARY_PATH:-}"
python examples/disaggregated/engram_rdma/verify_rows.py \
  --model "$MODEL" --ready "$OWNER_READY_FILE" \
  --library "$ENGRAM_NIXL_SHIM" --reader-port-base "$TEST_READER_PORT_BASE"
```

The 3.5 adapter and unmodified owner/provider/verifier from this vLLM commit
passed an actual-checkpoint, same-host reader check: both Engram layers,
padding, 514-query boundary and 16 alternating-stream reads. The application
modules ran in an existing vLLM source overlay, so that result does not qualify
a complete vLLM build or model-serving run. Stop readers and confirm lease
release before stopping the owner, as described in the vLLM instructions.

## Run one arm

Use the same NIC, GPU, software stack, inputs and workload in three separate
process groups: 3.1 pre, 3.5 candidate, 3.1 post. Configure the matching
runtime libraries and set the following for each arm:

```bash
export CUDA_VISIBLE_DEVICES='<physical-gpu>'
export GPUNETIO_DEVICE_API_RDMA='<rdma-device>'
export GPUNETIO_DEVICE_API_OOB_INTERFACE='<oob-interface>'
export GPUNETIO_DEVICE_API_GID_INDEX='<gid-index>'
export NIXL_PLUGIN_DIR='<matching-plugin-directory>'
export SHIM='<matching-abi3-shared-library>'
export RESULT_DIR='<fresh-results-directory>'
export PYTHONPATH="$(cd ../engram_shared_rdma_m2 && pwd):${PYTHONPATH:-}"
export ARM='<pre|new|post>'
export LANE_PORT=59910
export VLLM_PORT=48240
mkdir -p "$RESULT_DIR"
python3 make_ids.py --out "$RESULT_DIR/ids.npz"
for lanes in 16 1; do
  python3 bench_read_lanes.py --library "$SHIM" \
    --out "$RESULT_DIR/${ARM}-${lanes}qp.json" --gpu 0 --port-base "$LANE_PORT" \
    --samples 7 --windows 512 --lanes "$lanes" --pending 0 --layouts single \
    --pipeline 2 --polling parallel --ids-file "$RESULT_DIR/ids.npz" \
    --ids-key rows --cuda-graph
 done
python3 benchmark_vllm_context_onegpu.py --library "$SHIM" \
  --module ./vllm_runtime.py --port "$VLLM_PORT" \
  --out "$RESULT_DIR/${ARM}-vllm.json"
```

The ID generator uses seed 20261104 and produces 6144 rows, with exactly 1561
owned by rank 1; raw little-endian int64 bytes SHA256
`8ed78ae9e8890fb0db47ceb0ca849bbf679ce971e33a899dfc93817b2650bd58`.
The scripts check bytes and completion counts outside timed regions and record
all samples. Compare the candidate median with the mean of the pre/post
medians and report pre/post drift. These checks cover bounded warm synthetic
tables and component latency, not full SGLang/vLLM serving, output quality,
peer recovery, or tail latency.
