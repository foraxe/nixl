# SGLang CP8 early/deferred prefetch check

The probe uses the SGLang research `ReaderRuntime` and `ReadPlan` with a
16-QP row264 owner. It supplies table ranges directly to stand in for model
initialization; no full model, distributed CP collective, or service runs.
It exercises `start_forward`, CP8 `begin_prefetch`, explicit deferred layer-14
submission, graph replay and query-order gather on two ID orders. The 253
input token rows are padded to 256; each layer/frame checks 4096 packed and BF16
outputs against independently pread checkpoint bytes and verifies 4048 unique
READ completions across all 16 lanes.

A cross-node H20 run against a real 188.83 GiB checkpoint passed with DOCA
3.5.0098, the NIXL candidate and the same source-pinned adapter described in
this bundle. [`result.json`](result.json) contains the sanitized result, not
checkpoint data. This is correctness evidence, not service/performance or
MCST-required-hardware qualification.

First assemble `TARGET` using the parent directory's
`prepare_sglang_support.sh`, and start an owner with the same row264 checkpoint
and matching NIXL/DOCA runtime. Set the reader's `NIXL_PLUGIN_DIR`, dynamic
library path and GPUNETIO network variables, plus `MODEL`, `OWNER_READY_FILE`,
`SHIM` and a free 16-port `READER_PORT`. Use the same checkpoint on the reference
and reader hosts. From this directory, generate a private reference, then run:

```bash
python make_reference.py --modules "$TARGET" --model "$MODEL" --out "$PRIVATE_REFERENCE"
python probe.py --modules "$TARGET" --model "$MODEL" --ready "$OWNER_READY_FILE" \
  --library "$SHIM" --reference "$PRIVATE_REFERENCE" --port "$READER_PORT" \
  --out "$RESULT_JSON"
```

The reference contains checkpoint row bytes; keep it private and remove it
only after the test is complete. The ready file contains access information.
The reference generator uses a deterministic sequence which differs from the
one used in the recorded test. A newly assembled public bundle and this
reference generator passed an 8192-row synthetic fixture with the same CP8,
16-QP and graph checks. Stop the owner only after the reader has released its
lease.
