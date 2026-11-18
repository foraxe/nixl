# DOCA 3.5 GET: public component evidence

This package contains a synthetic two-layer row264 cross-node benchmark and the source needed for its application-side reader and owner. The candidate uses four rails with four QPs each, public FIFO completion and an opt-in shape dispatch of 1 QP through 16 rows, 4 through 4095, and 16 from 4096. The existing per-lookup dedup path is retained for 4096 and above. The delivered baseline uses 16 separate single-rail contexts and its original 1/16 dispatch. Improvements combine application dispatch and rail distribution; they are not an isolated NIXL kernel or FIFO gain. This is not a full model, serving, or general CUDA Graph lifecycle result.

The tested candidate NIXL tree corresponds to commit `e56e4afa985ee2997f78675ae897c964d1d3568f`, tree `8c6c96a38f70b3c95559b02a60894289cd264de5`, on upstream base `e0fd3c26a723da90e21167cd339404d166e30cc7`. It is not a claim about later main. The pinned candidate commit is retained in the history of this evidence branch; source hashes identify the tested tree independently of later rebases. The delivered baseline uses NIXL `bdf954038aa76cec4b63d2fb556debc0a62ea696` and SGLang `7fde8d0ac3073a403fef0ef3be401ff7444c4f42`. Both arms used 16 total QPs, but different source and libraries.

## Included results

`results/` has four measured Graph runs: baseline pre/post and two selected candidate runs. Each JSON preserves its original 20 GPU-event samples per size, lane/completion data, correctness checks and configuration fields; paths, host/device identifiers and environment details were omitted by an explicit allowlist. `selected-decode-overlap.json` contains the separate decode-stream correctness check, not a quantitative overlap measurement. `device-fault-excerpt.txt` contains the two verbatim PASS lines from the native mixed-CQE fault log, with all earlier runtime logs omitted; it is an excerpt, not a complete raw log. The complete original fault log is not distributed here. Five warmups preceded each 20-sample Graph run, with varying IDs and synthetic 1M-row tables per layer. CUDA event time includes the same per-layer completion instrumentation in both arms; host wall time includes prequeue and is not serving latency.

| Rows per layer | Baseline pre/post midpoint, us | Selected candidate, us |
| ---: | ---: | ---: |
| 2 | ~78 | ~80 |
| 64 | ~174 | ~107 |
| 1024 | ~1877 | ~550 |
| 4096 | ~812 | ~575 |
| 32768 | ~5734 | ~3999 |

The experiment used two H20 hosts, CUDA nvcc 13.0 and DOCA 3.5.0098. Tested base and facade SHA256: `12977f4a648721077eb69a0129f398ba88551f3fa738dd9835a5d9276d976947`, `d8f14caadacde4eace246d1a0869c24d09cba0570872114cfc436cb08aced6d5`. Baseline native shim SHA256: `5cf21976e76a76b221a8491655dcedd354bf0964440e0e56d676eafdbe15bd88`. Check actual binary identities independently; rebuilds may differ.

## Rebuild

Use the [NIXL project](https://github.com/ai-dynamo/nixl) build instructions for matching CUDA/DOCA core and GPUNETIO plugin. Python requires CUDA-enabled PyTorch and Triton. The following reconstructs the candidate source into a directory outside any Git working tree:

```sh
git clone --filter=blob:none --no-checkout --single-branch --branch research/get-closeout-20261118 https://github.com/foraxe/nixl.git "$NIXL_GIT"
mkdir -p "$NIXL_SRC"
git -C "$NIXL_GIT" archive e56e4afa985ee2997f78675ae897c964d1d3568f | tar -x -C "$NIXL_SRC"
printf '%s\n' e56e4afa985ee2997f78675ae897c964d1d3568f > "$NIXL_SRC/SOURCE_COMMIT"
(cd "$NIXL_SRC" && sha256sum -c "$PACKAGE/NIXL_SRC_SHA256SUMS")
```

Build matching core/plugin from that tree. Then run `NIXL_REPO="$NIXL_SRC" NIXL_LIBDIR="$NIXL_CORE_LIBDIR" DOCA_INCLUDE="$DOCA_INCLUDE" OUT_DIR="$OUT_DIR" "$PACKAGE/build.sh"`. The build script forces public FIFO, verifies the source manifest and checks the delivered group ABI. Compile the baseline source against the matching `bdf954...` NIXL core and headers with `NIXL_SOURCE="$BASE_NIXL_SRC" NIXL_BUILD="$BASE_NIXL_BUILD" CUDA_HOME="$CUDA_HOME" DOCA_INCLUDE="$DOCA_INCLUDE" OUTPUT="$BASE_SO" "$PACKAGE/baseline/build_baseline.sh"`. Runtime plugin and loader paths must match each build. The package does not contain binary DSOs.

## Run

Reserve four consecutive native ports per owner/reader, an independent control port, and compatible NICs on each host. Set `GPUNETIO_DEVICE_API_OOB_INTERFACE` and `GPUNETIO_DEVICE_API_GID_INDEX` if your network requires them, and set the matching `LD_LIBRARY_PATH` for each process. Start a fresh owner for every reader run. For the baseline, set `GPUNETIO_NATIVE_QPS=1` and `GPUNETIO_DEVICE_API_RDMA` on each endpoint and use the baseline shim as both `OWNER_SO` and `READER_SO`.

For the candidate owner, set `ENGRAM_ABI3_BASE_LIBRARY` to the candidate base, `ENGRAM_ABI3_RAILS` to its four comma-separated local NIC names, `ENGRAM_ABI3_RAIL_QPS=4`, and use the facade as `OWNER_SO`. The reader uses the base as `READER_SO`; pass its four local devices with `--rails`. Neither endpoint's NIC names nor control address need match the example placeholders:

```sh
python "$PACKAGE/matched_readplan.py" --role owner --library "$OWNER_SO" --port "$OWNER_PORT" --control-port "$CONTROL_PORT" --rows 1048576
python "$PACKAGE/matched_readplan.py" --role reader --library "$READER_SO" --port "$READER_PORT" --control-port "$CONTROL_PORT" --owner-address "$OWNER_ADDRESS" --rows 1048576 --mode graph --plan multirail --dispatch optimized --rails "$NIC0" "$NIC1" "$NIC2" "$NIC3" --sizes 2 64 1024 4096 32768 --arm candidate --out "$OUTPUT_JSON"
```

For the delivered baseline reader use `--plan delivered --dispatch delivered` and omit `--rails`. Repeat baseline before and after the candidate. Add `--decode-overlap` for the separate decode-Graph correctness check; it allocates an independent first-rail context and requires reader port + 40 to be free. The default output captures local runtime paths and environment: inspect and sanitize it before sharing. The owner prints `OWNER_CLOSED` on normal drain and retains resources if a reader disappears or errors. Fault injection requires a disposable process and safe handling of retained resources; it is not part of the timing comparison.

The included source is distributed under Apache-2.0; `LICENSE` is the SGLang license and `nixl/LICENSE` preserves NIXL's notices. Files under `baseline/` are the minimal Python modules from the pinned SGLang source. The experimental adapter and native bridge are isolated research code; framework integration and production lifecycle qualification remain future work.
