#!/usr/bin/env python3
"""Native SGLang pinned-Host gather probe; results are bounded-table evidence."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch


def quantiles(values):
    x = torch.tensor(values, dtype=torch.float64)
    return {f"p{p}": float(torch.quantile(x, p / 100)) for p in (50, 90, 99)}


def make_synthetic_table(rows, seed=20261027, layer_tag=None):
    if not 4 <= rows <= 1 << 24:
        raise ValueError("synthetic row identity prefix supports [4, 2**24] rows")
    torch.manual_seed(seed)
    weight = torch.empty((rows, 256), dtype=torch.uint8, pin_memory=True)
    scales = torch.empty((rows, 8), dtype=torch.uint8, pin_memory=True)
    for start in range(0, rows, 16384):
        end = min(start + 16384, rows)
        block = weight[start:end]
        block.random_(0, 256)
        block.masked_fill_((block & 0x7f) == 0x7f, 0x7e)
        row = torch.arange(start, end, dtype=torch.int64)
        for c in range(4):
            block[:, c] = ((row >> (6 * c)) & 63).to(torch.uint8)
        if layer_tag is not None:
            block[:, 4] = layer_tag
        scales[start:end] = (124 + (row[:, None] + torch.arange(8)) % 7).to(torch.uint8)
    return weight, scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--rows", type=int, default=1 << 20)
    ap.add_argument("--queries", type=int, nargs="+", default=[24, 384, 6144])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--transport-lib", type=Path)
    ap.add_argument("--owner-port", type=int, default=59231)
    ap.add_argument("--reader-port", type=int, default=59232)
    ap.add_argument("--evict-l2", action="store_true", help="evict GPU cache outside timing")
    ap.add_argument("--skip-scalar", action="store_true", help="reuse the already measured scalar control")
    ap.add_argument("--window-rows", type=int, default=32)
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--trace-layer", type=int, default=1)
    ap.add_argument("--trace-cases", nargs="+", default=["decode_b1", "decode_b16", "prefill_256"])
    args = ap.parse_args()
    if not 4 <= args.rows <= 1 << 24 or args.samples < 2:
        raise ValueError("rows must be [4, 2**24] and samples >= 2")
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(8)
    torch.manual_seed(20261027)
    trace = None
    if args.trace:
        trace = torch.load(args.trace, map_location="cpu", weights_only=True)
        trace_ids = [trace["cases"][name]["row_ids"][str(args.trace_layer)].to(torch.int64)
                     for name in args.trace_cases]
        args.queries = [ids.numel() for ids in trace_ids]
    if args.window_rows < 1:
        raise ValueError("window rows must be positive")
    spec = importlib.util.spec_from_file_location("native_engram_gather", args.reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weight, scales = make_synthetic_table(args.rows)

    owner = reader = None
    if args.transport_lib:
        from engram_transport import Context
        owner = Context(args.transport_lib, args.gpu, args.owner_port)
        owner.register([weight, scales], host=True)
        reader = Context(args.transport_lib, args.gpu, args.reader_port)
        payload_offset = 64
        scale_offset = payload_offset + max(args.queries) * 256 + 64
        dump_offset = (scale_offset + max(args.queries) * 8 + 127) // 64 * 64
        arena = torch.full((dump_offset + 64,), 0xa5, dtype=torch.uint8, device="cuda")
        stats = torch.zeros(8, dtype=torch.uint64, device="cuda")
        reader.register([arena], host=False)
        reader.connect(owner)

    cases = []
    eviction = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda") if args.evict_l2 else None
    def prepare_cache():
        if eviction is not None:
            eviction.add_(1)
            torch.cuda.synchronize()
    for case_index, n in enumerate(args.queries):
        if n < 4:
            raise ValueError("queries must be >= 4")
        if trace is None:
            ids_cpu = torch.randint(0, args.rows, (n,), dtype=torch.int64)
            ids_cpu[:4] = torch.tensor([0, args.rows - 1, args.rows // 2, 0])
            case_metadata = {"workload": "fixed random/repeated/boundary control"}
        else:
            global_ids = trace_ids[case_index]
            ids_cpu = global_ids.remainder(args.rows)
            case_metadata = {"workload": args.trace_cases[case_index], "model_layer": args.trace_layer,
                             "address_projection": "global row ID modulo synthetic backing rows",
                             "unique_model_rows": global_ids.unique().numel(),
                             "unique_backing_rows": ids_cpu.unique().numel()}
        ids = ids_cpu.cuda()
        storage = torch.full((n + 2, 256), 5.0, dtype=torch.bfloat16, device="cuda")
        out = storage[1:-1]
        selected = weight[ids_cpu].view(torch.float8_e4m3fn).float()
        factor = torch.exp2(scales[ids_cpu].float() - 127).repeat_interleave(32, dim=1)
        expected = (selected * factor).to(torch.bfloat16)
        def run():
            module.engram_gather(weight.data_ptr(), scales.data_ptr(), ids, out, 256, 32,
                                 row_lo=0, row_hi=args.rows)
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16)), "native output mismatch"
        assert bool((storage[0] == 5).all() and (storage[-1] == 5).all()), "guard overwrite"
        gpu_us, wall_us = [], []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(args.samples):
            prepare_cache()
            begin = time.perf_counter_ns()
            start_event.record()
            run()
            end_event.record()
            end_event.synchronize()
            wall_us.append((time.perf_counter_ns() - begin) / 1000)
            gpu_us.append(start_event.elapsed_time(end_event) * 1000)
        cases.append({"arm": "native_host", **case_metadata, "queries": n, "payload_and_scale_bytes": n * 264,
                      "gpu_event_us": quantiles(gpu_us), "host_to_complete_us": quantiles(wall_us),
                      "gpu_event_samples_us": gpu_us, "host_to_complete_samples_us": wall_us,
                      "validation": "PASS"})
        if reader is not None:
            sequential = torch.arange(n, dtype=torch.int64, device="cuda")
            codes = arena[payload_offset:payload_offset + n * 256].view(n, 256)
            fetched_scales = arena[scale_offset:scale_offset + n * 8].view(n, 8)
            for arm, window in (("scalar_get", 0), ("batched_get", args.window_rows)):
                if args.skip_scalar and arm == "scalar_get":
                    continue
                metrics = {k: [] for k in ("gpu_event_us", "host_to_complete_us", "posting_us",
                                            "wait_us", "fetch_device_us", "dequant_event_us")}
                midpoint = torch.cuda.Event(enable_timing=True)
                for _ in range(3):
                    arena.fill_(0xa5)
                    reader.fetch(ids, n, payload_offset, scale_offset, dump_offset, window, stats)
                    module.engram_gather(codes.data_ptr(), fetched_scales.data_ptr(), sequential,
                                         out, 256, 32, row_lo=0, row_hi=n)
                    reader.record_consumed()
                torch.cuda.synchronize()
                assert stats.cpu().tolist()[0] == 0, (arm, "warmup status")
                assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16)), (arm, "warmup output")
                for _ in range(args.samples):
                    arena.fill_(0xa5)
                    storage.fill_(5)
                    prepare_cache()
                    begin = time.perf_counter_ns()
                    start_event.record()
                    reader.fetch(ids, n, payload_offset, scale_offset, dump_offset, window, stats)
                    midpoint.record()
                    module.engram_gather(codes.data_ptr(), fetched_scales.data_ptr(), sequential,
                                         out, 256, 32, row_lo=0, row_hi=n)
                    reader.record_consumed()
                    end_event.record()
                    end_event.synchronize()
                    elapsed = (time.perf_counter_ns() - begin) / 1000
                    status = stats.cpu().tolist()
                    assert status[0] == 0, (arm, status)
                    assert status[4] == 2 * n, (arm, "READ count", status)
                    assert torch.equal(codes.cpu(), weight[ids_cpu]), (arm, "payload")
                    assert torch.equal(fetched_scales.cpu(), scales[ids_cpu]), (arm, "scales")
                    assert torch.equal(out.cpu().view(torch.int16), expected.view(torch.int16)), (arm, "native dequant")
                    assert bool((storage[0] == 5).all() and (storage[-1] == 5).all()), "output guard"
                    for region in (arena[:payload_offset], arena[payload_offset+n*256:scale_offset],
                                   arena[scale_offset+n*8:dump_offset], arena[dump_offset+1:]):
                        assert bool((region == 0xa5).all()), (arm, "scratch guard/hole")
                    metrics["gpu_event_us"].append(start_event.elapsed_time(end_event) * 1000)
                    metrics["host_to_complete_us"].append(elapsed)
                    metrics["posting_us"].append(status[1] / 1000)
                    metrics["wait_us"].append(status[2] / 1000)
                    metrics["fetch_device_us"].append(status[3] / 1000)
                    metrics["dequant_event_us"].append(midpoint.elapsed_time(end_event) * 1000)
                cases.append({"arm": arm, **case_metadata, "queries": n, "payload_and_scale_bytes": n * 264,
                              **{k: quantiles(v) for k, v in metrics.items()}, "samples": metrics,
                              "read_wqes": status[4], "dump_wqes": status[5],
                              "doorbells": status[6], "sq_depth": status[7], "validation": "PASS"})
    if reader is not None:
        reader.close()
        owner.close()
    result = {"scope": "single-reader same-backing Host/scalar/batched GET; synthetic unique rows; bounded table; initial probe",
              "rows": args.rows, "host_table_bytes": args.rows * 264,
              "gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(),
              "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "source_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "gpu_cache_mode": "128MiB eviction before each sample" if args.evict_l2 else "warm fixed IDs",
              "batch_window_rows_requested": args.window_rows,
              "samples_per_case": args.samples, "cases": cases}
    if trace is not None:
        result["workload_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
        result["workload_summary"] = trace["summary"]
    if args.transport_lib:
        result["transport_library_sha256"] = hashlib.sha256(args.transport_lib.read_bytes()).hexdigest()
        manifest = args.transport_lib.parent / "runtime-manifest.json"
        if manifest.is_file():
            result["runtime_identities"] = json.loads(manifest.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "runtime_identities")}))
    for case in cases:
        print(json.dumps({k: v for k, v in case.items() if "samples" not in k}))


if __name__ == "__main__":
    main()
