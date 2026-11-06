# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit backend-pool measurements against batched GDAKI; no superiority gate."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

root = Path(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else "pool-confirm-v1"
here = Path(__file__).resolve().parent
cases = {"p1": (1, 1, 1), "p32": (32, 1, 32), "i32c4": (32, 0, 4), "p32c4": (32, 1, 4)}
source = hashlib.sha256((here / "gin_nixl_bench.cu").read_bytes()).hexdigest()
adapter = hashlib.sha256((here / "engram_native_nixl35.cu").read_bytes()).hexdigest()
result = {
    "status": "PASS",
    "measured_pipelines": 0,
    "warmup_pipelines": 0,
    "matrix": {},
}
build_path = root / "builds.json"
if not build_path.exists():
    build_path = here / "pool_evidence/builds.json"
build = json.loads(build_path.read_text()) if build_path.exists() else None
if build:
    for filename, expected in build["core_sources"].items():
        assert (
            hashlib.sha256((here.parents[1] / filename).read_bytes()).hexdigest()
            == expected
        )
for name, (qps, pooled, ctas) in cases.items():
    runs = []
    for trial in (1, 2, 3):
        run = root / f"{prefix}-{name}-{trial}"
        data = json.loads((run / "rank1/results.json").read_text())
        engines = 1 if pooled else qps
        assert (
            data["backend_pool"],
            data["nixl_engines"],
            data["gin_contexts"],
            data["gin_sharing"],
            data["gin_aggregate"],
            data["vector_consumer"],
        ) == (pooled, engines, qps, 1, 512, 1)
        assert data["warmups"] == 20 and data["samples_per_mode"] == 100
        for rank in (0, 1):
            saved = run / f"rank{rank}/validation.json"
            if saved.exists():
                record = json.loads(saved.read_text())
                log = "\n".join(record["markers"])
                provenance = "\n".join(record["provenance"])
            else:
                log = (run / f"rank{rank}/stdout.log").read_text()
                provenance = (run / f"rank{rank}/provenance.txt").read_text()
            assert (
                f"GDAKI_VERIFIED rank={rank} device_type=3 contexts={qps} nic_handler=2"
                in log
            )
            assert (
                f"NIXL_PREPARED rank={rank} engines={engines} active_qps={qps} backend_pool={pooled}"
                in log
            )
            assert f"CLEAN_CLOSE rank={rank}" in log
            assert source in provenance and adapter in provenance
            if build:
                assert build["plugin_sha256"] in provenance
            for setting in (
                "NCCL_GIN_GDAKI_QP_DEPTH=2048",
                "NCCL_LSA_TEAM_SIZE=1",
                "NCCL_P2P_DISABLE=1",
                "GPUNETIO_DEVICE_API_RELAXED_ORDERING=true",
            ):
                assert setting in provenance
            assert ("BENCH_NIXL_SINGLE_CTA=4" in provenance) == name.endswith("c4")
            if rank == 1:
                assert log.count("native data MR relaxed ordering=true") == engines
                for rows in (384, 4096):
                    assert f"PASS rows={rows} arms=4 samples=100" in log
        assert set(data["modes"]) == {"384", "4096"}
        for modes in data["modes"].values():
            assert set(modes) == {"gin_eager", "nixl_eager", "gin_graph", "nixl_graph"}
            for samples in modes.values():
                assert len(samples) == 100
                for sample in samples:
                    assert all(
                        math.isfinite(sample[f]) and sample[f] > 0
                        for f in ("latency_ns", "gpu_us", "get_us", "consumer_us")
                    )
                    assert (
                        abs(sample["gpu_us"] - sample["get_us"] - sample["consumer_us"])
                        < 0.15
                    )
        runs.append(data)
        result["measured_pipelines"] += 800
        result["warmup_pipelines"] += 160
    entry = {
        "active_qps": qps,
        "nixl_engines": engines,
        "nixl_ctas": ctas,
        "gin_ctas": qps,
        "sizes": {},
    }
    for rows in ("384", "4096"):
        entry["sizes"][rows] = {}
        for mode in ("gin_eager", "nixl_eager", "gin_graph", "nixl_graph"):
            samples = [s for run in runs for s in run["modes"][rows][mode]]
            entry["sizes"][rows][mode] = {
                "wall_us": st.median(s["latency_ns"] / 1000 for s in samples),
                "gpu_us": st.median(s["gpu_us"] for s in samples),
                "id_get_and_ready_us": st.median(s["get_us"] for s in samples),
                "consumer_and_retirement_us": st.median(
                    s["consumer_us"] for s in samples
                ),
                "samples": len(samples),
            }
    result["matrix"][name] = entry
print(json.dumps(result, indent=2))
