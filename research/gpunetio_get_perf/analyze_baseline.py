# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate fresh unmodified-GET-branch controls, separate from opt-off controls."""
import hashlib
import json
from pathlib import Path
import statistics as st
import sys

root = Path(sys.argv[1])
runs = []
validation = []
for pair in (1, 2, 3):
    run = root / f"unchanged-base-{pair}"
    for rank in (0, 1):
        text = (run / f"rank{rank}/stdout.log").read_text()
        if (
            f"CLEAN_CLOSE rank={rank}" not in text
            or f"GDAKI_VERIFIED rank={rank}" not in text
        ):
            raise ValueError("unmodified-base qualification failed")
        provenance = (run / f"rank{rank}/provenance.txt").read_text()
        if "GPUNETIO_DEVICE_API_RELAXED_ORDERING=" in provenance:
            raise ValueError("unexpected unsupported option in base run")
        records = [
            line for line in provenance.splitlines() if len(line.split()[0]) == 64
        ]
        validation.append(
            {
                "pair": pair,
                "rank": rank,
                "hash_records": records,
                "raw_stdout_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
    data = json.loads((run / "rank1/results.json").read_text())
    if data["samples_per_mode"] != 100 or data["gin_sharing"] != 1:
        raise ValueError("unmodified-base matrix mismatch")
    runs.append(data)
result = {
    "status": "PASS",
    "validation": validation,
    "measured_pipelines": 2400,
    "matrix": {},
}
for rows in ("384", "4096"):
    result["matrix"][rows] = {}
    for schedule in ("eager", "graph"):
        result["matrix"][rows][schedule] = {
            backend: st.median(
                v["latency_ns"] / 1000
                for run in runs
                for v in run["modes"][rows][backend + "_" + schedule]
            )
            for backend in ("gin", "nixl")
        }
print(json.dumps(result, indent=2))
