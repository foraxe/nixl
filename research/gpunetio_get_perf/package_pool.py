# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain exact numeric records and explicitly selected validation markers/configuration."""
import hashlib
import json
from pathlib import Path
import sys

root, output = map(Path, sys.argv[1:3])
prefix = sys.argv[3] if len(sys.argv) > 3 else "pool-confirm-v1"
output.mkdir()
for name in ("p1", "p32", "i32c4", "p32c4"):
    for trial in (1, 2, 3):
        tag = f"{prefix}-{name}-{trial}"
        for rank in (0, 1):
            src, dst = root / tag / f"rank{rank}", output / tag / f"rank{rank}"
            dst.mkdir(parents=True)
            log = (src / "stdout.log").read_text()
            provenance = (src / "provenance.txt").read_text()
            markers = [
                line
                for line in log.splitlines()
                if any(
                    k in line
                    for k in (
                        "GDAKI_VERIFIED",
                        "NIXL_PREPARED",
                        "CLEAN_CLOSE",
                        "PASS rows=",
                        "Native READ pool QPs per peer:",
                        "native data MR relaxed ordering=",
                    )
                )
            ]
            selected = []
            for line in provenance.splitlines():
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and len(parts[0]) == 64:
                    selected.append(parts[0] + "  " + Path(parts[1]).name)
                elif line.startswith(
                    ("NCCL_", "NIXL_", "GPUNETIO_", "BENCH_", "CUDA_VISIBLE_DEVICES=")
                ):
                    selected.append(line)
            record = {
                "description": "Selected markers/configuration, not a full log",
                "markers": markers,
                "provenance": selected,
                "raw_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
                "raw_provenance_sha256": hashlib.sha256(
                    provenance.encode()
                ).hexdigest(),
            }
            (dst / "validation.json").write_text(json.dumps(record, indent=2) + "\n")
            if rank == 1:
                (dst / "results.json").write_bytes((src / "results.json").read_bytes())
