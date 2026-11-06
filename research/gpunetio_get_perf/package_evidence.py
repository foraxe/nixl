# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retain numeric samples and selected validation/configuration records, not machine logs."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(raw, output, tag):
    output.mkdir()  # Refuse to overwrite a retained experiment.
    for pair in (1, 2, 3):
        for arm in ("pre", "relaxed", "post"):
            name = f"{tag}-{pair}-{arm}"
            for rank in (0, 1):
                src, dst = raw / name / f"rank{rank}", output / name / f"rank{rank}"
                dst.mkdir(parents=True)
                stdout = src / "stdout.log"
                provenance = src / "provenance.txt"
                markers = [
                    line
                    for line in stdout.read_text().splitlines()
                    if any(
                        word in line
                        for word in (
                            "GDAKI_VERIFIED",
                            "CLEAN_CLOSE",
                            "PASS rows=",
                            "native data MR relaxed ordering=",
                        )
                    )
                ]
                selected = []
                for line in provenance.read_text().splitlines():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and len(parts[0]) == 64:
                        selected.append(parts[0] + "  " + Path(parts[1]).name)
                    elif line.startswith(
                        ("NCCL_", "NIXL_", "GPUNETIO_", "CUDA_VISIBLE_DEVICES=")
                    ):
                        selected.append(line)
                saved = {
                    "description": "Selected source-log markers and configuration; not a full log",
                    "raw_stdout_sha256": digest(stdout),
                    "raw_provenance_sha256": digest(provenance),
                    "markers": markers,
                    "provenance": selected,
                }
                (dst / "validation.json").write_text(json.dumps(saved, indent=2) + "\n")
                if rank == 1:
                    data = src / "results.json"
                    (dst / "results.json").write_bytes(data.read_bytes())


if __name__ == "__main__":
    package(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
