# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit three pre/relaxed/post brackets against tuned GDAKI, then evaluate the gate."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root, tag):
    runs = {}
    identities = None
    measured = 0
    for pair in (1, 2, 3):
        for arm in ("pre", "relaxed", "post"):
            run = root / f"{tag}-{pair}-{arm}"
            data = json.loads((run / "rank1/results.json").read_text())
            require(
                (
                    data["schema"],
                    data["gin_backend"],
                    data["nic_handler"],
                    data["gin_sharing"],
                )
                == (2, "gdaki", 2, 1),
                "not GDAKI/CTA/GPU posting",
            )
            require(
                data["samples_per_mode"] == 100 and data["warmups"] == 20,
                "wrong sample count",
            )
            require(
                data["nccl_runtime"] == 23203 and data["gin_contexts"] == 1,
                "runtime/context mismatch",
            )
            for rank in (0, 1):
                evidence = run / f"rank{rank}/validation.json"
                saved = json.loads(evidence.read_text()) if evidence.exists() else None
                log = (
                    "\n".join(saved["markers"])
                    if saved
                    else (run / f"rank{rank}/stdout.log").read_text()
                )
                require(
                    f"CLEAN_CLOSE rank={rank}" in log
                    and f"GDAKI_VERIFIED rank={rank} device_type=3 contexts=1 nic_handler=2"
                    in log,
                    "missing route or clean-close evidence",
                )
                require("MISMATCH" not in log, "payload mismatch")
                manifest = (
                    "\n".join(saved["provenance"])
                    if saved
                    else (run / f"rank{rank}/provenance.txt").read_text()
                )
                required = [
                    "NCCL_GIN_GDAKI_QP_DEPTH=2048",
                    "NCCL_LSA_TEAM_SIZE=1",
                    "NCCL_P2P_DISABLE=1",
                    "NCCL_GIN_GDAKI_NIC_HANDLER=2",
                    "GPUNETIO_DEVICE_API_RDMA=mlx5_0",
                    "GPUNETIO_DEVICE_API_RELAXED_ORDERING="
                    + ("true" if arm == "relaxed" else "false"),
                ]
                require(
                    all(value in manifest for value in required),
                    "configuration mismatch",
                )
                require(
                    "NCCL_IB_PCI_RELAXED_ORDERING=0" not in manifest,
                    "GDAKI RO was disabled",
                )
                hashes = {}
                for line in manifest.splitlines():
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and len(parts[0]) == 64:
                        hashes[Path(parts[1]).name] = parts[0]
                if identities is None:
                    identities = hashes
                require(
                    hashes == identities,
                    "build identities changed across ranks/brackets",
                )
                if rank == 1:
                    require(
                        "native data MR relaxed ordering="
                        + ("true" if arm == "relaxed" else "false")
                        in log,
                        "requested MR ordering was not accepted",
                    )
                    for rows in (384, 4096):
                        require(
                            f"PASS rows={rows} arms=4 samples=100" in log,
                            "missing correctness marker",
                        )
            require(set(data["modes"]) == {"384", "4096"}, "incomplete row matrix")
            for modes in data["modes"].values():
                require(
                    set(modes)
                    == {"gin_eager", "nixl_eager", "gin_graph", "nixl_graph"},
                    "incomplete modes",
                )
                require(
                    all(len(samples) == 100 for samples in modes.values()),
                    "incomplete samples",
                )
                require(
                    all(
                        math.isfinite(sample[field]) and sample[field] > 0
                        for samples in modes.values()
                        for sample in samples
                        for field in ("latency_ns", "gpu_us")
                    ),
                    "non-finite or non-positive timing",
                )
                measured += sum(len(samples) for samples in modes.values())
            runs[pair, arm] = data
    require(
        identities["gin_nixl_bench.cu"] == sha(HERE / "gin_nixl_bench.cu"),
        "source identity mismatch",
    )
    require(
        identities["engram_native_nixl35.cu"] == sha(HERE / "engram_native_nixl35.cu"),
        "adapter source mismatch",
    )
    result = {
        "status": "PASS",
        "measured_pipelines": measured,
        "warmup_pipelines": 1440,
        "identities": identities,
        "matrix": {},
    }
    for rows in ("384", "4096"):
        result["matrix"][rows] = {}
        for schedule in ("eager", "graph"):
            metrics = {}
            for field, scale in (("latency_ns", 0.001), ("gpu_us", 1)):
                a, b, reductions, controls = [], [], [], []
                for pair in (1, 2, 3):
                    relaxed = runs[pair, "relaxed"]["modes"][rows]
                    ga = [v[field] * scale for v in relaxed["gin_" + schedule]]
                    nb = [v[field] * scale for v in relaxed["nixl_" + schedule]]
                    reduction = 100 * (1 - st.median(nb) / st.median(ga))
                    require(
                        reduction >= 10,
                        f"10% target missed: {rows} {schedule} {field} pair {pair}",
                    )
                    reductions.append(reduction)
                    a += ga
                    b += nb
                    bracket = {}
                    for backend in ("gin_", "nixl_"):
                        pre = st.median(
                            v[field] * scale
                            for v in runs[pair, "pre"]["modes"][rows][
                                backend + schedule
                            ]
                        )
                        post = st.median(
                            v[field] * scale
                            for v in runs[pair, "post"]["modes"][rows][
                                backend + schedule
                            ]
                        )
                        drift = 100 * abs(post - pre) / ((pre + post) / 2)
                        require(
                            drift < 5,
                            f"control drift: {rows} {schedule} {backend} {drift}",
                        )
                        bracket[backend[:-1]] = {
                            "pre_us": pre,
                            "post_us": post,
                            "drift_pct": drift,
                        }
                    controls.append(bracket)
                metrics[field] = {
                    "gdaki_median_us": st.median(a),
                    "nixl_median_us": st.median(b),
                    "nixl_reduction_pct": 100 * (1 - st.median(b) / st.median(a)),
                    "per_bracket_reduction_pct": reductions,
                    "paired_nixl_wins": sum(y < x for x, y in zip(a, b)),
                    "pairs": len(a),
                    "strict_controls": controls,
                }
            for backend in ("gdaki", "nixl"):
                require(
                    0.5
                    < metrics["gpu_us"][backend + "_median_us"]
                    / metrics["latency_ns"][backend + "_median_us"]
                    < 1.2,
                    "GPU/wall units disagree",
                )
            result["matrix"][rows][schedule] = metrics
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1]), sys.argv[2]), indent=2))
