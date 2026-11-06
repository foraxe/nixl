# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize verified GDAKI/NIXL pipeline samples; never invent absent data."""
import json
import math
import statistics
import sys


def summarize(path):
    with open(path) as stream:
        data = json.load(stream)
    if (data.get("schema"), data.get("gin_backend"), data.get("nic_handler")) != (
        2,
        "gdaki",
        2,
    ):
        raise ValueError("not a GDAKI-only result")
    count = data["samples_per_mode"]
    if count < 2 or set(data["modes"]) != {"384", "4096"}:
        raise ValueError("incomplete measurement matrix")
    result = {}
    for rows, modes in data["modes"].items():
        if set(modes) != {"gin_eager", "nixl_eager", "gin_graph", "nixl_graph"}:
            raise ValueError("incomplete modes")
        result[rows] = {}
        for schedule in ("eager", "graph"):
            gin, nixl = modes["gin_" + schedule], modes["nixl_" + schedule]
            if len(gin) != count or len(nixl) != count:
                raise ValueError("incomplete samples")
            summary = {}
            for field, scale in (
                ("latency_ns", 0.001),
                ("gpu_us", 1),
                ("process_cpu_ns", 0.001),
                ("thread_cpu_ns", 0.001),
            ):
                a, b = [x[field] * scale for x in gin], [x[field] * scale for x in nixl]
                if not all(math.isfinite(x) and x > 0 for x in a + b):
                    raise ValueError("invalid timing")
                median_a, median_b = statistics.median(a), statistics.median(b)
                summary[field] = {
                    "gdaki_median_us": median_a,
                    "nixl_median_us": median_b,
                    "nixl_reduction_pct": 100 * (1 - median_b / median_a),
                    "paired_nixl_wins": sum(y < x for x, y in zip(a, b)),
                    "pairs": count,
                }
            result[rows][schedule] = summary
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=2))
