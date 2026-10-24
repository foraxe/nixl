#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
"""Validate matched EXCLUSIVE runs and compare fixed versus grouped commits."""

import argparse
import json
import statistics
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (
        position - lo
    )


def read_run(root, stem, grouped):
    assert (root / f"{stem}.exit").read_text().strip() == "SOURCE=0 TARGET=0", stem
    logs = {
        role: (root / f"{stem}.{role}.log").read_text().splitlines()
        for role in ("sender", "receiver")
    }
    sender = [
        line.split(",")
        for line in logs["sender"]
        if line.startswith("exclusive,sender,")
    ]
    receiver = [
        line.split(",")
        for line in logs["receiver"]
        if line.startswith("exclusive,receiver,")
    ]
    assert len(sender) == len(receiver) == 3600, (stem, len(sender), len(receiver))
    for b, (tx, rx) in enumerate(zip(sender, receiver)):
        window = (1, 8, 32)[b // 1200]
        phase = "warmup" if b % 1200 < 200 else "measure"
        assert len(tx) == 14 and len(rx) == 8, stem
        assert int(tx[2]) == int(rx[2]) == b, stem
        assert int(tx[3]) == int(rx[3]) == window, stem
        assert tx[4] == rx[4] == phase, stem
        assert int(tx[5]) > 0 and int(tx[9]) == 0, stem
        assert list(map(int, rx[5:])) == [0, 0, 0], stem
        assert int(tx[6]) == (-6 if b == 0 else 0), stem
        assert int(tx[7]) == (-6 if window == 32 else 0), stem
        assert int(tx[8]) == (-6 if b < 2 else -999), stem
        assert list(map(int, tx[10:13])) == [int(grouped), 0, 0], stem
        assert int(tx[13]) == (64 // window + 1 if grouped else 65), stem
    return {
        window: [
            int(row[5]) / 1000
            for row in sender
            if int(row[3]) == window and row[4] == "measure"
        ]
        for window in (1, 8, 32)
    }


def analyze(root, pairs=5):
    runs = {}
    for pair in range(pairs):
        for mode in ("fixed", "grouped"):
            runs[pair, mode] = read_run(root, f"pair{pair}-{mode}", mode == "grouped")
    result = {}
    for window in (1, 8, 32):
        metrics = {}
        for label, p in (("p50_us", 0.5), ("p99_us", 0.99)):
            baseline = [percentile(runs[i, "fixed"][window], p) for i in range(pairs)]
            grouped = [percentile(runs[i, "grouped"][window], p) for i in range(pairs)]
            changes = [100 * (b / a - 1) for a, b in zip(baseline, grouped)]
            metrics[label] = {
                "fixed": baseline,
                "grouped": grouped,
                "fixed_median": statistics.median(baseline),
                "grouped_median": statistics.median(grouped),
                "paired_pct": changes,
                "median_paired_pct": statistics.median(changes),
                "favorable_pairs": sum(x < 0 for x in changes),
            }
        result[window] = metrics
    return {
        "validation": "PASS",
        "pairs": pairs,
        "profile": "EXCLUSIVE",
        "measured_per_window_per_arm_run": 1000,
        "validated_batches": pairs * 2 * 3600,
        "scope": "GPU timed staging/commit/completion/lease release, same host",
        "metrics": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(analyze(args.results, args.pairs), indent=2))
