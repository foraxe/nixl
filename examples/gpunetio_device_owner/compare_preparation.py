#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
"""Validate EXCLUSIVE grouped runs with explicit preparation modes 0/1/2."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from compare_commits import percentile

HEADER = (
    "profile,role,batch,window,phase,gpu_elapsed_ns,wrong_grid_status,"
    "capacity_status,stale_token_status,release_status,batch_commit,"
    "precommit_check,precommit_checked,expected_commits"
).split(",")
MODES = ("mode0", "mode1", "mode2")


def require(condition, context):
    if not condition:
        raise ValueError(f"Invalid preparation comparison: {context}")


def arm_order(pair, include_parent=False):
    arms = list(MODES) + (["parent"] if include_parent else [])
    offset = pair % len(arms)
    order = arms[offset:] + arms[:offset]
    return order[::-1] if pair % 2 else order


def read_rows(path, role, parent):
    rows, headers = [], 0
    expected = HEADER + ([] if parent else ["prep_mode"])
    for row in csv.reader(path.read_text().splitlines()):
        if not row:
            continue
        if row[0] == "profile":
            require(role == "sender" and row == expected and headers == 0, path)
            require(not rows, f"{path}: late header")
            headers += 1
        elif row[0] in ("exclusive", "gpu-sharing-control") or (
            len(row) > 1 and row[1] in ("sender", "receiver")
        ):
            require(row[:2] == ["exclusive", role], f"{path}: profile/role")
            require(len(row) == (len(expected) if role == "sender" else 8), path)
            require(role != "sender" or headers == 1, f"{path}: missing header")
            rows.append(row)
    require(headers == (1 if role == "sender" else 0), f"{path}: header")
    require(len(rows) == 3600, f"{path}: expected 3600 rows, got {len(rows)}")
    return rows


def read_run(root, stem, mode):
    require((root / f"{stem}.exit").read_text().strip() == "SOURCE=0 TARGET=0", stem)
    tx = read_rows(root / f"{stem}.sender.log", "sender", mode == "parent")
    rx = read_rows(root / f"{stem}.receiver.log", "receiver", mode == "parent")
    samples = {window: [] for window in (1, 8, 32)}
    for batch, (sender, receiver) in enumerate(zip(tx, rx)):
        window = (1, 8, 32)[batch // 1200]
        phase = "warmup" if batch % 1200 < 200 else "measure"
        context = f"{stem}: batch {batch}"
        require(int(sender[2]) == int(receiver[2]) == batch, context)
        require(int(sender[3]) == int(receiver[3]) == window, context)
        require(sender[4] == receiver[4] == phase, context)
        require(int(sender[5]) > 0 and int(sender[9]) == 0, context)
        require(list(map(int, receiver[5:])) == [0, 0, 0], context)
        require(int(sender[6]) == (-6 if batch == 0 else 0), context)
        require(int(sender[7]) == (-6 if window == 32 else 0), context)
        require(int(sender[8]) == (-6 if batch < 2 else -999), context)
        require(list(map(int, sender[10:13])) == [1, 0, 0], context)
        require(int(sender[13]) == 64 // window + 1, context)
        if mode != "parent":
            require(sender[14] == mode[-1], f"{context}: prep_mode")
        if phase == "measure":
            samples[window].append(int(sender[5]) / 1000)
    require(all(len(values) == 1000 for values in samples.values()), stem)
    return samples


def analyze(root, pairs=5, include_parent=False):
    root = Path(root)
    require(pairs > 0, "pairs must be positive")
    modes = list(MODES) + (["parent"] if include_parent else [])
    stems = {f"pair{pair}-{mode}" for pair in range(pairs) for mode in modes}
    for suffix in ("sender.log", "receiver.log", "exit"):
        require(
            {path.name for path in root.glob(f"pair*.{suffix}")}
            == {f"{stem}.{suffix}" for stem in stems},
            f"unexpected/missing {suffix} files",
        )
    require(
        {path.name for path in root.glob("pair*.order")}
        == {f"pair{pair}.order" for pair in range(pairs)},
        "unexpected/missing order files",
    )
    runs, orders = {}, {}
    for pair in range(pairs):
        order = (root / f"pair{pair}.order").read_text().splitlines()
        require(
            len(order) == len(modes) and set(order) == set(modes),
            f"pair{pair}: arm order",
        )
        orders[pair] = order
        for mode in modes:
            runs[pair, mode] = read_run(root, f"pair{pair}-{mode}", mode)
    comparisons = [("mode1", "mode0"), ("mode2", "mode1"), ("mode2", "mode0")]
    if include_parent:
        comparisons += [("mode0", "parent"), ("mode2", "parent")]
    result = {}
    for window in (1, 8, 32):
        result[window] = {}
        for metric, p in (("p50_us", 0.5), ("p99_us", 0.99)):
            values = {
                mode: [percentile(runs[pair, mode][window], p) for pair in range(pairs)]
                for mode in modes
            }
            deltas = {}
            for candidate, baseline in comparisons:
                changes = [
                    100 * (b / a - 1)
                    for a, b in zip(values[baseline], values[candidate])
                ]
                deltas[f"{candidate}_vs_{baseline}"] = {
                    "paired_pct": changes,
                    "median_paired_pct": statistics.median(changes),
                    "favorable_pairs": sum(value < 0 for value in changes),
                }
            result[window][metric] = {
                "runs": values,
                "medians": {
                    mode: statistics.median(data) for mode, data in values.items()
                },
                "comparisons": deltas,
            }
    return {
        "validation": "PASS",
        "pairs": pairs,
        "modes": modes,
        "orders": orders,
        "order_provenance": (
            (root / "order_provenance.txt").read_text().strip()
            if (root / "order_provenance.txt").is_file()
            else "Unspecified; manifest order is not independently verified."
        ),
        "profile": "EXCLUSIVE",
        "batch_commit": 1,
        "precommit_check": 0,
        "measured_per_window_per_arm_run": 1000,
        "validated_batches": pairs * len(modes) * 3600,
        "expected_wqes_per_batch": 65,
        "expected_commits": {window: 64 // window + 1 for window in (1, 8, 32)},
        "scope": "source GPU staging/commit/completion/lease release; same host",
        "metrics": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--include-parent", action="store_true")
    args = parser.parse_args()
    try:
        print(
            json.dumps(analyze(args.results, args.pairs, args.include_parent), indent=2)
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"FAIL: {error}")
