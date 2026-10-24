#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
"""Validate five paired B runs and write matched per-arm p50/p99 comparisons."""

import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path


def fields(line):
    return dict(re.findall(r"([A-Za-z_]\w*)=([^\s]+)", line))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def episode_rows(path, prefix):
    return [
        fields(line)
        for line in path.read_text().splitlines()
        if line.startswith(prefix)
    ]


def percentile_rows(path, prefix):
    rows = [
        fields(line)
        for line in path.read_text().splitlines()
        if line.startswith(prefix)
    ]
    require(len(rows) == 2, f"{path}: expected one percentile row per arm")
    result = {row["arm"]: row for row in rows}
    require(set(result) == {"ordered", "cqwait"}, f"{path}: missing paired arm")
    return result


def validate_endpoint(path, prefix, expected_episodes, steps, receiver=False):
    rows = episode_rows(path, prefix)
    require(len(rows) == expected_episodes, f"{path}: got {len(rows)} episode rows")
    require(
        [int(row["episode"]) for row in rows] == list(range(expected_episodes)),
        f"{path}: episode sequence has gaps/duplicates",
    )
    counts = Counter((row["sample"], row["arm"]) for row in rows)
    require(
        counts
        == Counter(
            {
                ("warmup", "ordered"): 1,
                ("warmup", "cqwait"): 1,
                ("measured", "ordered"): 50,
                ("measured", "cqwait"): 50,
            }
        ),
        f"{path}: unexpected warmup/measured arm counts {counts}",
    )
    for row in rows:
        require(
            row.get("validation") == "PASS",
            f"{path}: episode {row.get('episode')} failed",
        )
        require(int(row["steps"]) == steps, f"{path}: wrong step count")
        if receiver:
            require(row.get("cpu_errors") == "0,0", f"{path}: CPU/GPU reference errors")
            require(
                float(row["max_attention_error"]) <= 1e-3,
                f"{path}: attention tolerance exceeded",
            )
        else:
            require(row.get("source_error") == "0", f"{path}: source error reported")


def pct_delta(cqwait, ordered):
    return "NA" if ordered == 0 else f"{(cqwait / ordered - 1) * 100:.2f}"


def nearest_rank(values, percentile):
    return sorted(values)[math.ceil(percentile * len(values) / 100) - 1]


def source_step_samples(path):
    rows = episode_rows(path, "REMOTE_ATTENTION_SOURCE_STEP_INTERVAL")
    require(len(rows) == 100, f"{path}: expected 100 measured step-array rows")
    result = {"ordered": [], "cqwait": []}
    counts = Counter(row["arm"] for row in rows)
    require(
        counts == Counter({"ordered": 50, "cqwait": 50}),
        f"{path}: wrong step-array arm counts",
    )
    for row in rows:
        values = row["post_to_next_post_ns"].split(",")
        require(
            row["sample"] == "measured"
            and int(row["steps"]) == 20
            and int(row["samples"]) == 19
            and len(values) == 19
            and all(value.isdigit() and int(value) > 0 for value in values),
            f"{path}: invalid host-derived step array",
        )
        result[row["arm"]].extend(map(int, values))
    require(
        all(len(result[arm]) == 950 for arm in result),
        f"{path}: expected 950 post-start intervals per arm",
    )
    return result


def analyze(result_dir):
    root = Path(result_dir)
    source_logs = sorted(root.glob("run-*.source.log"))
    require(
        len(source_logs) == 5,
        f"expected five source run logs, found {len(source_logs)}",
    )
    output_rows = []
    for index, source_path in enumerate(source_logs, 1):
        name = source_path.name.removesuffix(".source.log")
        consumer_path = root / f"{name}.consumer.log"
        for role, path in (("source", source_path), ("consumer", consumer_path)):
            require(path.is_file(), f"missing {role} log: {path}")
            rc_path = root / f"{name}.{role}.rc"
            require(
                rc_path.read_text().strip() == "0", f"{path}: process did not exit 0"
            )
        meta = fields((root / f"{name}.meta").read_text().replace("\n", " "))
        expected_first = "ordered" if index % 2 else "cqwait"
        require(
            meta.get("first_arm") == expected_first,
            f"{name}: first-arm order not alternated",
        )
        validate_endpoint(source_path, "REMOTE_ATTENTION_SOURCE_EPISODE", 102, 20)
        validate_endpoint(
            consumer_path, "REMOTE_ATTENTION_CONSUMER_EPISODE", 102, 20, receiver=True
        )

        source = percentile_rows(source_path, "REMOTE_ATTENTION_SOURCE_P99")
        source_steps = percentile_rows(source_path, "REMOTE_ATTENTION_SOURCE_STEP_P99")
        consumer = percentile_rows(consumer_path, "REMOTE_ATTENTION_CONSUMER_P99")
        raw_steps = source_step_samples(source_path)
        for arm in ("ordered", "cqwait"):
            require(
                int(source[arm]["samples"]) == 1000,
                f"{name} {arm}: source sample count != 1000",
            )
            require(
                int(consumer[arm]["samples"]) == 1000,
                f"{name} {arm}: receiver sample count != 1000",
            )
            require(
                int(source_steps[arm]["samples"]) == 950,
                f"{name} {arm}: host-step sample count != 950",
            )
            for percentile in (50, 99):
                require(
                    int(source_steps[arm][f"p{percentile}_ns"])
                    == nearest_rank(raw_steps[arm], percentile),
                    f"{name} {arm}: host-step p{percentile} does not match raw intervals",
                )
        row = {"run": name, "first_arm": expected_first}
        metric_keys = {
            "source_rtt": (source, "p50_ns", "p99_ns"),
            "post_to_next_post": (source_steps, "p50_ns", "p99_ns"),
            "ready_to_payload_check": (
                consumer,
                "ready_to_payload_check_p50_ns",
                "ready_to_payload_check_p99_ns",
            ),
            "ready_to_attention": (
                consumer,
                "ready_to_attention_p50_ns",
                "ready_to_attention_p99_ns",
            ),
            "ready_to_ack_post": (
                consumer,
                "ready_to_ack_post_p50_ns",
                "ready_to_ack_post_p99_ns",
            ),
        }
        for metric, (data, p50_key, p99_key) in metric_keys.items():
            for percentile, key in (("p50", p50_key), ("p99", p99_key)):
                ordered, cqwait = int(data["ordered"][key]), int(data["cqwait"][key])
                row[f"ordered_{metric}_{percentile}_ns"] = ordered
                row[f"cqwait_{metric}_{percentile}_ns"] = cqwait
                row[f"cqwait_vs_ordered_{metric}_{percentile}_pct"] = pct_delta(
                    cqwait, ordered
                )
        output_rows.append(row)

    csv_path = root / "paired_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        "PAIR  FIRST  RTT_P50 O/C  RTT_P99 O/C  STEP_P50 O/C  STEP_P99 O/C  ATTENTION_P99 O/C"
    )
    for row in output_rows:
        n = row["run"].split("-")[1]
        print(
            f"{n}     {row['first_arm']:<7} "
            f"{row['ordered_source_rtt_p50_ns']}/{row['cqwait_source_rtt_p50_ns']}  "
            f"{row['ordered_source_rtt_p99_ns']}/{row['cqwait_source_rtt_p99_ns']}  "
            f"{row['ordered_post_to_next_post_p50_ns']}/{row['cqwait_post_to_next_post_p50_ns']}  "
            f"{row['ordered_post_to_next_post_p99_ns']}/{row['cqwait_post_to_next_post_p99_ns']}  "
            f"{row['ordered_ready_to_attention_p99_ns']}/{row['cqwait_ready_to_attention_p99_ns']}"
        )
    print(
        f"PASS: 5 pairs; episode/reference gates pass; 1,000 RTT and 950 GPU-step intervals (host-derived)/arm/run; {csv_path}"
    )
    return output_rows


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} RESULT_DIR")
    try:
        analyze(sys.argv[1])
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"FAIL: {error}")
