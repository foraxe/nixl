#!/usr/bin/env python3
"""Recompute the published paired component medians from recorded samples."""
import json
import statistics
from pathlib import Path

DATA = json.loads(Path(__file__).with_name("component_samples.json").read_text())
ARMS = ("pre", "new", "post")
EXPECTED = {
    "pre": "ad6086bda139657787cf382e1e6209b6c4134a2e39428877f207bf5b879dea5a",
    "new": "db51790fdca8832e53a15bcba8674d7c9a1e77d73086ab3e03cc17c810c6d00f",
    "post": "ad6086bda139657787cf382e1e6209b6c4134a2e39428877f207bf5b879dea5a",
}


def values(case, arm, queries=None):
    data = DATA[f"public-{case}-{arm}"]
    if data["library_sha256"] != EXPECTED[arm]:
        raise ValueError(f"{case}/{arm}: unexpected adapter hash")
    if queries is None:
        if data["validation"] != "PASS" or not data["cuda_graph"]:
            raise ValueError(f"{case}/{arm}: validation or graph missing")
        row, = data["results"]
        samples = row["samples"]
        if row["owned"] != 1561 or any(s["read_wqes"] != 1561 for s in samples):
            raise ValueError(f"{case}/{arm}: wrong READ count")
        numbers = [s["gpu_us"] for s in samples]
    else:
        row, = (r for r in data["results"] if r["queries"] == queries)
        if row["validation"] != "PASS":
            raise ValueError(f"{case}/{arm}/{queries}: validation failed")
        numbers = row["gpu_event_us"]
    if len(numbers) != 7 or any(v <= 0 for v in numbers):
        raise ValueError(f"{case}/{arm}: expected seven positive GPU samples")
    return statistics.median(numbers)


print("case | pre us | new us | post us | candidate vs midpoint | baseline drift | gate")
for label, case, queries in (
    ("16 QPs", "16qp", None),
    ("1 QP", "1qp", None),
    ("vLLM 384", "vllm", 384),
    ("vLLM 6144", "vllm", 6144),
):
    pre, new, post = (values(case, arm, queries) for arm in ARMS)
    midpoint = (pre + post) / 2
    delta = 100 * (new / midpoint - 1)
    drift = 100 * abs(post - pre) / midpoint
    gate = "PASS" if delta <= 5 and drift <= 5 else "FAIL"
    print(f"{label} | {pre:.3f} | {new:.3f} | {post:.3f} | {delta:+.2f}% | {drift:.2f}% | {gate}")
    if gate != "PASS":
        raise SystemExit(1)
