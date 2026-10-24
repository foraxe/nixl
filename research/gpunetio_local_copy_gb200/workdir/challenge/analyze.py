"""Summarize matched CE controls without selecting a favorable CE encoding."""
import argparse
import json
from pathlib import Path
import random
import statistics as S


def summarize(path):
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.startswith("{")]
    layouts = {(r["count"], r["bytes"]): r for r in rows if r["kind"] == "layout"}
    perf = [r for r in rows if r["kind"] == "perf"]
    out = []
    for count, size in sorted({(r["count"], r["bytes"]) for r in perf}):
        data = [r for r in perf if r["count"] == count and r["bytes"] == size]
        assert all(r["mismatch"] == 0 for r in data)
        layout = layouts[(count, size)]
        assert sorted(layout["indices"]) == list(range(count))
        arms = {arm: [r for r in data if r["arm"] == arm] for arm in {r["arm"] for r in data}}
        assert all(sorted(r["block"] for r in a) == list(range(7)) for a in arms.values())
        medians = {arm: S.median(r["device_us"] for r in a) for arm, a in arms.items()}
        best = min((arm for arm in arms if arm.startswith("CUDA_")), key=medians.get)
        ratios = []
        for block in range(7):
            candidate = next(r["device_us"] for r in arms["NIC_WARP"] if r["block"] == block)
            control = min(r["device_us"] for r in data if r["block"] == block and r["arm"].startswith("CUDA_"))
            ratios.append(control / candidate)
        rng = random.Random(20261023)
        bootstrap = sorted(S.median(rng.choices(ratios, k=7)) for _ in range(10000))
        out.append(dict(run=path.name, seed=layout["seed"], count=count, bytes=size,
                        nic_warp_us=medians["NIC_WARP"], best_cuda=best, best_cuda_us=medians[best],
                        sm_us=medians["SM_GATHER"], median_paired_speedup=S.median(ratios),
                        bootstrap95=[bootstrap[249], bootstrap[9749]],
                        wins=sum(x > 1 for x in ratios), wins_over_10pct=sum(x > 1.1 for x in ratios),
                        pairs=7, all_arm_medians=medians))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([row for path in args.inputs for row in summarize(path)], indent=2))
