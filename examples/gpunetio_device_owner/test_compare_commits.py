# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
"""Analysis gates must reject asymmetric probes and incomplete validation."""

import tempfile
import unittest
from pathlib import Path

from compare_commits import analyze


class CommitComparisonTest(unittest.TestCase):
    def write_runs(self, root, mutation=None):
        for mode in ("fixed", "grouped"):
            stem = f"pair0-{mode}"
            (root / f"{stem}.exit").write_text("SOURCE=0 TARGET=0\n")
            tx, rx = [], []
            for b in range(3600):
                window = (1, 8, 32)[b // 1200]
                phase = "warmup" if b % 1200 < 200 else "measure"
                grouped = mode == "grouped"
                elapsed = 100000 if grouped and window != 1 else 120000
                row = [
                    "exclusive",
                    "sender",
                    b,
                    window,
                    phase,
                    elapsed,
                    -6 if b == 0 else 0,
                    -6 if window == 32 else 0,
                    -6 if b < 2 else -999,
                    0,
                    int(grouped),
                    0,
                    0,
                    64 // window + 1 if grouped else 65,
                ]
                if grouped and b == 0 and mutation == "probes":
                    row[11:13] = [1, 64]
                if grouped and b == 0 and mutation == "commit_count":
                    row[13] = 1
                tx.append(",".join(map(str, row)))
                rx.append(f"exclusive,receiver,{b},{window},{phase},0,0,0")
            if mode == "grouped" and mutation == "truncated_receiver":
                rx.pop()
            (root / f"{stem}.sender.log").write_text("\n".join(tx) + "\n")
            (root / f"{stem}.receiver.log").write_text("\n".join(rx) + "\n")

    def test_matched_and_rejected_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_runs(root)
            result = analyze(root, pairs=1)
            self.assertEqual(result["metrics"][1]["p50_us"]["median_paired_pct"], 0)
            self.assertAlmostEqual(
                result["metrics"][8]["p50_us"]["median_paired_pct"], -100 / 6
            )
            for mutation in ("probes", "commit_count", "truncated_receiver"):
                with self.subTest(mutation=mutation):
                    self.write_runs(root, mutation)
                    with self.assertRaises(AssertionError):
                        analyze(root, pairs=1)


if __name__ == "__main__":
    unittest.main()
