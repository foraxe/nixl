# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures validate preparation metadata and reject incomplete evidence."""

import tempfile
import unittest
from pathlib import Path

from compare_preparation import HEADER, MODES, analyze, arm_order


class PreparationComparisonTest(unittest.TestCase):
    def write_runs(self, root, parent=False):
        modes = list(MODES) + (["parent"] if parent else [])
        (root / "pair0.order").write_text("\n".join(arm_order(0, parent)) + "\n")
        for mode in modes:
            stem = f"pair0-{mode}"
            tx = [",".join(HEADER + ([] if mode == "parent" else ["prep_mode"]))]
            rx = []
            for batch in range(3600):
                window = (1, 8, 32)[batch // 1200]
                phase = "warmup" if batch % 1200 < 200 else "measure"
                elapsed = {
                    "mode0": 120000,
                    "mode1": 100000,
                    "mode2": 80000,
                    "parent": 120000,
                }[mode]
                row = [
                    "exclusive",
                    "sender",
                    batch,
                    window,
                    phase,
                    elapsed,
                    -6 if batch == 0 else 0,
                    -6 if window == 32 else 0,
                    -6 if batch < 2 else -999,
                    0,
                    1,
                    0,
                    0,
                    64 // window + 1,
                ]
                if mode != "parent":
                    row.append(int(mode[-1]))
                tx.append(",".join(map(str, row)))
                rx.append(f"exclusive,receiver,{batch},{window},{phase},0,0,0")
            (root / f"{stem}.sender.log").write_text("\n".join(tx) + "\n")
            (root / f"{stem}.receiver.log").write_text("\n".join(rx) + "\n")
            (root / f"{stem}.exit").write_text("SOURCE=0 TARGET=0\n")

    def test_modes_and_optional_parent(self):
        for parent in (False, True):
            with self.subTest(parent=parent), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.write_runs(root, parent)
                result = analyze(root, pairs=1, include_parent=parent)
                self.assertEqual(
                    result["validated_batches"], 3600 * (4 if parent else 3)
                )
                self.assertEqual(result["expected_wqes_per_batch"], 65)
                comparison = result["metrics"][8]["p50_us"]["comparisons"]
                self.assertAlmostEqual(
                    comparison["mode2_vs_mode1"]["median_paired_pct"], -20
                )
                if parent:
                    self.assertEqual(
                        comparison["mode0_vs_parent"]["median_paired_pct"], 0
                    )
                    with self.assertRaises(ValueError):
                        analyze(root, pairs=1)
                reverse = arm_order(0, parent)[::-1]
                (root / "pair0.order").write_text("\n".join(reverse) + "\n")
                (root / "order_provenance.txt").write_text(
                    "Reconstructed from executed script.\n"
                )
                result = analyze(root, pairs=1, include_parent=parent)
                self.assertEqual(result["orders"][0], reverse)
                self.assertEqual(
                    result["order_provenance"], "Reconstructed from executed script."
                )

    def test_reject_bad_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_runs(root)
            path = root / "pair0-mode2.sender.log"
            original = path.read_text()
            mutations = {
                "profile": (0, "gpu-sharing-control"),
                "role": (1, "receiver"),
                "batch": (2, "1201"),
                "window": (3, "32"),
                "phase": (4, "measure"),
                "elapsed": (5, "0"),
                "wrong_grid": (6, "-6"),
                "capacity": (7, "-6"),
                "stale_token": (8, "0"),
                "release": (9, "-3"),
                "fixed_commits": (10, "0"),
                "probes": (11, "1"),
                "probe_count": (12, "64"),
                "commits": (13, "65"),
                "mode": (14, "1"),
            }
            for name, (column, value) in mutations.items():
                with self.subTest(name=name):
                    lines = original.splitlines()
                    row = lines[1201].split(",")
                    row[column] = value
                    lines[1201] = ",".join(row)
                    path.write_text("\n".join(lines) + "\n")
                    with self.assertRaises(ValueError):
                        analyze(root, pairs=1)
            for name, text in (
                ("missing_header", "\n".join(original.splitlines()[1:])),
                ("bad_header", original.replace("prep_mode", "unknown_mode", 1)),
                ("legacy_row", original.replace(",2\n", "\n", 1)),
            ):
                with self.subTest(name=name):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        analyze(root, pairs=1)
            path.write_text(original)
            for suffix, bad in (
                ("receiver.log", "exclusive,receiver,0,1,warmup,0,0,0\n"),
                ("exit", "SOURCE=0 TARGET=1\n"),
            ):
                bad_path = root / f"pair0-mode2.{suffix}"
                saved = bad_path.read_text()
                bad_path.write_text(bad)
                with self.assertRaises(ValueError):
                    analyze(root, pairs=1)
                bad_path.write_text(saved)
            (root / "pair0.order").write_text("mode2\nmode1\nmode1\n")
            with self.assertRaises(ValueError):
                analyze(root, pairs=1)
            with self.assertRaises(ValueError):
                analyze(root, pairs=0)


if __name__ == "__main__":
    unittest.main()
