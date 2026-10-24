# SPDX-FileCopyrightText: Copyright (c) 2026, Yunxiao Ning
# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path

from analyze_pair_results import analyze


class AnalyzerTest(unittest.TestCase):
    def test_five_valid_pairs_and_all_sample_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for run in range(1, 6):
                name = f"run-{run:02d}-test"
                first = "ordered" if run % 2 else "cqwait"
                (root / f"{name}.meta").write_text(f"first_arm={first}\n")
                for role, prefix in (
                    ("source", "REMOTE_ATTENTION_SOURCE_EPISODE"),
                    ("consumer", "REMOTE_ATTENTION_CONSUMER_EPISODE"),
                ):
                    lines = []
                    for episode in range(102):
                        if episode < 2:
                            arm, sample = (
                                first
                                if episode == 0
                                else ("cqwait" if first == "ordered" else "ordered")
                            ), "warmup"
                        else:
                            pair = (episode - 2) // 2
                            pair_first = (
                                first
                                if pair % 2 == 0
                                else ("cqwait" if first == "ordered" else "ordered")
                            )
                            arm = (
                                pair_first
                                if (episode - 2) % 2 == 0
                                else (
                                    "cqwait" if pair_first == "ordered" else "ordered"
                                )
                            )
                            sample = "measured"
                        pair_index = max(0, (episode - 2) // 2)
                        common = f"episode={episode} pair={pair_index} arm={arm} sample={sample} steps=20 validation=PASS"
                        if role == "source":
                            lines.append(f"{prefix} {common} source_error=0\n")
                            if sample == "measured":
                                lines.append(
                                    f"REMOTE_ATTENTION_SOURCE_STEP_INTERVAL episode={episode} "
                                    f"pair={pair_index} arm={arm} sample=measured "
                                    f"steps=20 samples=19 post_to_next_post_ns={','.join(['100'] * 19)}\n"
                                )
                        else:
                            lines.append(
                                f"{prefix} {common} cpu_errors=0,0 max_attention_error=0.0001\n"
                            )
                    if role == "source":
                        lines += [
                            f"REMOTE_ATTENTION_SOURCE_P99 arm={arm} samples=1000 p50_ns=100 p99_ns=200\n"
                            for arm in ("ordered", "cqwait")
                        ]
                        lines += [
                            f"REMOTE_ATTENTION_SOURCE_STEP_P99 arm={arm} samples=950 p50_ns=100 p99_ns=100\n"
                            for arm in ("ordered", "cqwait")
                        ]
                    else:
                        lines += [
                            f"REMOTE_ATTENTION_CONSUMER_P99 arm={arm} samples=1000 "
                            "ready_to_payload_check_p50_ns=10 ready_to_payload_check_p99_ns=20 "
                            "ready_to_attention_p50_ns=30 ready_to_attention_p99_ns=40 "
                            "ready_to_ack_post_p50_ns=50 ready_to_ack_post_p99_ns=60\n"
                            for arm in ("ordered", "cqwait")
                        ]
                    (root / f"{name}.{role}.log").write_text("".join(lines))
                    (root / f"{name}.{role}.rc").write_text("0\n")
            rows = analyze(root)
            self.assertEqual(len(rows), 5)
            self.assertTrue((root / "paired_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
