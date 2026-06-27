import unittest

import pandas as pd

from pasm_validation import (
    evaluate_episodes,
    run_synthetic_benchmark,
    run_train_holdout_benchmark,
    summarize_benchmark,
)


class PASMValidationTest(unittest.TestCase):
    def test_episode_iou_matching_metrics(self):
        truth = pd.DataFrame(
            [
                {"start_s": 10.0, "end_s": 20.0, "type": "af_like"},
                {"start_s": 40.0, "end_s": 50.0, "type": "sinus_tachy"},
            ]
        )
        pred = pd.DataFrame(
            [
                {"start_s": 11.0, "end_s": 21.0, "type": "af_like"},
                {"start_s": 80.0, "end_s": 90.0, "type": "af_like"},
                {"start_s": 40.0, "end_s": 49.0, "type": "tachy"},
            ]
        )

        metrics = evaluate_episodes(pred, truth, duration_s=120.0, iou_threshold=0.30)
        macro = metrics[metrics["type"] == "macro"].iloc[0]

        self.assertEqual(int(macro["tp"]), 2)
        self.assertEqual(int(macro["fp"]), 1)
        self.assertEqual(int(macro["fn"]), 0)
        self.assertGreater(macro["f1"], 0.75)

    def test_synthetic_benchmark_runs_pasm_only(self):
        metrics, records = run_synthetic_benchmark(n_records=6, seed=100)
        summary = summarize_benchmark(metrics)

        self.assertEqual(len(records), 6)
        self.assertIn("pasm", set(summary["model"]))
        self.assertEqual(set(summary["model"]), {"pasm"})

        pasm_f1 = float(summary.loc[summary["model"] == "pasm", "episode_f1_mean"].iloc[0])

        self.assertGreaterEqual(pasm_f1, 0.70)

    def test_train_holdout_threshold_tuning_runs(self):
        result = run_train_holdout_benchmark(train_records=4, holdout_records=4, seed=300)
        thresholds = result["thresholds"]
        summary = result["holdout_summary"]

        self.assertIn("af_like", thresholds)
        self.assertIn("pasm_tuned", set(summary["model"]))
        self.assertIn("pasm_default", set(summary["model"]))

        tuned_f1 = float(summary.loc[summary["model"] == "pasm_tuned", "episode_f1_mean"].iloc[0])
        default_f1 = float(summary.loc[summary["model"] == "pasm_default", "episode_f1_mean"].iloc[0])
        self.assertGreaterEqual(tuned_f1, default_f1)


if __name__ == "__main__":
    unittest.main()
