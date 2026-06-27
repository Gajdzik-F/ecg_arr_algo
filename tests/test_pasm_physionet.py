import unittest

import numpy as np
import pandas as pd

from pasm_physionet import (
    beat_labels_to_episodes,
    detect_fast_irregular_af,
    detect_short_coupled_ectopy,
    filter_ectopy_candidate_flood,
    filter_predictions_for_annotation_scope,
    map_mitdb_symbols_to_labels,
    match_diagnostic_episodes,
    merge_close_same_type_episodes,
    merge_physionet_evidence,
    postprocess_physionet_episodes,
    rhythm_aux_to_episodes,
)


class PASMPhysioNetTest(unittest.TestCase):
    def test_mitdb_symbol_mapping(self):
        symbols = ["N", "L", "A", "V", "F", "/", "?"]
        labels = map_mitdb_symbols_to_labels(symbols)

        self.assertEqual(labels.tolist(), [
            "normal",
            "normal",
            "ectopic_like",
            "ectopic_like",
            "ectopic_like",
            "normal",
            "normal",
        ])

    def test_rhythm_aux_to_episodes(self):
        samples = np.array([0, 250, 1000, 1500, 2200])
        aux = ["(N", "(AFIB", "", "(N", "(AFL"]
        episodes = rhythm_aux_to_episodes(samples, aux, fs=250, end_sample=3000)

        self.assertEqual(episodes["type"].tolist(), ["af_like", "af_like"])
        self.assertAlmostEqual(float(episodes.iloc[0]["start_s"]), 1.0)
        self.assertAlmostEqual(float(episodes.iloc[0]["end_s"]), 6.0)
        self.assertAlmostEqual(float(episodes.iloc[1]["start_s"]), 8.8)
        self.assertAlmostEqual(float(episodes.iloc[1]["end_s"]), 12.0)

    def test_beat_labels_to_episodes_min_len(self):
        rpeaks = np.arange(0, 10) * 250
        labels = np.array(
            [
                "normal",
                "ectopic_like",
                "ectopic_like",
                "normal",
                "ectopic_like",
                "ectopic_like",
                "ectopic_like",
                "normal",
                "af_like",
                "af_like",
            ],
            dtype=object,
        )
        episodes = beat_labels_to_episodes(rpeaks, labels, fs=250, min_len=3)

        self.assertEqual(episodes["type"].tolist(), ["ectopic_like"])
        self.assertAlmostEqual(float(episodes.iloc[0]["start_s"]), 4.0)
        self.assertAlmostEqual(float(episodes.iloc[0]["end_s"]), 6.0)

    def test_fast_irregular_af_evidence(self):
        n = 80
        features = pd.DataFrame(
            {
                "time_s": np.arange(n, dtype=float),
                "hr": np.r_[np.ones(20) * 80, np.ones(45) * 135, np.ones(15) * 82],
                "local_cv": np.r_[np.ones(20) * 0.02, np.ones(45) * 0.22, np.ones(15) * 0.02],
                "local_rmssd": np.r_[np.ones(20) * 0.02, np.ones(45) * 0.13, np.ones(15) * 0.02],
                "sqi": np.ones(n),
            }
        )
        episodes = detect_fast_irregular_af(features, win_beats=12, min_beats=10)

        self.assertEqual(episodes["type"].tolist(), ["af_like"])
        self.assertLess(float(episodes.iloc[0]["start_s"]), 35.0)
        self.assertGreater(float(episodes.iloc[0]["end_s"]), 55.0)

    def test_merge_physionet_evidence_removes_overlapping_tachy(self):
        base = pd.DataFrame(
            [
                {"start_s": 10.0, "end_s": 20.0, "type": "sinus_tachy"},
                {"start_s": 40.0, "end_s": 45.0, "type": "ectopic_like"},
            ]
        )
        af = pd.DataFrame([{"start_s": 12.0, "end_s": 25.0, "type": "af_like"}])
        merged = merge_physionet_evidence(base, af)

        self.assertNotIn("sinus_tachy", merged["type"].tolist())
        self.assertIn("af_like", merged["type"].tolist())
        self.assertIn("ectopic_like", merged["type"].tolist())

    def test_postprocess_physionet_episodes_consolidates_af_fragments(self):
        episodes = pd.DataFrame(
            [
                {"start_s": 10.0, "end_s": 12.0, "type": "sinus_tachy"},
                {"start_s": 20.0, "end_s": 35.0, "type": "af_like", "beats": 20, "confidence": 0.4},
                {"start_s": 62.0, "end_s": 80.0, "type": "af_like", "beats": 24, "confidence": 0.6},
                {"start_s": 90.0, "end_s": 92.0, "type": "sinus_tachy"},
                {"start_s": 160.0, "end_s": 172.0, "type": "sinus_tachy"},
            ]
        )

        merged = postprocess_physionet_episodes(
            episodes,
            af_merge_gap_s=45.0,
            af_tachy_margin_s=10.0,
            min_tachy_duration_s=3.0,
        )

        self.assertEqual(merged["type"].tolist(), ["af_like", "sinus_tachy"])
        self.assertAlmostEqual(float(merged.iloc[0]["start_s"]), 20.0)
        self.assertAlmostEqual(float(merged.iloc[0]["end_s"]), 80.0)

    def test_short_coupled_ectopy_evidence(self):
        features = pd.DataFrame(
            {
                "time_s": [0.0, 0.8, 1.30, 1.75, 2.55, 3.30],
                "rr_prev": [np.nan, 0.8, 0.50, 0.45, 0.80, 0.75],
                "sqi": np.ones(6),
            }
        )
        episodes = detect_short_coupled_ectopy(features, short_rr_s=0.55, max_following_rr_s=1.05)

        self.assertEqual(episodes["type"].tolist(), ["ectopic_like"])
        self.assertAlmostEqual(float(episodes.iloc[0]["start_s"]), 1.30)
        self.assertAlmostEqual(float(episodes.iloc[0]["end_s"]), 2.55)

    def test_short_coupled_ectopy_ignores_isolated_short_rr(self):
        features = pd.DataFrame(
            {
                "time_s": [0.0, 0.8, 1.25, 2.05, 2.85],
                "rr_prev": [np.nan, 0.8, 0.45, 0.80, 0.80],
                "sqi": np.ones(5),
            }
        )

        episodes = detect_short_coupled_ectopy(features, short_rr_s=0.55, max_following_rr_s=1.05)

        self.assertEqual(len(episodes), 0)

    def test_ectopy_flood_filter_removes_weak_isolated_candidates(self):
        episodes = pd.DataFrame(
            [
                {
                    "start_s": float(i * 3),
                    "end_s": float(i * 3 + 1),
                    "type": "ectopic_like",
                    "confidence": 0.20,
                    "beats": 3,
                    "mean_sqi": 1.0,
                    "flood_filtered": False,
                }
                for i in range(12)
            ]
        )

        filtered = filter_ectopy_candidate_flood(
            episodes,
            recording_duration_s=120.0,
            flood_rate_per_hour=30.0,
            flood_min_confidence=0.40,
            flood_density_window_s=1.0,
            flood_min_density=4,
            flood_min_candidates=10,
        )

        self.assertEqual(len(filtered), 0)

    def test_ectopy_flood_filter_requires_morphology_for_flooded_candidates(self):
        episodes = pd.DataFrame(
            [
                {
                    "start_s": float(i * 2),
                    "end_s": float(i * 2 + 1),
                    "type": "ectopic_like",
                    "confidence": 0.45,
                    "beats": 3,
                    "mean_sqi": 1.0,
                    "mean_morph_z": 0.30,
                    "flood_filtered": False,
                }
                for i in range(12)
            ]
        )
        episodes.loc[5, "mean_morph_z"] = 0.56

        filtered = filter_ectopy_candidate_flood(
            episodes,
            recording_duration_s=120.0,
            flood_rate_per_hour=30.0,
            flood_min_confidence=0.40,
            flood_density_window_s=10.0,
            flood_min_density=4,
            flood_min_candidates=10,
            flood_strong_morph_z=0.55,
            flood_dense_morph_z=0.60,
        )

        self.assertEqual(len(filtered), 1)
        self.assertAlmostEqual(float(filtered.iloc[0]["mean_morph_z"]), 0.56)

    def test_ectopy_flood_filter_keeps_dense_morphology_supported_run(self):
        episodes = pd.DataFrame(
            [
                {
                    "start_s": float(i),
                    "end_s": float(i + 0.5),
                    "type": "ectopic_like",
                    "confidence": 0.25,
                    "beats": 3,
                    "mean_sqi": 1.0,
                    "mean_morph_z": 0.70,
                    "flood_filtered": False,
                }
                for i in range(12)
            ]
        )

        filtered = filter_ectopy_candidate_flood(
            episodes,
            recording_duration_s=120.0,
            flood_rate_per_hour=30.0,
            flood_min_confidence=0.40,
            flood_density_window_s=10.0,
            flood_min_density=4,
            flood_min_candidates=10,
            flood_strong_morph_z=0.55,
            flood_dense_morph_z=0.60,
        )

        self.assertEqual(len(filtered), len(episodes))

    def test_diagnostic_matcher_marks_tp_fp_fn(self):
        predicted = pd.DataFrame(
            [
                {"start_s": 0.0, "end_s": 10.0, "type": "ectopic_like", "confidence": 0.8},
                {"start_s": 30.0, "end_s": 35.0, "type": "ectopic_like", "confidence": 0.4},
            ]
        )
        truth = pd.DataFrame(
            [
                {"start_s": 1.0, "end_s": 9.0, "type": "ectopic_like"},
                {"start_s": 50.0, "end_s": 55.0, "type": "ectopic_like"},
            ]
        )

        diagnostics = match_diagnostic_episodes(predicted, truth, iou_threshold=0.30)

        self.assertEqual(diagnostics["status"].tolist(), ["TP", "FP", "FN"])
        self.assertGreater(float(diagnostics.iloc[0]["iou"]), 0.30)

    def test_mitdb_annotation_scope_keeps_only_ectopy_predictions(self):
        record = type("Record", (), {"record_id": "mitdb/200"})()
        episodes = pd.DataFrame(
            [
                {"start_s": 1.0, "end_s": 2.0, "type": "af_like"},
                {"start_s": 3.0, "end_s": 4.0, "type": "ectopic_like"},
                {"start_s": 5.0, "end_s": 6.0, "type": "sinus_tachy"},
            ]
        )

        scoped = filter_predictions_for_annotation_scope(record, episodes)

        self.assertEqual(scoped["type"].tolist(), ["ectopic_like"])

    def test_merge_close_same_type_episodes(self):
        episodes = pd.DataFrame(
            [
                {"start_s": 1.0, "end_s": 2.0, "type": "ectopic_like", "beats": 3, "confidence": 0.4},
                {"start_s": 2.7, "end_s": 3.5, "type": "ectopic_like", "beats": 3, "confidence": 0.6},
                {"start_s": 8.0, "end_s": 9.0, "type": "ectopic_like", "beats": 3, "confidence": 0.5},
            ]
        )

        merged = merge_close_same_type_episodes(episodes, merge_gap_s=1.0)

        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(float(merged.iloc[0]["end_s"]), 3.5)
        self.assertEqual(int(merged.iloc[0]["beats"]), 6)


if __name__ == "__main__":
    unittest.main()
