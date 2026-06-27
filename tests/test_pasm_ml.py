import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import pasm_ml_validation
from pasm_dataset import FEATURE_COLUMNS, assign_beat_labels, build_pasm_feature_frame
from pasm_ml_decoder import _balanced_sample_weights, fit_softmax_decoder
from pasm_physionet import PhysioNetRecord
from pasm_rhythm import PASM_STATES, PatientMemory


def fake_record(record_id, state="af_like"):
    return PhysioNetRecord(
        record_id=record_id,
        fs=100.0,
        signal=np.zeros(1000),
        rpeaks=np.arange(10) * 100,
        truth_episodes=pd.DataFrame([{"start_s": 2.0, "end_s": 5.0, "type": state}]),
    )


def fake_pipeline(record):
    n = 10
    times = np.arange(n, dtype=float)
    state = str(record.truth_episodes.iloc[0]["type"])
    in_episode = (times >= 2.0) & (times <= 5.0)
    rr = np.r_[np.nan, np.ones(n - 1) * 0.8]
    features = pd.DataFrame(
        {
            "time_s": times,
            "rr_prev": rr,
            "rr_next": np.r_[np.ones(n - 1) * 0.8, np.nan],
            "hr": np.where(np.isfinite(rr), 60.0 / np.nan_to_num(rr, nan=0.8), np.nan),
            "delta_rr": np.r_[np.nan, np.diff(np.nan_to_num(rr, nan=0.8))],
            "rr_ratio": np.ones(n),
            "local_rr_median": np.ones(n) * 0.8,
            "local_rmssd": np.where(in_episode, 0.18, 0.02),
            "local_cv": np.where(in_episode, 0.25, 0.02),
            "sqi": np.ones(n),
            "rpeak_uncertainty": np.zeros(n),
            "reliability": np.ones(n),
        }
    )
    scores = pd.DataFrame(0.02, index=np.arange(n), columns=PASM_STATES)
    scores.loc[~in_episode, "normal"] = 0.90
    scores.loc[in_episode, state] = 0.90
    beats = np.tile(np.array([0.0, 1.0, 0.0, 0.2]), (n, 1))
    beats[in_episode] += 0.4
    return {
        "episodes": record.truth_episodes.copy(),
        "features": features,
        "state_scores": scores,
        "patient_memory": PatientMemory(
            morphology_prototype=np.array([0.0, 1.0, 0.0, 0.2]),
            morphology_scale=0.5,
            rr_median=0.8,
            rr_mad=0.05,
            rmssd_median=0.02,
            rmssd_mad=0.02,
            sqi_median=1.0,
            n_baseline_beats=4,
        ),
        "beats": beats,
    }


class PASMMLTest(unittest.TestCase):
    def test_ml_presets_include_current_tiny_split(self):
        train, holdout = pasm_ml_validation.resolve_ml_preset("tiny")

        self.assertEqual(train, pasm_ml_validation.DEFAULT_TRAIN_RECORD_IDS)
        self.assertEqual(holdout, pasm_ml_validation.DEFAULT_HOLDOUT_RECORD_IDS)
        self.assertIn("mini", pasm_ml_validation.ML_BENCHMARK_PRESETS)
        self.assertIn("mitdb-mini", pasm_ml_validation.ML_BENCHMARK_PRESETS)
        self.assertIn("afdb-mini", pasm_ml_validation.ML_BENCHMARK_PRESETS)

    def test_assign_beat_labels_marks_truth_episode(self):
        labels = assign_beat_labels(
            np.arange(7, dtype=float),
            pd.DataFrame([{"start_s": 2.0, "end_s": 4.0, "type": "af_like"}]),
        )

        self.assertEqual(labels.tolist(), ["normal", "normal", "af_like", "af_like", "af_like", "normal", "normal"])

    def test_feature_frame_has_expected_columns(self):
        record = fake_record("afdb/00001", state="af_like")
        frame = build_pasm_feature_frame(record, split="train", pipeline=fake_pipeline(record))

        for column in FEATURE_COLUMNS:
            self.assertIn(column, frame.columns)
        self.assertEqual(set(frame["split"]), {"train"})
        self.assertIn("af_like", set(frame["label"]))
        self.assertIn("normal", set(frame["label"]))

    def test_softmax_decoder_learns_separable_toy_data(self):
        rows = []
        for i in range(40):
            row = {col: 0.0 for col in FEATURE_COLUMNS}
            row["label"] = "normal" if i < 20 else "af_like"
            row["local_rmssd"] = 0.01 if i < 20 else 1.0
            row["score_normal"] = 0.9 if i < 20 else 0.1
            row["score_af_like"] = 0.1 if i < 20 else 0.9
            rows.append(row)
        train = pd.DataFrame(rows)

        model = fit_softmax_decoder(train, FEATURE_COLUMNS, epochs=300, lr=0.1, seed=7)
        probs = model.predict_proba(train)

        self.assertTrue(np.allclose(probs.sum(axis=1).to_numpy(), 1.0))
        self.assertGreater(float(probs.iloc[0]["normal"]), 0.5)
        self.assertGreater(float(probs.iloc[-1]["af_like"]), 0.5)

    def test_capped_class_weights_do_not_exceed_limit(self):
        y = np.array([0] * 100 + [1], dtype=int)
        weights = _balanced_sample_weights(y, n_classes=2, max_class_weight=3.0)

        self.assertLessEqual(float(weights.max()), 3.0)
        self.assertGreater(float(weights[-1]), float(weights[0]))

    def test_hard_negative_boost_changes_normal_sample_weight_effect(self):
        rows = []
        for i in range(20):
            row = {col: 0.0 for col in FEATURE_COLUMNS}
            row["label"] = "normal"
            row["score_normal"] = 0.6
            row["score_ectopic_like"] = 0.4
            rows.append(row)
        for i in range(5):
            row = {col: 0.0 for col in FEATURE_COLUMNS}
            row["label"] = "ectopic_like"
            row["score_normal"] = 0.1
            row["score_ectopic_like"] = 0.9
            rows.append(row)
        train = pd.DataFrame(rows)
        boost = np.ones(len(train), dtype=float)
        boost[:10] = 4.0

        plain = fit_softmax_decoder(train, FEATURE_COLUMNS, epochs=80, lr=0.05, seed=11)
        boosted = fit_softmax_decoder(train, FEATURE_COLUMNS, epochs=80, lr=0.05, seed=11, sample_weight_boost=boost)

        self.assertGreaterEqual(
            float(boosted.predict_proba(train.iloc[:10]).mean()["normal"]),
            float(plain.predict_proba(train.iloc[:10]).mean()["normal"]),
        )

    def test_guarded_decoder_removes_weak_false_positive_candidates(self):
        episodes = pd.DataFrame(
            [
                {"start_s": 1.0, "end_s": 5.0, "type": "af_like", "confidence": 0.40, "mean_sqi": 1.0, "beats": 20},
                {"start_s": 10.0, "end_s": 20.0, "type": "af_like", "confidence": 0.80, "mean_sqi": 1.0, "beats": 20},
                {"start_s": 30.0, "end_s": 31.0, "type": "ectopic_like", "confidence": 0.90, "mean_sqi": 1.0, "beats": 2},
            ]
        )

        guarded = pasm_ml_validation.guard_ml_episodes(
            episodes,
            {
                "min_episode_confidence": 0.55,
                "min_episode_sqi": 0.50,
                "min_beats_by_state": {"af_like": 12, "ectopic_like": 4},
            },
        )

        self.assertEqual(len(guarded), 1)
        self.assertAlmostEqual(float(guarded.iloc[0]["confidence"]), 0.80)

    def test_ectopy_specific_guard_requires_rr_or_morphology_support(self):
        episodes = pd.DataFrame(
            [
                {
                    "start_s": 1.0,
                    "end_s": 4.0,
                    "type": "ectopic_like",
                    "confidence": 0.90,
                    "mean_sqi": 1.0,
                    "beats": 8,
                    "max_morph_z": 0.10,
                    "max_delta_rr_z_abs": 0.20,
                    "max_score_ectopic_like": 0.20,
                },
                {
                    "start_s": 6.0,
                    "end_s": 9.0,
                    "type": "ectopic_like",
                    "confidence": 0.90,
                    "mean_sqi": 1.0,
                    "beats": 8,
                    "max_morph_z": 0.80,
                    "max_delta_rr_z_abs": 0.20,
                    "max_score_ectopic_like": 0.20,
                },
            ]
        )

        guarded, removed = pasm_ml_validation.guard_ml_episodes_with_report(episodes)

        self.assertEqual(len(guarded), 1)
        self.assertEqual(removed["reason"].tolist(), ["weak_ectopy_support"])

    def test_guard_tuning_prefers_lower_false_alarm_score(self):
        def fake_eval(records, pipelines, model, thresholds=None, model_name="pasm_ml_decoder_fpaware", guarded_config=None):
            normal_bias = guarded_config["normal_bias"]
            faph = 2.0 if normal_bias == 0.40 else 10.0
            return pd.DataFrame(
                [
                    {
                        "record_id": "mock",
                        "model": model_name,
                        "type": "macro",
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": 0.8,
                        "recall": 0.8,
                        "f1": 0.8,
                        "mean_iou": np.nan,
                    },
                    {
                        "record_id": "mock",
                        "model": model_name,
                        "type": "false_alarms_per_hour",
                        "tp": 1,
                        "fp": 0,
                        "fn": 0,
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": faph,
                        "mean_iou": np.nan,
                    },
                ]
            )

        with mock.patch.object(pasm_ml_validation, "evaluate_ml_records", side_effect=fake_eval):
            cfg, _ = pasm_ml_validation.tune_guarded_config([], {}, object(), {})

        self.assertEqual(cfg["normal_bias"], 0.40)

    def test_ml_validation_report_with_mocked_loaders(self):
        def load_mitdb(name, max_seconds=None):
            return fake_record(f"mitdb/{name}", state="ectopic_like")

        def load_afdb(name, max_seconds=None):
            return fake_record(f"afdb/{name}", state="af_like")

        with mock.patch.object(pasm_ml_validation, "load_mitdb_record", side_effect=load_mitdb):
            with mock.patch.object(pasm_ml_validation, "load_afdb_record", side_effect=load_afdb):
                with mock.patch.object(pasm_ml_validation, "run_pasm_physionet_pipeline", side_effect=fake_pipeline):
                    result = pasm_ml_validation.run_ml_validation(
                        preset=None,
                        train_record_ids=["mitdb/200", "afdb/04015"],
                        holdout_record_ids=["mitdb/201", "afdb/04126"],
                        epochs=20,
                        lr=0.05,
                    )

        self.assertIn("pasm_ml_decoder", set(result["holdout_summary"]["model"]))
        self.assertIn("pasm_ml_decoder_guarded", set(result["holdout_summary"]["model"]))
        self.assertIn("pasm_ml_decoder_fpaware", set(result["holdout_summary"]["model"]))
        self.assertIn("pasm_physionet", set(result["holdout_summary"]["model"]))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ml.md"
            pasm_ml_validation.write_ml_validation_report(out, result)
            text = out.read_text(encoding="utf-8")

        self.assertIn("PASM-Rhythm ML Validation", text)
        self.assertIn("Patient-Wise Split", text)
        self.assertIn("pasm_ml_decoder", text)
        self.assertIn("pasm_ml_decoder_fpaware", text)
        self.assertIn("Tuned Guard Config", text)
        self.assertIn("FP Removed By Guard Reason", text)
        self.assertIn("Holdout False Positives By Type", text)
        self.assertIn("Top Holdout False-Positive Episodes", text)


if __name__ == "__main__":
    unittest.main()
