import unittest

import numpy as np

from pasm_rhythm import compute_rhythm_features, run_pasm_rhythm


def make_recording():
    rng = np.random.default_rng(7)
    rr = []
    labels = []

    rr.extend(0.82 + rng.normal(0.0, 0.015, 80))
    labels.extend(["normal"] * 80)

    rr.extend(0.43 + rng.normal(0.0, 0.010, 24))
    labels.extend(["sinus_tachy"] * 24)

    af = 0.78 + rng.normal(0.0, 0.16, 36)
    af = np.clip(af, 0.45, 1.25)
    rr.extend(af)
    labels.extend(["af_like"] * 36)

    ect = 0.82 + rng.normal(0.0, 0.015, 24)
    ect[::3] = 0.48
    ect[1::3] = 1.08
    rr.extend(ect)
    labels.extend(["ectopic_like"] * 24)

    rr.extend(0.82 + rng.normal(0.0, 0.015, 16))
    labels.extend(["noise_uncertain"] * 16)

    rr = np.asarray(rr, dtype=float)
    r_times = np.cumsum(rr)
    rr_prev = rr.copy()
    rr_prev[0] = np.nan
    rr_next = np.r_[rr[1:], np.nan]

    sqi = np.ones_like(rr) * 0.92
    uncertainty = np.zeros_like(rr)
    noise_start = len(rr) - 16
    sqi[noise_start:] = 0.18
    uncertainty[noise_start:] = 0.55

    t = np.linspace(-1.0, 1.0, 90)
    prototype = np.exp(-(t * 8.0) ** 2)
    beats = np.tile(prototype, (len(rr), 1))
    beats += rng.normal(0.0, 0.01, beats.shape)
    ect_start = 80 + 24 + 36
    beats[ect_start : ect_start + 24] += 0.45 * np.exp(-((t - 0.18) * 8.0) ** 2)

    return r_times, rr_prev, rr_next, beats, sqi, uncertainty, np.asarray(labels)


class PASMRhythmTest(unittest.TestCase):
    def test_rhythm_features_include_context(self):
        r_times, rr_prev, rr_next, _, sqi, uncertainty, _ = make_recording()
        features = compute_rhythm_features(
            r_times,
            rr_prev,
            rr_next=rr_next,
            sqi_at_r=sqi,
            rpeak_uncertainty=uncertainty,
            win_beats=10,
        )

        self.assertEqual(len(features), len(r_times))
        for column in ["delta_rr", "rr_ratio", "local_rmssd", "local_cv", "reliability"]:
            self.assertIn(column, features.columns)
        self.assertLess(features["reliability"].iloc[-1], 0.2)

    def test_pasm_detects_synthetic_episode_types(self):
        r_times, rr_prev, rr_next, beats, sqi, uncertainty, labels = make_recording()
        result = run_pasm_rhythm(
            r_times,
            rr_prev,
            rr_next=rr_next,
            beats=beats,
            sqi_at_r=sqi,
            rpeak_uncertainty=uncertainty,
            win_beats=10,
            memory_warmup_beats=70,
        )

        episodes = result["episodes"]
        detected = set(episodes["type"].tolist())

        self.assertIn("sinus_tachy", detected)
        self.assertIn("af_like", detected)
        self.assertIn("ectopic_like", detected)
        self.assertIn("noise_uncertain", detected)
        self.assertGreaterEqual(result["patient_memory"].n_baseline_beats, 30)
        self.assertGreater(len(result["graph"].edges), len(r_times))

        scores = result["state_scores"]
        for target in ["sinus_tachy", "af_like", "ectopic_like", "noise_uncertain"]:
            mask = labels == target
            predicted = scores.loc[mask].idxmax(axis=1)
            hit_rate = np.mean(predicted.to_numpy() == target)
            self.assertGreaterEqual(hit_rate, 0.45, target)


if __name__ == "__main__":
    unittest.main()
