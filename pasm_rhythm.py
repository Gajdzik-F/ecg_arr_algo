from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PASM_STATES = (
    "normal",
    "sinus_tachy",
    "sinus_brady",
    "af_like",
    "ectopic_like",
    "noise_uncertain",
)


DEFAULT_MIN_CONFIDENCE_BY_STATE = {
    "sinus_tachy": 0.30,
    "sinus_brady": 0.30,
    "af_like": 0.42,
    "ectopic_like": 0.30,
    "noise_uncertain": 0.30,
}


@dataclass
class PatientMemory:
    """Patient-specific baseline used by the first PASM-Rhythm prototype."""

    morphology_prototype: Optional[np.ndarray]
    morphology_scale: float
    rr_median: float
    rr_mad: float
    rmssd_median: float
    rmssd_mad: float
    sqi_median: float
    n_baseline_beats: int


@dataclass
class RhythmGraph:
    """Compact typed graph representation for beat-window-state evidence."""

    beat_nodes: pd.DataFrame
    state_scores: pd.DataFrame
    edges: pd.DataFrame


def robust_z(x, center, scale):
    scale = max(float(scale), 1e-12)
    return (np.asarray(x, dtype=float) - float(center)) / scale


def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-12)


def _safe_nanmedian(x, default=np.nan):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return default
    return float(np.median(x))


def compute_rhythm_features(
    r_times,
    rr_prev,
    rr_next=None,
    sqi_at_r=None,
    rpeak_uncertainty=None,
    win_beats=10,
):
    """
    Build per-beat rhythm features used by PASM-Rhythm.

    The function keeps the existing toolkit's R-peak-centric API but adds
    multi-beat context: RR changes, RR ratios, local RMSSD, local CV, and
    signal/reliability features.
    """
    r_times = np.asarray(r_times, dtype=float)
    rr_prev = np.asarray(rr_prev, dtype=float)
    if rr_next is None:
        rr_next = np.full_like(rr_prev, np.nan, dtype=float)
    rr_next = np.asarray(rr_next, dtype=float)

    n = len(r_times)
    if len(rr_prev) != n or len(rr_next) != n:
        raise ValueError("r_times, rr_prev, and rr_next must have the same length.")

    if sqi_at_r is None:
        sqi_at_r = np.ones(n, dtype=float)
    sqi_at_r = np.asarray(sqi_at_r, dtype=float)
    if len(sqi_at_r) != n:
        raise ValueError("sqi_at_r must align 1:1 with r_times.")

    if rpeak_uncertainty is None:
        rpeak_uncertainty = np.zeros(n, dtype=float)
    rpeak_uncertainty = np.asarray(rpeak_uncertainty, dtype=float)
    if len(rpeak_uncertainty) != n:
        raise ValueError("rpeak_uncertainty must align 1:1 with r_times.")

    hr = np.full(n, np.nan, dtype=float)
    valid_rr = np.isfinite(rr_prev) & (rr_prev > 0)
    hr[valid_rr] = 60.0 / rr_prev[valid_rr]

    delta_rr = np.full(n, np.nan, dtype=float)
    if n > 1:
        delta_rr[1:] = rr_prev[1:] - rr_prev[:-1]

    rr_ratio = np.full(n, np.nan, dtype=float)
    valid_ratio = valid_rr & np.isfinite(rr_next) & (rr_next > 0)
    rr_ratio[valid_ratio] = rr_next[valid_ratio] / rr_prev[valid_ratio]

    local_rmssd = np.full(n, np.nan, dtype=float)
    local_cv = np.full(n, np.nan, dtype=float)
    local_rr_median = np.full(n, np.nan, dtype=float)

    for i in range(n):
        a = max(0, i - win_beats + 1)
        seg = rr_prev[a : i + 1]
        seg = seg[np.isfinite(seg) & (seg > 0)]
        if len(seg) >= max(3, win_beats // 2):
            local_rr_median[i] = np.median(seg)
            local_cv[i] = np.std(seg) / (np.mean(seg) + 1e-12)
            if len(seg) >= 3:
                d = np.diff(seg)
                local_rmssd[i] = np.sqrt(np.mean(d * d))

    reliability = np.clip(sqi_at_r * (1.0 - np.clip(rpeak_uncertainty, 0.0, 1.0)), 0.0, 1.0)

    return pd.DataFrame(
        {
            "time_s": r_times,
            "rr_prev": rr_prev,
            "rr_next": rr_next,
            "hr": hr,
            "delta_rr": delta_rr,
            "rr_ratio": rr_ratio,
            "local_rr_median": local_rr_median,
            "local_rmssd": local_rmssd,
            "local_cv": local_cv,
            "sqi": sqi_at_r,
            "rpeak_uncertainty": rpeak_uncertainty,
            "reliability": reliability,
        }
    )


def build_patient_memory(
    beats=None,
    rhythm_features=None,
    min_sqi=0.75,
    max_rpeak_uncertainty=0.25,
    warmup_beats=300,
):
    """
    Estimate a patient-specific normal baseline from high-quality early beats.

    This is deliberately conservative: it prefers a smaller clean baseline over
    a larger contaminated one. Later model versions can replace this with a
    learned patient-memory module.
    """
    if rhythm_features is None:
        raise ValueError("rhythm_features is required.")
    rf = rhythm_features.reset_index(drop=True)
    n = len(rf)
    if n == 0:
        raise ValueError("Cannot build patient memory from an empty recording.")

    baseline_limit = min(n, int(warmup_beats))
    cand = rf.iloc[:baseline_limit]
    high_quality = (cand["sqi"].to_numpy() >= min_sqi) & (
        cand["rpeak_uncertainty"].to_numpy() <= max_rpeak_uncertainty
    )
    valid_rr = np.isfinite(cand["rr_prev"].to_numpy()) & (cand["rr_prev"].to_numpy() > 0)
    idx = np.where(high_quality & valid_rr)[0]

    if len(idx) < max(8, min(30, baseline_limit // 4)):
        idx = np.where(valid_rr)[0]
    if len(idx) == 0:
        idx = np.arange(min(n, baseline_limit))

    rr = cand["rr_prev"].to_numpy()[idx]
    rr = rr[np.isfinite(rr) & (rr > 0)]
    rr_median = _safe_nanmedian(rr, default=1.0)
    rr_mad = _mad(rr)

    rmssd = cand["local_rmssd"].to_numpy()[idx]
    rmssd_median = _safe_nanmedian(rmssd, default=0.0)
    rmssd_mad = _mad(rmssd)

    sqi = cand["sqi"].to_numpy()[idx]
    sqi_median = _safe_nanmedian(sqi, default=1.0)

    morphology_prototype = None
    morphology_scale = 1.0
    if beats is not None:
        beats = np.asarray(beats, dtype=float)
        if len(beats) != n:
            raise ValueError("beats must align 1:1 with rhythm_features.")
        beat_idx = idx[idx < len(beats)]
        if len(beat_idx) > 0:
            baseline_beats = beats[beat_idx]
            morphology_prototype = np.median(baseline_beats, axis=0)
            distances = np.linalg.norm(baseline_beats - morphology_prototype[None, :], axis=1)
            morphology_scale = max(float(np.percentile(distances, 90)), _mad(distances))

    return PatientMemory(
        morphology_prototype=morphology_prototype,
        morphology_scale=float(morphology_scale),
        rr_median=float(rr_median),
        rr_mad=float(rr_mad),
        rmssd_median=float(rmssd_median),
        rmssd_mad=float(rmssd_mad),
        sqi_median=float(sqi_median),
        n_baseline_beats=int(len(idx)),
    )


def score_pasm_states(
    rhythm_features,
    patient_memory,
    beats=None,
    tachy_hr=110.0,
    brady_hr=50.0,
    low_sqi=0.45,
):
    """
    Produce calibrated-looking state scores for the first PASM prototype.

    Scores are deterministic evidence scores, not final clinical probabilities.
    They combine patient-relative RR deviation, local irregularity, morphology
    deviation, and signal reliability.
    """
    rf = rhythm_features.reset_index(drop=True)
    n = len(rf)
    if n == 0:
        return pd.DataFrame(columns=PASM_STATES)

    rr_z = robust_z(rf["rr_prev"].to_numpy(), patient_memory.rr_median, patient_memory.rr_mad)
    delta_rr_z = np.abs(
        robust_z(rf["delta_rr"].fillna(0.0).to_numpy(), 0.0, patient_memory.rr_mad)
    )
    rmssd_z = robust_z(
        rf["local_rmssd"].fillna(patient_memory.rmssd_median).to_numpy(),
        patient_memory.rmssd_median,
        patient_memory.rmssd_mad,
    )
    reliability = rf["reliability"].fillna(0.0).to_numpy()
    hr = rf["hr"].to_numpy()
    rr_ratio = rf["rr_ratio"].to_numpy()

    has_morphology = beats is not None and patient_memory.morphology_prototype is not None
    morph_z = np.zeros(n, dtype=float)
    if has_morphology:
        beats = np.asarray(beats, dtype=float)
        if len(beats) != n:
            raise ValueError("beats must align 1:1 with rhythm_features.")
        dist = np.linalg.norm(beats - patient_memory.morphology_prototype[None, :], axis=1)
        morph_z = robust_z(dist, 0.0, patient_memory.morphology_scale)
        morph_z = np.nan_to_num(morph_z, nan=0.0, posinf=8.0, neginf=0.0)

    tachy_e = np.clip((hr - tachy_hr) / 35.0, 0.0, 3.0)
    brady_e = np.clip((brady_hr - hr) / 25.0, 0.0, 3.0)
    ratio_dev = np.abs(np.nan_to_num(rr_ratio, nan=1.0) - 1.0)
    local_irregular_e = np.clip((rmssd_z - 1.0) / 3.0, 0.0, 3.0)
    current_irregular_e = np.maximum(
        np.clip((ratio_dev - 0.12) / 0.45, 0.0, 2.0),
        np.clip((delta_rr_z - 2.0) / 4.0, 0.0, 2.0),
    )
    irregular_e = np.clip(0.45 * local_irregular_e + 0.75 * current_irregular_e, 0.0, 3.0)
    rr_jump_e = np.clip((delta_rr_z - 2.0) / 3.0, 0.0, 3.0)
    rr_jump_e += np.clip((ratio_dev - 0.2) / 0.8, 0.0, 2.0)
    morph_e = np.clip((morph_z - 2.0) / 3.0, 0.0, 3.0)
    if has_morphology:
        ectopic_e = 1.45 * morph_e + 0.25 * rr_jump_e
    else:
        ectopic_e = rr_jump_e
    noise_e = np.clip((low_sqi - rf["sqi"].to_numpy()) / max(low_sqi, 1e-12), 0.0, 2.0)
    noise_e += np.clip(rf["rpeak_uncertainty"].to_numpy() / 0.5, 0.0, 2.0)

    # Reliability gates true arrhythmia evidence, while poor quality raises the
    # uncertain state. This is the first small step toward SQI as uncertainty.
    arrhythmia_gate = np.clip(reliability, 0.0, 1.0)
    scores = np.column_stack(
        [
            1.2 - 0.10 * np.abs(rr_z) - 0.45 * irregular_e - 0.20 * morph_z - 0.9 * noise_e,
            1.8 * tachy_e * arrhythmia_gate,
            1.8 * brady_e * arrhythmia_gate,
            irregular_e * arrhythmia_gate,
            ectopic_e * arrhythmia_gate,
            noise_e + (1.0 - reliability) * 0.75,
        ]
    )
    scores = np.nan_to_num(scores, nan=-3.0, posinf=3.0, neginf=-3.0)
    probs = _softmax(scores)
    return pd.DataFrame(probs, columns=PASM_STATES)


def build_rhythm_graph(rhythm_features, state_scores, patient_memory):
    rf = rhythm_features.reset_index(drop=True)
    scores = state_scores.reset_index(drop=True)
    if len(rf) != len(scores):
        raise ValueError("rhythm_features and state_scores must align.")

    beat_nodes = rf.copy()
    beat_nodes.insert(0, "node_id", [f"beat_{i}" for i in range(len(rf))])
    beat_nodes["best_state"] = scores.idxmax(axis=1).to_numpy()
    beat_nodes["confidence"] = scores.max(axis=1).to_numpy()
    beat_nodes["patient_rr_z"] = robust_z(
        beat_nodes["rr_prev"].to_numpy(), patient_memory.rr_median, patient_memory.rr_mad
    )

    edges = []
    for i in range(len(rf) - 1):
        edges.append(
            {
                "source": f"beat_{i}",
                "target": f"beat_{i + 1}",
                "relation": "temporal_next",
                "weight": 1.0,
            }
        )
    for i, row in scores.iterrows():
        for state, weight in row.items():
            if weight >= 0.20:
                edges.append(
                    {
                        "source": f"beat_{i}",
                        "target": f"state_{state}",
                        "relation": "state_likelihood",
                        "weight": float(weight),
                    }
                )

    return RhythmGraph(beat_nodes=beat_nodes, state_scores=scores, edges=pd.DataFrame(edges))


def decode_pasm_episodes(
    rhythm_features,
    state_scores,
    min_len_by_state=None,
    min_confidence=0.30,
    min_confidence_by_state=None,
    merge_gap_beats=1,
):
    """
    Decode beat-level state scores into duration-aware rhythm episodes.
    """
    rf = rhythm_features.reset_index(drop=True)
    scores = state_scores.reset_index(drop=True)
    if len(rf) != len(scores):
        raise ValueError("rhythm_features and state_scores must align.")

    if min_len_by_state is None:
        min_len_by_state = {
            "sinus_tachy": 5,
            "sinus_brady": 5,
            "af_like": 8,
            "ectopic_like": 3,
            "noise_uncertain": 2,
        }
    if min_confidence_by_state is None:
        min_confidence_by_state = DEFAULT_MIN_CONFIDENCE_BY_STATE

    best_state = scores.idxmax(axis=1).to_numpy()
    confidence = scores.max(axis=1).to_numpy()
    thresholds = np.array(
        [min_confidence_by_state.get(state, min_confidence) for state in best_state],
        dtype=float,
    )
    candidate = (best_state != "normal") & (confidence >= thresholds)

    runs = []
    start = None
    last = None
    current_state = None
    for i, ok in enumerate(candidate):
        state = best_state[i]
        if ok and (start is None):
            start = i
            last = i
            current_state = state
        elif ok and state == current_state and i <= last + merge_gap_beats + 1:
            last = i
        else:
            if start is not None:
                runs.append((start, last, current_state))
            start = i if ok else None
            last = i if ok else None
            current_state = state if ok else None
    if start is not None:
        runs.append((start, last, current_state))

    episodes = []
    for a, b, state in runs:
        min_len = int(min_len_by_state.get(state, 3))
        if (b - a + 1) < min_len:
            continue
        seg_scores = scores.iloc[a : b + 1]
        seg_rf = rf.iloc[a : b + 1]
        episodes.append(
            {
                "start_s": float(rf.loc[a, "time_s"]),
                "end_s": float(rf.loc[b, "time_s"]),
                "type": state,
                "confidence": float(seg_scores[state].mean()),
                "beats": int(b - a + 1),
                "mean_sqi": float(seg_rf["sqi"].mean()),
                "reason": _episode_reason(state, seg_rf, seg_scores),
            }
        )

    return pd.DataFrame(episodes)


def run_pasm_rhythm(
    r_times,
    rr_prev,
    rr_next=None,
    beats=None,
    sqi_at_r=None,
    rpeak_uncertainty=None,
    win_beats=10,
    memory_warmup_beats=300,
    min_confidence_by_state=None,
    min_len_by_state=None,
):
    """
    End-to-end PASM-Rhythm prototype from beat/RR inputs to episodes.
    """
    features = compute_rhythm_features(
        r_times,
        rr_prev,
        rr_next=rr_next,
        sqi_at_r=sqi_at_r,
        rpeak_uncertainty=rpeak_uncertainty,
        win_beats=win_beats,
    )
    memory = build_patient_memory(
        beats=beats,
        rhythm_features=features,
        warmup_beats=memory_warmup_beats,
    )
    scores = score_pasm_states(features, memory, beats=beats)
    graph = build_rhythm_graph(features, scores, memory)
    episodes = decode_pasm_episodes(
        features,
        scores,
        min_confidence_by_state=min_confidence_by_state,
        min_len_by_state=min_len_by_state,
    )
    return {
        "features": features,
        "patient_memory": memory,
        "state_scores": scores,
        "graph": graph,
        "episodes": episodes,
    }


def _softmax(scores):
    scores = np.asarray(scores, dtype=float)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)


def _episode_reason(state, rf, scores):
    if state == "sinus_tachy":
        return "sustained patient-relative fast rhythm"
    if state == "sinus_brady":
        return "sustained patient-relative slow rhythm"
    if state == "af_like":
        return "high local RR irregularity with acceptable signal quality"
    if state == "ectopic_like":
        return "beat sequence deviates from patient RR or morphology baseline"
    if state == "noise_uncertain":
        return "low SQI or high R-peak uncertainty limits interpretation"
    return "non-normal rhythm state"
