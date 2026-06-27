import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from beats import extract_beats
from preprocess import bandpass_filter, robust_normalize
from pasm_rhythm import DEFAULT_MIN_CONFIDENCE_BY_STATE, run_pasm_rhythm
from pasm_validation import evaluate_episodes, episode_iou, normalize_episode_types, summarize_benchmark


MITDB_NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
MITDB_ECTOPIC_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F"}
QRS_SYMBOLS = MITDB_NORMAL_SYMBOLS | MITDB_ECTOPIC_SYMBOLS | {
    "/",
    "f",
    "Q",
    "?",
    "!",
    "x",
    "n",
}

RHYTHM_AUX_TO_STATE = {
    "(N": "normal",
    "(NSR": "normal",
    "(SBR": "sinus_brady",
    "(B": "sinus_brady",
    "(SVTA": "sinus_tachy",
    "(T": "sinus_tachy",
    "(VT": "sinus_tachy",
    "(AFIB": "af_like",
    "(AFL": "af_like",
    "(J": "ectopic_like",
    "(BII": "ectopic_like",
    "(PREX": "ectopic_like",
}

PHYSIONET_AF_MERGE_GAP_S = 45.0
PHYSIONET_AF_TACHY_MARGIN_S = 10.0
PHYSIONET_MIN_TACHY_DURATION_S = 3.0
ECTOPY_SHORT_RR_S = 0.50
ECTOPY_RELATIVE_RR_FRACTION = 0.75
ECTOPY_MERGE_GAP_S = 1.0
ECTOPY_FLOOD_RATE_PER_HOUR = 30.0
ECTOPY_FLOOD_MIN_CONFIDENCE = 0.40
ECTOPY_FLOOD_DENSITY_WINDOW_S = 10.0
ECTOPY_FLOOD_MIN_DENSITY = 6
ECTOPY_FLOOD_MIN_CANDIDATES = 10
ECTOPY_FLOOD_STRONG_MORPH_Z = 0.55
ECTOPY_FLOOD_DENSE_MORPH_Z = 0.60


@dataclass
class PhysioNetRecord:
    record_id: str
    fs: float
    signal: np.ndarray
    rpeaks: np.ndarray
    truth_episodes: pd.DataFrame


def require_wfdb():
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError(
            "WFDB is required for PhysioNet validation. Install it with: "
            "py -3.11 -m pip install wfdb"
        ) from exc
    return wfdb


def map_mitdb_symbols_to_labels(symbols):
    labels = []
    for symbol in symbols:
        if symbol in MITDB_ECTOPIC_SYMBOLS:
            labels.append("ectopic_like")
        elif symbol in MITDB_NORMAL_SYMBOLS:
            labels.append("normal")
        else:
            labels.append("normal")
    return np.asarray(labels, dtype=object)


def rhythm_aux_to_episodes(samples, aux_notes, fs, end_sample):
    """
    Convert PhysioNet rhythm-change aux notes into PASM episode intervals.

    Rhythm annotations describe a state from an annotation sample until the next
    rhythm annotation. Normal intervals are omitted from the returned episodes.
    """
    events = []
    for sample, aux in zip(samples, aux_notes):
        aux = (aux or "").strip()
        if not aux:
            continue
        if aux in RHYTHM_AUX_TO_STATE:
            events.append((int(sample), RHYTHM_AUX_TO_STATE[aux]))

    episodes = []
    for i, (sample, state) in enumerate(events):
        next_sample = events[i + 1][0] if i + 1 < len(events) else int(end_sample)
        if next_sample <= sample or state == "normal":
            continue
        episodes.append(
            {
                "start_s": float(sample) / float(fs),
                "end_s": float(next_sample) / float(fs),
                "type": state,
            }
        )
    return pd.DataFrame(episodes)


def beat_labels_to_episodes(rpeaks, labels, fs, min_len=3):
    rpeaks = np.asarray(rpeaks, dtype=int)
    labels = np.asarray(labels, dtype=object)
    episodes = []
    start = None
    current = None
    for i, label in enumerate(labels):
        if label != "normal" and start is None:
            start = i
            current = label
        elif start is not None and label != current:
            if (i - start) >= min_len:
                episodes.append(
                    {
                        "start_s": float(rpeaks[start]) / float(fs),
                        "end_s": float(rpeaks[i - 1]) / float(fs),
                        "type": current,
                    }
                )
            start = i if label != "normal" else None
            current = label if label != "normal" else None
    if start is not None and (len(labels) - start) >= min_len:
        episodes.append(
            {
                "start_s": float(rpeaks[start]) / float(fs),
                "end_s": float(rpeaks[-1]) / float(fs),
                "type": current,
            }
        )
    return pd.DataFrame(episodes)


def load_mitdb_record(record_name, max_seconds=None, pn_dir="mitdb"):
    wfdb = require_wfdb()
    sampto = None
    if max_seconds is not None:
        # MIT-BIH Arrhythmia Database is 360 Hz, but we read fs afterward too.
        sampto = int(float(max_seconds) * 360)

    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=sampto)
    ann = wfdb.rdann(record_name, "atr", pn_dir=pn_dir, sampto=sampto)
    signal = np.asarray(record.p_signal[:, 0], dtype=float)
    fs = float(record.fs)

    samples = np.asarray(ann.sample, dtype=int)
    symbols = np.asarray(ann.symbol, dtype=object)
    beat_mask = np.array([s in QRS_SYMBOLS for s in symbols], dtype=bool)
    rpeaks = samples[beat_mask]
    labels = map_mitdb_symbols_to_labels(symbols[beat_mask])
    truth = beat_labels_to_episodes(rpeaks, labels, fs, min_len=3)
    return PhysioNetRecord(
        record_id=f"{pn_dir}/{record_name}",
        fs=fs,
        signal=signal,
        rpeaks=rpeaks,
        truth_episodes=truth,
    )


def load_afdb_record(record_name, max_seconds=None, pn_dir="afdb"):
    wfdb = require_wfdb()
    sampto = None
    if max_seconds is not None:
        # MIT-BIH AFDB is 250 Hz for the public records.
        sampto = int(float(max_seconds) * 250)

    record = wfdb.rdrecord(record_name, pn_dir=pn_dir, sampto=sampto)
    rhythm_ann = wfdb.rdann(record_name, "atr", pn_dir=pn_dir, sampto=sampto)
    qrs_ann = wfdb.rdann(record_name, "qrs", pn_dir=pn_dir, sampto=sampto)
    signal = np.asarray(record.p_signal[:, 0], dtype=float)
    fs = float(record.fs)

    rpeaks = np.asarray(qrs_ann.sample, dtype=int)
    end_sample = len(signal) if sampto is None else min(int(sampto), len(signal))
    truth = rhythm_aux_to_episodes(rhythm_ann.sample, rhythm_ann.aux_note, fs, end_sample=end_sample)
    return PhysioNetRecord(
        record_id=f"{pn_dir}/{record_name}",
        fs=fs,
        signal=signal,
        rpeaks=rpeaks,
        truth_episodes=truth,
    )


def run_pasm_on_physionet_record(record, thresholds=None):
    return run_pasm_physionet_pipeline(record, thresholds=thresholds)["episodes"]


def run_pasm_physionet_pipeline(record, thresholds=None):
    signal = prepare_ecg_for_pasm(record.signal, record.fs)
    beats, beat_r = extract_beats(signal, record.rpeaks, record.fs)
    if len(beat_r) < 3:
        raise ValueError(f"{record.record_id} has too few usable beats after extraction.")

    beat_r = np.asarray(beat_r, dtype=int)
    r_times = beat_r / record.fs
    rr_prev = np.full(len(beat_r), np.nan, dtype=float)
    rr_next = np.full(len(beat_r), np.nan, dtype=float)
    if len(beat_r) >= 2:
        rr = np.diff(r_times)
        rr_prev[1:] = rr
        rr_next[:-1] = rr

    pasm_result = run_pasm_rhythm(
        r_times,
        rr_prev,
        rr_next=rr_next,
        beats=beats,
        sqi_at_r=np.ones(len(beat_r), dtype=float),
        rpeak_uncertainty=np.zeros(len(beat_r), dtype=float),
        memory_warmup_beats=min(300, max(20, len(beat_r) // 5)),
        min_confidence_by_state=thresholds or DEFAULT_MIN_CONFIDENCE_BY_STATE,
    )
    af_evidence = detect_fast_irregular_af(pasm_result["features"])
    extra_evidence = [af_evidence]
    ectopy_evidence = pd.DataFrame()
    if record.record_id.startswith("mitdb/"):
        ectopy_evidence = detect_short_coupled_ectopy(
            pasm_result["features"],
            beats=beats,
            patient_memory=pasm_result["patient_memory"],
        )
        extra_evidence.append(ectopy_evidence)
    merged = merge_physionet_evidence(pasm_result["episodes"], extra_evidence)
    scoped = filter_predictions_for_annotation_scope(record, merged)
    return {
        "episodes": scoped,
        "raw_episodes": pasm_result["episodes"],
        "af_evidence": af_evidence,
        "ectopy_evidence": ectopy_evidence,
        "features": pasm_result["features"],
        "patient_memory": pasm_result["patient_memory"],
        "beats": beats,
        "beat_r": beat_r,
        "r_times": r_times,
        "rr_prev": rr_prev,
        "rr_next": rr_next,
        "state_scores": pasm_result["state_scores"],
    }


def filter_predictions_for_annotation_scope(record, episodes):
    """
    Keep predictions that are supported by the annotation scope of each loader.

    MITDB records in this harness currently use beat symbols to build short
    ectopic-like truth episodes. Rhythm states such as AF or sinus tachycardia
    would be counted as false positives only because this loader does not expose
    rhythm-level truth for them.
    """
    if episodes is None or len(episodes) == 0:
        return episodes
    if record.record_id.startswith("mitdb/"):
        return episodes[episodes["type"] == "ectopic_like"].reset_index(drop=True)
    return episodes


def prepare_ecg_for_pasm(signal, fs):
    signal = np.asarray(signal, dtype=float)
    try:
        return robust_normalize(bandpass_filter(signal, fs, low=0.5, high=40.0))
    except Exception:
        return robust_normalize(signal)


def detect_fast_irregular_af(features, win_beats=30, min_beats=25, merge_gap_s=12.0):
    """Detect long fast-irregular rhythm windows common in AFDB annotations."""
    if len(features) == 0:
        return pd.DataFrame(columns=["start_s", "end_s", "type", "confidence", "beats", "mean_sqi", "reason"])

    hr = features["hr"].to_numpy(dtype=float)
    cv = features["local_cv"].fillna(0.0).to_numpy(dtype=float)
    rmssd = features["local_rmssd"].fillna(0.0).to_numpy(dtype=float)
    times = features["time_s"].to_numpy(dtype=float)
    flags = np.zeros(len(features), dtype=bool)
    conf = np.zeros(len(features), dtype=float)

    for i in range(len(features)):
        a = max(0, i - win_beats + 1)
        if i - a + 1 < max(8, win_beats // 2):
            continue
        h = float(np.nanmean(hr[a : i + 1]))
        c = float(np.nanmean(cv[a : i + 1]))
        r = float(np.nanmean(rmssd[a : i + 1]))
        evidence = min(1.0, max(0.0, (h - 105.0) / 55.0)) * min(1.0, max(0.0, (c - 0.10) / 0.18))
        evidence = max(evidence, min(1.0, max(0.0, (r - 0.07) / 0.16)) * min(1.0, max(0.0, (h - 110.0) / 50.0)))
        if h >= 110.0 and c >= 0.12 and r >= 0.07 and evidence >= 0.18:
            flags[i] = True
            conf[i] = evidence

    runs = []
    start = None
    for i, ok in enumerate(flags):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_beats:
                runs.append([start, i - 1])
            start = None
    if start is not None and len(flags) - start >= min_beats:
        runs.append([start, len(flags) - 1])

    merged = []
    for run in runs:
        if merged and times[run[0]] - times[merged[-1][1]] <= merge_gap_s:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    episodes = []
    for a, b in merged:
        if times[b] - times[a] < 15.0:
            continue
        episodes.append(
            {
                "start_s": float(times[a]),
                "end_s": float(times[b]),
                "type": "af_like",
                "confidence": float(np.mean(conf[a : b + 1][flags[a : b + 1]])),
                "beats": int(b - a + 1),
                "mean_sqi": float(features["sqi"].iloc[a : b + 1].mean()),
                "reason": "fast irregular rhythm window evidence",
            }
        )
    return pd.DataFrame(episodes)


def detect_short_coupled_ectopy(
    features,
    short_rr_s=ECTOPY_SHORT_RR_S,
    relative_rr_fraction=ECTOPY_RELATIVE_RR_FRACTION,
    max_following_rr_s=1.05,
    merge_gap_s=ECTOPY_MERGE_GAP_S,
    beats=None,
    patient_memory=None,
    min_morph_z=None,
    flood_rate_per_hour=ECTOPY_FLOOD_RATE_PER_HOUR,
    flood_min_confidence=ECTOPY_FLOOD_MIN_CONFIDENCE,
    flood_density_window_s=ECTOPY_FLOOD_DENSITY_WINDOW_S,
    flood_min_density=ECTOPY_FLOOD_MIN_DENSITY,
    flood_min_candidates=ECTOPY_FLOOD_MIN_CANDIDATES,
):
    """
    Detect short runs of closely coupled ectopic-like beats.

    This targets MITDB-style beat-level runs such as V-V-V, where the event is
    too short for a rhythm-state decoder but still clinically meaningful.
    """
    if len(features) < 3:
        return pd.DataFrame(columns=["start_s", "end_s", "type", "confidence", "beats", "mean_sqi", "reason"])

    rr_prev = features["rr_prev"].to_numpy(dtype=float)
    times = features["time_s"].to_numpy(dtype=float)
    sqi = features["sqi"].to_numpy(dtype=float)
    reference_rr = short_rr_s / relative_rr_fraction
    if patient_memory is not None and np.isfinite(patient_memory.rr_median) and patient_memory.rr_median > 0:
        reference_rr = float(patient_memory.rr_median)
    relative_short_rr_s = min(float(short_rr_s), float(reference_rr) * float(relative_rr_fraction))

    morph_z = np.zeros(len(features), dtype=float)
    has_morphology = (
        beats is not None
        and patient_memory is not None
        and patient_memory.morphology_prototype is not None
        and patient_memory.morphology_scale > 0
    )
    if has_morphology:
        beats = np.asarray(beats, dtype=float)
        if len(beats) == len(features):
            dist = np.linalg.norm(beats - patient_memory.morphology_prototype[None, :], axis=1)
            morph_z = np.nan_to_num(dist / float(patient_memory.morphology_scale), nan=0.0, posinf=8.0)
        else:
            has_morphology = False

    episodes = []
    i = 1
    while i < len(features) - 2:
        two_short = rr_prev[i] <= relative_short_rr_s and rr_prev[i + 1] <= relative_short_rr_s
        follows_without_long_pause = rr_prev[i + 2] < max_following_rr_s
        morphology_support = (
            min_morph_z is None
            or (not has_morphology)
            or float(np.nanmean(morph_z[i : i + 3])) >= float(min_morph_z)
        )
        very_short_support = rr_prev[i] < 0.46 and rr_prev[i + 1] < 0.46
        if two_short and follows_without_long_pause and (morphology_support or very_short_support):
            start = i
            end = i + 2
            confidence = float(
                np.clip(
                    ((relative_short_rr_s - rr_prev[i]) + (relative_short_rr_s - rr_prev[i + 1]))
                    / max(relative_short_rr_s, 1e-12),
                    0.0,
                    1.0,
                )
            )
            episodes.append(
                {
                    "start_s": float(times[start]),
                    "end_s": float(times[end]),
                    "type": "ectopic_like",
                    "confidence": confidence,
                    "beats": int(end - start + 1),
                    "mean_sqi": float(np.nanmean(sqi[start : end + 1])),
                    "mean_rr_prev": float(np.nanmean(rr_prev[start : end + 1])),
                    "min_rr_prev": float(np.nanmin(rr_prev[start : end + 1])),
                    "max_rr_prev": float(np.nanmax(rr_prev[start : end + 1])),
                    "mean_rr_next": float(
                        np.nanmean(features["rr_next"].to_numpy(dtype=float)[start : end + 1])
                        if "rr_next" in features
                        else np.nan
                    ),
                    "local_cv": float(np.nanmean(features["local_cv"].iloc[start : end + 1]))
                    if "local_cv" in features
                    else np.nan,
                    "local_rmssd": float(np.nanmean(features["local_rmssd"].iloc[start : end + 1]))
                    if "local_rmssd" in features
                    else np.nan,
                    "mean_morph_z": float(np.nanmean(morph_z[start : end + 1])) if has_morphology else np.nan,
                    "flood_filtered": False,
                    "reason": "short-coupled ectopic run evidence",
                }
            )
            i = end + 1
        else:
            i += 1
    episodes = pd.DataFrame(episodes)
    episodes = filter_ectopy_candidate_flood(
        episodes,
        recording_duration_s=float(times[-1] - times[0]) if len(times) else 0.0,
        flood_rate_per_hour=flood_rate_per_hour,
        flood_min_confidence=flood_min_confidence,
        flood_density_window_s=flood_density_window_s,
        flood_min_density=flood_min_density,
        flood_min_candidates=flood_min_candidates,
    )
    return merge_close_same_type_episodes(episodes, merge_gap_s=merge_gap_s)


def filter_ectopy_candidate_flood(
    episodes,
    recording_duration_s,
    flood_rate_per_hour=ECTOPY_FLOOD_RATE_PER_HOUR,
    flood_min_confidence=ECTOPY_FLOOD_MIN_CONFIDENCE,
    flood_density_window_s=ECTOPY_FLOOD_DENSITY_WINDOW_S,
    flood_min_density=ECTOPY_FLOOD_MIN_DENSITY,
    flood_min_candidates=ECTOPY_FLOOD_MIN_CANDIDATES,
    flood_strong_morph_z=ECTOPY_FLOOD_STRONG_MORPH_Z,
    flood_dense_morph_z=ECTOPY_FLOOD_DENSE_MORPH_Z,
):
    if episodes is None or len(episodes) == 0:
        return episodes

    out = episodes.copy().sort_values(["start_s", "end_s"]).reset_index(drop=True)
    starts = out["start_s"].to_numpy(dtype=float)
    density = np.array(
        [int(np.sum(np.abs(starts - start) <= float(flood_density_window_s))) for start in starts],
        dtype=int,
    )
    out["candidate_density"] = density
    duration_hours = max(float(recording_duration_s) / 3600.0, 1e-12)
    candidate_rate = len(out) / duration_hours
    out["candidate_rate_per_hour"] = float(candidate_rate)

    if len(out) < int(flood_min_candidates) or candidate_rate <= float(flood_rate_per_hour):
        return out

    strong = out["confidence"].astype(float) >= float(flood_min_confidence)
    dense = out["candidate_density"].astype(int) >= int(flood_min_density)
    if "mean_morph_z" in out:
        morph = out["mean_morph_z"].astype(float)
        morph_known = np.isfinite(morph)
        strong = strong & (~morph_known | (morph >= float(flood_strong_morph_z)))
        dense = dense & (~morph_known | (morph >= float(flood_dense_morph_z)))
    keep = strong | dense
    out.loc[~keep, "flood_filtered"] = True
    return out.loc[keep].reset_index(drop=True)


def diagnose_physionet_record(record, thresholds=None, iou_threshold=0.30):
    pipeline = run_pasm_physionet_pipeline(record, thresholds=thresholds)
    predicted = normalize_episode_types(pipeline["episodes"])
    truth = normalize_episode_types(record.truth_episodes)
    rows = match_diagnostic_episodes(predicted, truth, pipeline, iou_threshold=iou_threshold)
    return {
        "record": record,
        "pipeline": pipeline,
        "diagnostics": rows,
        "predicted": predicted,
        "truth": truth,
        "iou_threshold": float(iou_threshold),
    }


def match_diagnostic_episodes(predicted, truth, pipeline=None, iou_threshold=0.30):
    predicted = normalize_episode_types(predicted)
    truth = normalize_episode_types(truth)
    pairs = []
    for pi, pred in predicted.iterrows():
        for ti, tr in truth.iterrows():
            if pred.get("type") != tr.get("type"):
                continue
            iou = episode_iou(pred, tr)
            if iou >= iou_threshold:
                pairs.append((iou, pi, ti))
    pairs.sort(reverse=True)

    matched_pred = set()
    matched_truth = set()
    pred_to_match = {}
    for iou, pi, ti in pairs:
        if pi in matched_pred or ti in matched_truth:
            continue
        matched_pred.add(pi)
        matched_truth.add(ti)
        pred_to_match[pi] = (ti, iou)

    rows = []
    for pi, pred in predicted.iterrows():
        ti, iou = pred_to_match.get(pi, (None, 0.0))
        row = _diagnostic_row("TP" if ti is not None else "FP", pred, pipeline=pipeline)
        row["matched_truth_index"] = "" if ti is None else int(ti)
        row["iou"] = float(iou)
        rows.append(row)

    for ti, tr in truth.iterrows():
        if ti in matched_truth:
            continue
        row = _diagnostic_row("FN", tr, pipeline=pipeline)
        row["matched_truth_index"] = int(ti)
        row["iou"] = 0.0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["status", "start_s", "end_s", "type", "iou"])
    return pd.DataFrame(rows).sort_values(["start_s", "status"]).reset_index(drop=True)


def _diagnostic_row(status, episode, pipeline=None):
    start_s = float(episode.get("start_s", np.nan))
    end_s = float(episode.get("end_s", np.nan))
    row = {
        "status": status,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s if np.isfinite(start_s) and np.isfinite(end_s) else np.nan,
        "type": episode.get("type", ""),
        "confidence": episode.get("confidence", np.nan),
        "beats": episode.get("beats", np.nan),
        "candidate_density": episode.get("candidate_density", np.nan),
        "candidate_rate_per_hour": episode.get("candidate_rate_per_hour", np.nan),
        "mean_rr_prev": episode.get("mean_rr_prev", np.nan),
        "min_rr_prev": episode.get("min_rr_prev", np.nan),
        "max_rr_prev": episode.get("max_rr_prev", np.nan),
        "mean_rr_next": episode.get("mean_rr_next", np.nan),
        "local_cv": episode.get("local_cv", np.nan),
        "local_rmssd": episode.get("local_rmssd", np.nan),
        "mean_morph_z": episode.get("mean_morph_z", np.nan),
        "reason": episode.get("reason", ""),
    }
    if pipeline is not None and np.isfinite(start_s):
        row.update(_feature_context_at_time(start_s, pipeline))
    return row


def _feature_context_at_time(time_s, pipeline):
    features = pipeline.get("features")
    if features is None or len(features) == 0:
        return {}
    times = features["time_s"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(times - float(time_s))))
    out = {}
    for col in ["rr_prev", "rr_next", "local_cv", "local_rmssd"]:
        if col in features:
            out[f"context_{col}"] = float(features.iloc[idx][col])
    return out


def merge_close_same_type_episodes(episodes, merge_gap_s=1.0):
    if episodes is None or len(episodes) == 0:
        return episodes
    episodes = episodes.sort_values(["start_s", "end_s"]).reset_index(drop=True)
    merged = []
    current = episodes.iloc[0].to_dict()
    for _, row in episodes.iloc[1:].iterrows():
        same_type = row.get("type") == current.get("type")
        close = float(row["start_s"]) - float(current["end_s"]) <= float(merge_gap_s)
        if same_type and close:
            current["end_s"] = float(row["end_s"])
            current["confidence"] = max(float(current.get("confidence", 0.0)), float(row.get("confidence", 0.0)))
            current["beats"] = int(current.get("beats", 0)) + int(row.get("beats", 0))
            mean_sqi_values = [
                float(v)
                for v in [current.get("mean_sqi", np.nan), row.get("mean_sqi", np.nan)]
                if np.isfinite(v)
            ]
            current["mean_sqi"] = float(np.mean(mean_sqi_values)) if mean_sqi_values else np.nan
        else:
            merged.append(current)
            current = row.to_dict()
    merged.append(current)
    return pd.DataFrame(merged)


def merge_physionet_evidence(episodes, evidence):
    if episodes is None or len(episodes) == 0:
        base = pd.DataFrame(columns=["start_s", "end_s", "type", "confidence", "beats", "mean_sqi", "reason"])
    else:
        base = episodes.copy()

    if evidence is None:
        return base
    if isinstance(evidence, pd.DataFrame):
        evidence_frames = [evidence]
    else:
        evidence_frames = [frame for frame in evidence if frame is not None and len(frame) > 0]
    if not evidence_frames:
        return base
    evidence_all = pd.concat(evidence_frames, ignore_index=True)
    af_evidence = evidence_all[evidence_all["type"] == "af_like"] if "type" in evidence_all else pd.DataFrame()

    keep = []
    for _, row in base.iterrows():
        if row.get("type") != "sinus_tachy":
            keep.append(True)
            continue
        overlaps_af = False
        for _, af in af_evidence.iterrows():
            if min(float(row["end_s"]), float(af["end_s"])) > max(float(row["start_s"]), float(af["start_s"])):
                overlaps_af = True
                break
        keep.append(not overlaps_af)
    base = base.loc[keep] if len(base) else base
    frames = [frame for frame in [base, evidence_all] if frame is not None and len(frame) > 0]
    out = pd.concat(frames, ignore_index=True) if frames else base
    return postprocess_physionet_episodes(out)


def postprocess_physionet_episodes(
    episodes,
    af_merge_gap_s=PHYSIONET_AF_MERGE_GAP_S,
    af_tachy_margin_s=PHYSIONET_AF_TACHY_MARGIN_S,
    min_tachy_duration_s=PHYSIONET_MIN_TACHY_DURATION_S,
):
    """
    Consolidate evidence-layer fragments before episode scoring.

    AFDB-style rhythm annotations often span long AF intervals, while the first
    PASM decoder can emit short sinus-tachy fragments around the same fast
    irregular rhythm. Merging nearby AF evidence and dropping very brief tachy
    fragments reduces annotation-scope false alarms without changing the core
    PASM state scores.
    """
    if episodes is None or len(episodes) == 0:
        return episodes

    out = episodes.copy()
    if "type" not in out:
        return out.sort_values(["start_s", "end_s"]).reset_index(drop=True)

    if min_tachy_duration_s is not None and "end_s" in out and "start_s" in out:
        duration_s = out["end_s"].astype(float) - out["start_s"].astype(float)
        out = out[(out["type"] != "sinus_tachy") | (duration_s >= float(min_tachy_duration_s))].copy()

    af = out[out["type"] == "af_like"]
    non_af = out[out["type"] != "af_like"]
    if len(af) > 0:
        af = merge_close_same_type_episodes(af, merge_gap_s=af_merge_gap_s)
    out = pd.concat([non_af, af], ignore_index=True) if len(non_af) or len(af) else out.iloc[0:0]

    af = out[out["type"] == "af_like"]
    if len(af) > 0 and len(out) > 0:
        keep = []
        for _, row in out.iterrows():
            if row.get("type") != "sinus_tachy":
                keep.append(True)
                continue
            tachy_start = float(row["start_s"])
            tachy_end = float(row["end_s"])
            near_af = False
            for _, af_row in af.iterrows():
                af_start = float(af_row["start_s"]) - float(af_tachy_margin_s)
                af_end = float(af_row["end_s"]) + float(af_tachy_margin_s)
                if min(tachy_end, af_end) > max(tachy_start, af_start):
                    near_af = True
                    break
            keep.append(not near_af)
        out = out.loc[keep]

    return out.sort_values(["start_s", "end_s"]).reset_index(drop=True)


def evaluate_physionet_records(records, thresholds=None, skip_empty_truth=True):
    rows = []
    for record in records:
        if skip_empty_truth and len(record.truth_episodes) == 0:
            continue
        pred = run_pasm_on_physionet_record(record, thresholds=thresholds)
        metrics = evaluate_episodes(
            pred,
            normalize_episode_types(record.truth_episodes),
            duration_s=float(len(record.signal)) / float(record.fs),
        )
        metrics.insert(0, "model", "pasm_physionet")
        metrics.insert(0, "record_id", record.record_id)
        rows.append(metrics)
    if not rows:
        raise ValueError("No informative records to evaluate; all records had empty ground-truth episodes.")
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run PASM-Rhythm on PhysioNet records via WFDB.")
    parser.add_argument("--db", choices=["mitdb", "afdb"], default="mitdb")
    parser.add_argument("--records", nargs="+", default=["100"])
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--out", default="PASM_PHYSIONET_VALIDATION.md")
    parser.add_argument("--include-empty-truth", action="store_true")
    args = parser.parse_args(argv)

    loaded = []
    for record_name in args.records:
        if args.db == "mitdb":
            loaded.append(load_mitdb_record(record_name, max_seconds=args.max_seconds))
        else:
            loaded.append(load_afdb_record(record_name, max_seconds=args.max_seconds))

    metrics = evaluate_physionet_records(loaded, skip_empty_truth=not args.include_empty_truth)
    summary = summarize_benchmark(metrics)
    lines = [
        "# PASM-Rhythm PhysioNet Validation",
        "",
        "This report uses WFDB/PhysioNet annotations. It is a research validation harness, not clinical certification.",
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Evidence Layer Parameters",
        "",
        f"- AF merge gap: {PHYSIONET_AF_MERGE_GAP_S:.1f} s",
        f"- AF-adjacent tachy suppression margin: {PHYSIONET_AF_TACHY_MARGIN_S:.1f} s",
        f"- Minimum retained sinus tachy duration: {PHYSIONET_MIN_TACHY_DURATION_S:.1f} s",
        f"- Ectopy short RR: {ECTOPY_SHORT_RR_S:.2f} s",
        f"- Ectopy relative RR fraction: {ECTOPY_RELATIVE_RR_FRACTION:.2f}",
        f"- Ectopy merge gap: {ECTOPY_MERGE_GAP_S:.1f} s",
        f"- Ectopy flood rate threshold: {ECTOPY_FLOOD_RATE_PER_HOUR:.1f} / h",
        f"- Ectopy flood minimum confidence: {ECTOPY_FLOOD_MIN_CONFIDENCE:.2f}",
        f"- Ectopy flood density window: {ECTOPY_FLOOD_DENSITY_WINDOW_S:.1f} s",
        f"- Ectopy flood minimum density: {ECTOPY_FLOOD_MIN_DENSITY}",
        f"- Ectopy flood minimum candidates: {ECTOPY_FLOOD_MIN_CANDIDATES}",
        "",
        "## Raw Metrics",
        "",
        metrics.to_string(index=False),
        "",
    ]
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
