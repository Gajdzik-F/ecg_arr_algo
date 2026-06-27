import numpy as np
import pandas as pd

from pasm_physionet import run_pasm_physionet_pipeline
from pasm_rhythm import PASM_STATES, robust_z


FEATURE_COLUMNS = [
    "hr",
    "rr_prev",
    "rr_next",
    "delta_rr",
    "rr_ratio",
    "local_rr_median",
    "local_rmssd",
    "local_cv",
    "sqi",
    "rpeak_uncertainty",
    "reliability",
    "patient_rr_z",
    "delta_rr_z_abs",
    "rmssd_z",
    "morph_z",
    "score_normal",
    "score_sinus_tachy",
    "score_sinus_brady",
    "score_af_like",
    "score_ectopic_like",
    "score_noise_uncertain",
]


def build_pasm_feature_frame(record, split="unspecified", pipeline=None):
    """
    Build a per-beat PASM feature table for one PhysioNetRecord.

    Labels are derived from truth episodes. Beats outside truth episodes are
    labelled normal. If episodes overlap, the first episode in time order wins.
    """
    if pipeline is None:
        pipeline = run_pasm_physionet_pipeline(record)

    features = pipeline["features"].reset_index(drop=True).copy()
    state_scores = pipeline["state_scores"].reset_index(drop=True)
    patient_memory = pipeline["patient_memory"]
    beats = pipeline.get("beats")

    if len(features) != len(state_scores):
        raise ValueError("features and state_scores must align.")

    out = pd.DataFrame(
        {
            "record_id": record.record_id,
            "time_s": features["time_s"].to_numpy(dtype=float),
            "split": split,
            "label": assign_beat_labels(features["time_s"].to_numpy(dtype=float), record.truth_episodes),
        }
    )
    for column in [
        "hr",
        "rr_prev",
        "rr_next",
        "delta_rr",
        "rr_ratio",
        "local_rr_median",
        "local_rmssd",
        "local_cv",
        "sqi",
        "rpeak_uncertainty",
        "reliability",
    ]:
        out[column] = features[column].to_numpy(dtype=float)

    out["patient_rr_z"] = robust_z(out["rr_prev"].to_numpy(), patient_memory.rr_median, patient_memory.rr_mad)
    out["delta_rr_z_abs"] = np.abs(
        robust_z(out["delta_rr"].fillna(0.0).to_numpy(), 0.0, patient_memory.rr_mad)
    )
    out["rmssd_z"] = robust_z(
        out["local_rmssd"].fillna(patient_memory.rmssd_median).to_numpy(),
        patient_memory.rmssd_median,
        patient_memory.rmssd_mad,
    )
    out["morph_z"] = compute_morph_z(beats, patient_memory, len(out))

    for state in PASM_STATES:
        out[f"score_{state}"] = state_scores[state].to_numpy(dtype=float)

    return out[["record_id", "time_s", "split", "label"] + FEATURE_COLUMNS]


def assign_beat_labels(times_s, truth_episodes):
    times_s = np.asarray(times_s, dtype=float)
    labels = np.full(len(times_s), "normal", dtype=object)
    if truth_episodes is None or len(truth_episodes) == 0:
        return labels

    truth = truth_episodes.sort_values(["start_s", "end_s"]).reset_index(drop=True)
    for _, row in truth.iterrows():
        mask = (labels == "normal") & (times_s >= float(row["start_s"])) & (times_s <= float(row["end_s"]))
        labels[mask] = str(row["type"])
    return labels


def compute_morph_z(beats, patient_memory, n_rows):
    if beats is None or patient_memory.morphology_prototype is None:
        return np.zeros(n_rows, dtype=float)
    beats = np.asarray(beats, dtype=float)
    if len(beats) != n_rows:
        raise ValueError("beats must align with feature rows.")
    dist = np.linalg.norm(beats - patient_memory.morphology_prototype[None, :], axis=1)
    morph_z = robust_z(dist, 0.0, patient_memory.morphology_scale)
    return np.nan_to_num(morph_z, nan=0.0, posinf=8.0, neginf=0.0)


def build_pasm_dataset(records_by_split, pipeline_by_record_id=None):
    rows = []
    pipeline_by_record_id = pipeline_by_record_id or {}
    for split, records in records_by_split.items():
        for record in records:
            pipeline = pipeline_by_record_id.get(record.record_id)
            rows.append(build_pasm_feature_frame(record, split=split, pipeline=pipeline))
    if not rows:
        return pd.DataFrame(columns=["record_id", "time_s", "split", "label"] + FEATURE_COLUMNS)
    return pd.concat(rows, ignore_index=True)
