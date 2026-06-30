import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pasm_dataset import FEATURE_COLUMNS, build_pasm_dataset, build_pasm_feature_frame
from pasm_ai_reranker import (
    RERANKER_FEATURE_COLUMNS,
    build_episode_candidate_dataset,
    fit_episode_reranker,
    label_episode_candidates,
    predict_reranked_episodes,
    tune_pattern_policy,
    tune_reranker_threshold,
)
from pasm_ml_decoder import fit_softmax_decoder
from pasm_physionet import (
    filter_predictions_for_annotation_scope,
    load_afdb_record,
    load_mitdb_record,
    run_pasm_physionet_pipeline,
)
from pasm_rhythm import decode_pasm_episodes
from pasm_validation import evaluate_episodes, normalize_episode_types, summarize_benchmark, threshold_grid


DEFAULT_TRAIN_RECORD_IDS = ("mitdb/200", "mitdb/205", "afdb/04015", "afdb/04043")
DEFAULT_HOLDOUT_RECORD_IDS = ("mitdb/201", "mitdb/203", "afdb/04126")
DEFAULT_MAX_SECONDS_BY_DB = {"mitdb": 900.0, "afdb": 1200.0}
ML_BENCHMARK_PRESETS = {
    "tiny": {
        "train": DEFAULT_TRAIN_RECORD_IDS,
        "holdout": DEFAULT_HOLDOUT_RECORD_IDS,
    },
    "mini": {
        "train": ("mitdb/200", "mitdb/205", "afdb/04015", "afdb/04043", "afdb/04048"),
        "holdout": ("mitdb/201", "mitdb/203", "mitdb/208", "afdb/04126", "afdb/04746"),
    },
    "mitdb-mini": {
        "train": ("mitdb/200", "mitdb/205"),
        "holdout": ("mitdb/201", "mitdb/203", "mitdb/208"),
    },
    "afdb-mini": {
        "train": ("afdb/04015", "afdb/04043", "afdb/04048"),
        "holdout": ("afdb/04126", "afdb/04746"),
    },
}
DEFAULT_GUARDED_CONFIG = {
    "normal_bias": 0.18,
    "min_episode_confidence": 0.55,
    "min_episode_sqi": 0.50,
    "ectopy_min_morph_z": 0.55,
    "ectopy_min_delta_rr_z_abs": 3.0,
    "ectopy_min_score": 0.45,
    "min_beats_by_state": {
        "sinus_tachy": 8,
        "sinus_brady": 8,
        "af_like": 12,
        "ectopic_like": 4,
        "noise_uncertain": 3,
    },
}
GUARD_TUNING_GRID = {
    "normal_bias": (0.10, 0.18, 0.28, 0.40),
    "min_episode_confidence": (0.50, 0.55, 0.65),
    "ectopic_like_min_beats": (4, 6, 8),
}
HARD_NEGATIVE_BOOST = 3.0
MAX_CLASS_WEIGHT = 8.0


def load_records_by_id(record_ids, max_seconds_by_db=None):
    max_seconds_by_db = max_seconds_by_db or DEFAULT_MAX_SECONDS_BY_DB
    records = []
    for record_id in record_ids:
        db, name = parse_record_id(record_id)
        max_seconds = max_seconds_by_db.get(db)
        if db == "mitdb":
            records.append(load_mitdb_record(name, max_seconds=max_seconds))
        elif db == "afdb":
            records.append(load_afdb_record(name, max_seconds=max_seconds))
        else:
            raise ValueError(f"Unsupported database in record id {record_id!r}.")
    return records


def parse_record_id(record_id):
    parts = str(record_id).split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Record id must have form db/name, got {record_id!r}.")
    return parts[0], parts[1]


def resolve_ml_preset(preset):
    if preset not in ML_BENCHMARK_PRESETS:
        raise ValueError(f"Unknown ML benchmark preset {preset!r}. Available: {', '.join(sorted(ML_BENCHMARK_PRESETS))}")
    spec = ML_BENCHMARK_PRESETS[preset]
    return tuple(spec["train"]), tuple(spec["holdout"])


def run_ml_validation(
    train_record_ids=DEFAULT_TRAIN_RECORD_IDS,
    holdout_record_ids=DEFAULT_HOLDOUT_RECORD_IDS,
    preset=None,
    epochs=800,
    lr=0.05,
    l2=1e-3,
    seed=2026,
    guarded_config=None,
):
    if preset is not None:
        train_record_ids, holdout_record_ids = resolve_ml_preset(preset)
    guarded_config = dict(DEFAULT_GUARDED_CONFIG if guarded_config is None else guarded_config)

    loaded_train_records = load_records_by_id(train_record_ids)
    loaded_holdout_records = load_records_by_id(holdout_record_ids)
    train_records = _informative_records(loaded_train_records)
    holdout_records = _informative_records(loaded_holdout_records)
    if not train_records:
        raise ValueError("No informative train records with truth episodes.")
    if not holdout_records:
        raise ValueError("No informative holdout records with truth episodes.")

    all_records = train_records + holdout_records
    pipelines = {record.record_id: run_pasm_physionet_pipeline(record) for record in all_records}
    dataset = build_pasm_dataset(
        {"train": train_records, "holdout": holdout_records},
        pipelines,
        expand_short_ectopy=True,
    )
    train_df = dataset[dataset["split"] == "train"].reset_index(drop=True)
    holdout_df = dataset[dataset["split"] == "holdout"].reset_index(drop=True)
    reranker_train_df, reranker_train_uncovered = build_episode_candidate_dataset(train_records, pipelines, split="train")
    reranker_holdout_df, reranker_holdout_uncovered = build_episode_candidate_dataset(
        holdout_records,
        pipelines,
        split="holdout",
    )

    model = fit_softmax_decoder(
        train_df,
        FEATURE_COLUMNS,
        epochs=epochs,
        lr=lr,
        l2=l2,
        seed=seed,
        max_class_weight=MAX_CLASS_WEIGHT,
    )
    thresholds, train_ml_metrics, train_ml_summary = tune_ml_thresholds(train_records, pipelines, model)
    hard_negative_boost = build_hard_negative_boosts(train_records, pipelines, train_df, model, thresholds)
    fpaware_model = fit_softmax_decoder(
        train_df,
        FEATURE_COLUMNS,
        epochs=epochs,
        lr=lr,
        l2=l2,
        seed=seed + 1,
        max_class_weight=MAX_CLASS_WEIGHT,
        sample_weight_boost=hard_negative_boost,
    )
    fpaware_thresholds, train_fpaware_metrics_raw, _ = tune_ml_thresholds(
        train_records,
        pipelines,
        fpaware_model,
        model_name="pasm_ml_decoder_fpaware_raw",
    )
    tuned_guarded_config, guarded_train_metrics = tune_guarded_config(
        train_records,
        pipelines,
        fpaware_model,
        fpaware_thresholds,
        base_config=guarded_config,
    )

    baseline_train_metrics = evaluate_pipeline_records(train_records, pipelines, model_name="pasm_physionet")
    baseline_holdout_metrics = evaluate_pipeline_records(holdout_records, pipelines, model_name="pasm_physionet")
    ml_holdout_metrics = evaluate_ml_records(
        holdout_records,
        pipelines,
        model,
        thresholds=thresholds,
        model_name="pasm_ml_decoder",
    )
    guarded_train_metrics = evaluate_ml_records(
        train_records,
        pipelines,
        model,
        thresholds=thresholds,
        model_name="pasm_ml_decoder_guarded",
        guarded_config=tuned_guarded_config,
    )
    guarded_holdout_metrics = evaluate_ml_records(
        holdout_records,
        pipelines,
        model,
        thresholds=thresholds,
        model_name="pasm_ml_decoder_guarded",
        guarded_config=tuned_guarded_config,
    )
    fpaware_train_metrics = evaluate_ml_records(
        train_records,
        pipelines,
        fpaware_model,
        thresholds=fpaware_thresholds,
        model_name="pasm_ml_decoder_fpaware",
        guarded_config=tuned_guarded_config,
    )
    fpaware_holdout_metrics = evaluate_ml_records(
        holdout_records,
        pipelines,
        fpaware_model,
        thresholds=fpaware_thresholds,
        model_name="pasm_ml_decoder_fpaware",
        guarded_config=tuned_guarded_config,
    )
    reranker_fit_df = _reranker_fit_frame(reranker_train_df)
    reranker_safe_model = fit_episode_reranker(
        reranker_fit_df,
        RERANKER_FEATURE_COLUMNS,
        epochs=epochs,
        lr=lr,
        l2=l2,
        seed=seed + 2,
        max_class_weight=MAX_CLASS_WEIGHT,
    )
    reranker_safe_threshold, reranker_safe_train_metrics = tune_reranker_threshold(
        train_records,
        pipelines,
        reranker_safe_model,
        fp_per_hour_limit=9.0,
        candidate_policy="safe",
        model_name="pasm_ai_reranker_safe",
    )
    reranker_episode_boost = build_episode_hard_negative_boosts(
        train_records,
        pipelines,
        reranker_fit_df,
        reranker_safe_model,
        threshold=reranker_safe_threshold,
        candidate_policy="pattern_v2",
    )
    reranker_v2_model = fit_episode_reranker(
        reranker_fit_df,
        RERANKER_FEATURE_COLUMNS,
        epochs=epochs,
        lr=lr,
        l2=l2,
        seed=seed + 3,
        max_class_weight=MAX_CLASS_WEIGHT,
        sample_weight_boost=reranker_episode_boost,
    )
    reranker_v2_threshold, reranker_v2_policy_config, reranker_v2_train_metrics = tune_pattern_policy(
        train_records,
        pipelines,
        reranker_v2_model,
        target_fp_per_hour=9.0,
    )
    reranker_safe_holdout_metrics = evaluate_reranker_records(
        holdout_records,
        pipelines,
        reranker_safe_model,
        threshold=reranker_safe_threshold,
        candidate_policy="safe",
        model_name="pasm_ai_reranker_safe",
    )
    reranker_v2_holdout_metrics = evaluate_reranker_records(
        holdout_records,
        pipelines,
        reranker_v2_model,
        threshold=reranker_v2_threshold,
        candidate_policy="pattern_v2",
        policy_config=reranker_v2_policy_config,
        model_name="pasm_ai_reranker_v2",
    )
    train_metrics = pd.concat(
        [
            baseline_train_metrics,
            train_ml_metrics,
            guarded_train_metrics,
            fpaware_train_metrics,
            reranker_safe_train_metrics,
            reranker_v2_train_metrics,
        ],
        ignore_index=True,
    )
    holdout_metrics = pd.concat(
        [
            baseline_holdout_metrics,
            ml_holdout_metrics,
            guarded_holdout_metrics,
            fpaware_holdout_metrics,
            reranker_safe_holdout_metrics,
            reranker_v2_holdout_metrics,
        ],
        ignore_index=True,
    )
    holdout_predictions = collect_holdout_predictions(
        holdout_records,
        pipelines,
        model,
        thresholds,
        fpaware_model,
        fpaware_thresholds,
        tuned_guarded_config,
        reranker_safe_model,
        reranker_safe_threshold,
        reranker_v2_model,
        reranker_v2_threshold,
        reranker_v2_policy_config,
    )
    guard_removed = collect_guard_removed(
        holdout_records,
        pipelines,
        [
            ("pasm_ml_decoder_guarded", model, thresholds),
            ("pasm_ml_decoder_fpaware", fpaware_model, fpaware_thresholds),
        ],
        tuned_guarded_config,
    )
    reranker_removed = collect_reranker_removed(
        holdout_records,
        pipelines,
        [
            ("pasm_ai_reranker_safe", reranker_safe_model, reranker_safe_threshold, "safe", None),
            ("pasm_ai_reranker_v2", reranker_v2_model, reranker_v2_threshold, "pattern_v2", reranker_v2_policy_config),
        ],
    )
    reranker_loro = run_mitdb_loro_reranker_validation(all_records, pipelines, epochs=epochs, lr=lr, l2=l2, seed=seed)

    return {
        "preset": preset or "custom",
        "requested_train_record_ids": list(train_record_ids),
        "requested_holdout_record_ids": list(holdout_record_ids),
        "train_record_ids": [record.record_id for record in train_records],
        "holdout_record_ids": [record.record_id for record in holdout_records],
        "skipped_records": skipped_records_table(loaded_train_records, loaded_holdout_records),
        "pipelines": pipelines,
        "dataset": dataset,
        "train_df": train_df,
        "holdout_df": holdout_df,
        "reranker_train_df": reranker_train_df,
        "reranker_holdout_df": reranker_holdout_df,
        "reranker_train_uncovered": reranker_train_uncovered,
        "reranker_holdout_uncovered": reranker_holdout_uncovered,
        "model": model,
        "fpaware_model": fpaware_model,
        "reranker_model": reranker_safe_model,
        "reranker_safe_model": reranker_safe_model,
        "reranker_v2_model": reranker_v2_model,
        "reranker_threshold": reranker_safe_threshold,
        "reranker_safe_threshold": reranker_safe_threshold,
        "reranker_v2_threshold": reranker_v2_threshold,
        "reranker_v2_policy_config": reranker_v2_policy_config,
        "thresholds": thresholds,
        "fpaware_thresholds": fpaware_thresholds,
        "guarded_config": tuned_guarded_config,
        "hard_negative_count": int(np.sum(hard_negative_boost > 1.0)),
        "hard_negative_boost": HARD_NEGATIVE_BOOST,
        "reranker_hard_negative_count": int(np.sum(reranker_episode_boost > 1.0)),
        "reranker_hard_negative_boost": HARD_NEGATIVE_BOOST,
        "guard_removed": guard_removed,
        "reranker_removed": reranker_removed,
        "reranker_loro": reranker_loro,
        "holdout_predictions": holdout_predictions,
        "train_metrics": train_metrics,
        "train_summary": summarize_benchmark(train_metrics),
        "holdout_metrics": holdout_metrics,
        "holdout_summary": summarize_benchmark(holdout_metrics),
    }


def _informative_records(records):
    return [record for record in records if record.truth_episodes is not None and len(record.truth_episodes) > 0]


def _reranker_fit_frame(candidate_df):
    if candidate_df is None or len(candidate_df) == 0:
        return candidate_df
    labels = set(candidate_df["accepted"].astype(int).tolist()) if "accepted" in candidate_df else set()
    if labels == {0, 1}:
        return candidate_df
    if labels == {0}:
        synthetic = candidate_df.head(1).copy()
        synthetic["accepted"] = 1
        synthetic["source"] = "fallback_accept"
        synthetic["confidence"] = 1.0
        synthetic["mean_state_score"] = 1.0
        synthetic["max_state_score"] = 1.0
        return pd.concat([candidate_df, synthetic], ignore_index=True)
    synthetic = candidate_df.copy()
    synthetic["accepted"] = 0
    synthetic["source"] = "fallback_reject"
    for col in ["confidence", "mean_state_score", "max_state_score", "rr_support", "pause_support", "morph_support", "density_support"]:
        if col in synthetic:
            synthetic[col] = 0.0
    return pd.concat([candidate_df, synthetic], ignore_index=True)


def run_mitdb_loro_reranker_validation(records, pipelines, epochs=800, lr=0.05, l2=1e-3, seed=2026):
    mitdb_records = [record for record in records if str(record.record_id).startswith("mitdb/")]
    if len(mitdb_records) < 2:
        return pd.DataFrame(columns=["test_record_id", "train_record_ids", "f1", "false_alarms_per_hour"])
    rows = []
    for fold, test_record in enumerate(mitdb_records):
        train_records = [record for record in mitdb_records if record.record_id != test_record.record_id]
        train_df, _ = build_episode_candidate_dataset(train_records, pipelines, split="train")
        fit_df = _reranker_fit_frame(train_df)
        if fit_df is None or len(fit_df) == 0:
            continue
        model = fit_episode_reranker(
            fit_df,
            RERANKER_FEATURE_COLUMNS,
            epochs=min(120, max(40, int(epochs // 6))),
            lr=lr,
            l2=l2,
            seed=seed + 100 + fold,
            max_class_weight=MAX_CLASS_WEIGHT,
        )
        boost = build_episode_hard_negative_boosts(
            train_records,
            pipelines,
            fit_df,
            model,
            threshold=0.50,
            candidate_policy="pattern_v2",
        )
        model = fit_episode_reranker(
            fit_df,
            RERANKER_FEATURE_COLUMNS,
            epochs=min(120, max(40, int(epochs // 6))),
            lr=lr,
            l2=l2,
            seed=seed + 200 + fold,
            max_class_weight=MAX_CLASS_WEIGHT,
            sample_weight_boost=boost,
        )
        threshold, policy_config, _ = tune_pattern_policy(train_records, pipelines, model, target_fp_per_hour=9.0)
        metrics = evaluate_reranker_records(
            [test_record],
            pipelines,
            model,
            threshold=threshold,
            candidate_policy="pattern_v2",
            policy_config=policy_config,
            model_name="pasm_ai_reranker_v2_loro",
        )
        macro = metrics[metrics["type"] == "macro"].iloc[0]
        faph = metrics[metrics["type"] == "false_alarms_per_hour"].iloc[0]
        rows.append(
            {
                "test_record_id": test_record.record_id,
                "train_record_ids": ",".join(record.record_id for record in train_records),
                "threshold": threshold,
                "f1": float(macro["f1"]),
                "precision": float(macro["precision"]),
                "recall": float(macro["recall"]),
                "false_alarms_per_hour": float(faph["f1"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["test_record_id", "train_record_ids", "threshold", "f1", "false_alarms_per_hour"])
    return pd.DataFrame(rows)


def skipped_records_table(train_records, holdout_records):
    rows = []
    for split, records in [("train", train_records), ("holdout", holdout_records)]:
        for record in records:
            if record.truth_episodes is None or len(record.truth_episodes) == 0:
                rows.append({"split": split, "record_id": record.record_id, "reason": "empty_truth"})
    return pd.DataFrame(rows, columns=["split", "record_id", "reason"])


def tune_ml_thresholds(records, pipelines, model, model_name="pasm_ml_decoder"):
    best_cfg = None
    best_score = -np.inf
    best_metrics = None
    best_summary = None
    for cfg in threshold_grid():
        metrics = evaluate_ml_records(records, pipelines, model, thresholds=cfg, model_name=model_name)
        summary = summarize_benchmark(metrics)
        row = summary.iloc[0]
        score = float(row["episode_f1_mean"]) - 0.001 * float(row["false_alarms_per_hour_mean"])
        if score > best_score:
            best_score = score
            best_cfg = dict(cfg)
            best_metrics = metrics
            best_summary = summary
    return best_cfg, best_metrics, best_summary


def build_hard_negative_boosts(records, pipelines, train_df, model, thresholds):
    boosts = np.ones(len(train_df), dtype=float)
    for record in records:
        pipeline = pipelines[record.record_id]
        episodes = predict_ml_episodes(record, pipeline, model, thresholds, guarded_config=None)
        diagnostics = diagnostic_rows_for_predictions(
            record,
            episodes,
            model_name="pasm_ml_decoder",
            duration_s=float(len(record.signal)) / float(record.fs),
        )
        fp = diagnostics[diagnostics["status"] == "FP"] if len(diagnostics) else pd.DataFrame()
        if len(fp) == 0:
            continue
        record_mask = (train_df["record_id"] == record.record_id) & (train_df["label"] == "normal")
        for _, row in fp.iterrows():
            time_mask = (train_df["time_s"] >= float(row["start_s"])) & (train_df["time_s"] <= float(row["end_s"]))
            boosts[record_mask & time_mask] = HARD_NEGATIVE_BOOST
    return boosts


def build_episode_hard_negative_boosts(records, pipelines, candidate_df, reranker_model, threshold=0.50, candidate_policy="pattern_v2"):
    boosts = np.ones(len(candidate_df), dtype=float)
    if candidate_df is None or len(candidate_df) == 0:
        return boosts
    for record in records:
        candidates = predict_reranked_candidates(
            record,
            pipelines[record.record_id],
            reranker_model,
            threshold=threshold,
            candidate_policy=candidate_policy,
        )
        if len(candidates) == 0:
            continue
        labeled, _ = label_episode_candidates(candidates, record.truth_episodes, record_id=record.record_id, split="train")
        labeled["ai_decision"] = candidates["ai_decision"].to_numpy(dtype=object)
        false_accepts = labeled[(labeled["ai_decision"] == "accept") & (labeled["accepted"].astype(int) == 0)]
        if len(false_accepts) == 0:
            continue
        record_mask = candidate_df["record_id"].astype(str) == record.record_id
        for _, row in false_accepts.iterrows():
            start = float(row.get("start_s", np.nan))
            end = float(row.get("end_s", np.nan))
            pattern = str(row.get("pattern", ""))
            source = str(row.get("source", ""))
            if not np.isfinite(start) or not np.isfinite(end):
                continue
            same_candidate = (
                record_mask
                & (candidate_df["source"].astype(str) == source)
                & (candidate_df["pattern"].astype(str) == pattern)
                & np.isclose(candidate_df["start_s"].astype(float), start)
                & np.isclose(candidate_df["end_s"].astype(float), end)
            )
            weight = HARD_NEGATIVE_BOOST
            duration = max(0.0, end - start)
            if source == "relaxed_ectopy" and (pattern == "morphology_cluster" or duration > 2.5):
                weight = HARD_NEGATIVE_BOOST * 1.5
            if source == "relaxed_ectopy" and pattern == "short_coupled_run" and duration <= 1.5:
                weight = HARD_NEGATIVE_BOOST * 1.25
            boosts[same_candidate.to_numpy(dtype=bool)] = weight
    return boosts


def tune_guarded_config(records, pipelines, model, thresholds, base_config=None):
    base_config = dict(DEFAULT_GUARDED_CONFIG if base_config is None else base_config)
    best_config = None
    best_score = -np.inf
    best_metrics = None
    for normal_bias in GUARD_TUNING_GRID["normal_bias"]:
        for min_conf in GUARD_TUNING_GRID["min_episode_confidence"]:
            for ectopy_min_beats in GUARD_TUNING_GRID["ectopic_like_min_beats"]:
                cfg = dict(base_config)
                cfg["normal_bias"] = normal_bias
                cfg["min_episode_confidence"] = min_conf
                min_beats = dict(cfg.get("min_beats_by_state", {}))
                min_beats["ectopic_like"] = ectopy_min_beats
                cfg["min_beats_by_state"] = min_beats
                metrics = evaluate_ml_records(
                    records,
                    pipelines,
                    model,
                    thresholds=thresholds,
                    model_name="pasm_ml_decoder_fpaware",
                    guarded_config=cfg,
                )
                summary = summarize_benchmark(metrics)
                row = summary.iloc[0]
                score = float(row["episode_f1_mean"]) - 0.002 * float(row["false_alarms_per_hour_mean"])
                if score > best_score:
                    best_score = score
                    best_config = cfg
                    best_metrics = metrics
    return best_config, best_metrics


def evaluate_pipeline_records(records, pipelines, model_name="pasm_physionet"):
    rows = []
    for record in records:
        pred = pipelines[record.record_id]["episodes"]
        metrics = evaluate_episodes(
            pred,
            normalize_episode_types(record.truth_episodes),
            duration_s=float(len(record.signal)) / float(record.fs),
        )
        metrics.insert(0, "model", model_name)
        metrics.insert(0, "record_id", record.record_id)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def evaluate_ml_records(records, pipelines, model, thresholds=None, model_name="pasm_ml_decoder", guarded_config=None):
    rows = []
    for record in records:
        episodes = predict_ml_episodes(record, pipelines[record.record_id], model, thresholds, guarded_config)
        metrics = evaluate_episodes(
            episodes,
            normalize_episode_types(record.truth_episodes),
            duration_s=float(len(record.signal)) / float(record.fs),
        )
        metrics.insert(0, "model", model_name)
        metrics.insert(0, "record_id", record.record_id)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def evaluate_reranker_records(
    records,
    pipelines,
    reranker_model,
    threshold=0.50,
    model_name="pasm_ai_reranker_safe",
    candidate_policy="safe",
    policy_config=None,
):
    rows = []
    for record in records:
        episodes = predict_reranked_episodes(
            record,
            pipelines[record.record_id],
            reranker_model,
            threshold=threshold,
            candidate_policy=candidate_policy,
            policy_config=policy_config,
        )
        metrics = evaluate_episodes(
            episodes,
            normalize_episode_types(record.truth_episodes),
            duration_s=float(len(record.signal)) / float(record.fs),
        )
        metrics.insert(0, "model", model_name)
        metrics.insert(0, "record_id", record.record_id)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def predict_ml_episodes(record, pipeline, model, thresholds=None, guarded_config=None):
    frame = build_pasm_feature_frame(record, split="eval", pipeline=pipeline)
    state_scores = model.predict_proba(frame)
    if guarded_config:
        state_scores = apply_normal_bias(state_scores, guarded_config.get("normal_bias", 0.0))
    episodes = decode_pasm_episodes(
        pipeline["features"],
        state_scores,
        min_confidence_by_state=thresholds,
    )
    episodes = attach_episode_support(episodes, frame)
    episodes = postprocess_ml_episodes(episodes, frame, candidate_policy="balanced")
    if guarded_config:
        episodes = guard_ml_episodes(episodes, guarded_config)
    return filter_predictions_for_annotation_scope(record, episodes)


def apply_normal_bias(state_scores, normal_bias):
    if not normal_bias:
        return state_scores
    scores = state_scores.copy()
    scores["normal"] = scores["normal"] + float(normal_bias)
    values = scores.to_numpy(dtype=float)
    values = values / (values.sum(axis=1, keepdims=True) + 1e-12)
    return pd.DataFrame(values, columns=scores.columns, index=scores.index)


def guard_ml_episodes(episodes, guarded_config=None):
    kept, _ = guard_ml_episodes_with_report(episodes, guarded_config=guarded_config)
    return kept


def postprocess_ml_episodes(episodes, frame=None, candidate_policy="balanced"):
    if episodes is None or len(episodes) == 0 or candidate_policy is None:
        return episodes
    if candidate_policy != "balanced":
        raise ValueError(f"Unknown ML episode postprocess policy: {candidate_policy!r}")

    out = episodes.copy().reset_index(drop=True)
    keep = np.ones(len(out), dtype=bool)
    duration = out["end_s"].astype(float) - out["start_s"].astype(float)
    beats = out.get("beats", pd.Series(np.zeros(len(out)))).fillna(0).astype(float)
    confidence = out.get("confidence", pd.Series(np.zeros(len(out)))).fillna(0.0).astype(float)

    af = out["type"].to_numpy(dtype=object) == "af_like"
    short_af_fp_shape = af & (duration < 10.0) & (beats < 20) & (confidence < 0.95)
    keep &= ~short_af_fp_shape

    ectopy = out["type"].to_numpy(dtype=object) == "ectopic_like"
    if np.any(ectopy):
        morph = out.get("max_morph_z", pd.Series(np.zeros(len(out)))).fillna(0.0).astype(float)
        delta = out.get("max_delta_rr_z_abs", pd.Series(np.zeros(len(out)))).fillna(0.0).astype(float)
        score = out.get("max_score_ectopic_like", pd.Series(np.zeros(len(out)))).fillna(0.0).astype(float)
        short_run = out.get("max_short_run_length", pd.Series(np.zeros(len(out)))).fillna(0.0).astype(float)
        ectopy_supported = (morph >= 0.55) | (delta >= 3.0) | (score >= 0.45) | (short_run >= 2.0)
        keep[ectopy] &= ectopy_supported[ectopy]

    return out.loc[keep].reset_index(drop=True)


def guard_ml_episodes_with_report(episodes, guarded_config=None, record_id="", model_name=""):
    guarded_config = dict(DEFAULT_GUARDED_CONFIG if guarded_config is None else guarded_config)
    if episodes is None or len(episodes) == 0:
        return episodes, pd.DataFrame(columns=["record_id", "model", "type", "reason", "removed"])
    out = episodes.copy()
    keep = np.ones(len(out), dtype=bool)
    reasons = [[] for _ in range(len(out))]
    min_conf = float(guarded_config.get("min_episode_confidence", 0.0))
    min_sqi = float(guarded_config.get("min_episode_sqi", 0.0))
    min_beats_by_state = guarded_config.get("min_beats_by_state", {})
    if "confidence" in out:
        mask = out["confidence"].fillna(0.0).to_numpy(dtype=float) >= min_conf
        _add_guard_reasons(reasons, mask, "low_confidence")
        keep &= mask
    if "mean_sqi" in out:
        mask = out["mean_sqi"].fillna(0.0).to_numpy(dtype=float) >= min_sqi
        _add_guard_reasons(reasons, mask, "low_sqi")
        keep &= mask
    if "beats" in out:
        min_beats = np.array([int(min_beats_by_state.get(state, 1)) for state in out["type"]], dtype=int)
        mask = out["beats"].fillna(0).to_numpy(dtype=float) >= min_beats
        _add_guard_reasons(reasons, mask, "too_few_beats")
        keep &= mask
    ectopy_mask = ectopy_support_mask(out, guarded_config)
    _add_guard_reasons(reasons, ectopy_mask, "weak_ectopy_support")
    keep &= ectopy_mask

    removed_rows = []
    for i, reason_list in enumerate(reasons):
        if keep[i]:
            continue
        reason = "+".join(reason_list) if reason_list else "filtered"
        removed_rows.append(
            {
                "record_id": record_id,
                "model": model_name,
                "type": out.iloc[i].get("type", ""),
                "reason": reason,
                "removed": 1,
            }
        )
    return out.loc[keep].reset_index(drop=True), pd.DataFrame(removed_rows)


def _add_guard_reasons(reasons, passing_mask, reason):
    for i, passed in enumerate(passing_mask):
        if not bool(passed):
            reasons[i].append(reason)


def ectopy_support_mask(episodes, guarded_config=None):
    guarded_config = dict(DEFAULT_GUARDED_CONFIG if guarded_config is None else guarded_config)
    if episodes is None or len(episodes) == 0:
        return np.ones(0, dtype=bool)
    mask = np.ones(len(episodes), dtype=bool)
    ectopy = episodes["type"].to_numpy(dtype=object) == "ectopic_like"
    if not np.any(ectopy):
        return mask
    morph = episodes.get("max_morph_z", pd.Series(np.zeros(len(episodes)))).fillna(0.0).to_numpy(dtype=float)
    delta_rr = episodes.get("max_delta_rr_z_abs", pd.Series(np.zeros(len(episodes)))).fillna(0.0).to_numpy(dtype=float)
    score = episodes.get("max_score_ectopic_like", pd.Series(np.zeros(len(episodes)))).fillna(0.0).to_numpy(dtype=float)
    supported = (
        (morph >= float(guarded_config.get("ectopy_min_morph_z", 0.0)))
        | (delta_rr >= float(guarded_config.get("ectopy_min_delta_rr_z_abs", 0.0)))
        | (score >= float(guarded_config.get("ectopy_min_score", 0.0)))
    )
    mask[ectopy] = supported[ectopy]
    return mask


def attach_episode_support(episodes, frame):
    if episodes is None or len(episodes) == 0:
        return episodes
    out = episodes.copy()
    support_cols = ["morph_z", "delta_rr_z_abs", "score_ectopic_like", "short_run_length"]
    for col in support_cols:
        if col not in frame:
            frame[col] = 0.0
    max_morph = []
    max_delta = []
    max_score = []
    max_short_run = []
    for _, episode in out.iterrows():
        seg = frame[(frame["time_s"] >= float(episode["start_s"])) & (frame["time_s"] <= float(episode["end_s"]))]
        if len(seg) == 0:
            max_morph.append(0.0)
            max_delta.append(0.0)
            max_score.append(0.0)
            max_short_run.append(0.0)
        else:
            max_morph.append(float(seg["morph_z"].max()))
            max_delta.append(float(seg["delta_rr_z_abs"].max()))
            max_score.append(float(seg["score_ectopic_like"].max()))
            max_short_run.append(float(seg["short_run_length"].max()))
    out["max_morph_z"] = max_morph
    out["max_delta_rr_z_abs"] = max_delta
    out["max_score_ectopic_like"] = max_score
    out["max_short_run_length"] = max_short_run
    return out


def collect_holdout_predictions(
    records,
    pipelines,
    model,
    thresholds,
    fpaware_model,
    fpaware_thresholds,
    guarded_config,
    reranker_safe_model=None,
    reranker_safe_threshold=0.50,
    reranker_v2_model=None,
    reranker_v2_threshold=0.50,
    reranker_v2_policy_config=None,
):
    rows = []
    for record in records:
        baseline = pipelines[record.record_id]["episodes"]
        raw_ml = predict_ml_episodes(record, pipelines[record.record_id], model, thresholds, guarded_config=None)
        guarded_ml = predict_ml_episodes(record, pipelines[record.record_id], model, thresholds, guarded_config=guarded_config)
        fpaware_ml = predict_ml_episodes(
            record,
            pipelines[record.record_id],
            fpaware_model,
            fpaware_thresholds,
            guarded_config=guarded_config,
        )
        model_episodes = [
            ("pasm_physionet", baseline),
            ("pasm_ml_decoder", raw_ml),
            ("pasm_ml_decoder_guarded", guarded_ml),
            ("pasm_ml_decoder_fpaware", fpaware_ml),
        ]
        if reranker_safe_model is not None:
            reranked = predict_reranked_episodes(
                record,
                pipelines[record.record_id],
                reranker_safe_model,
                threshold=reranker_safe_threshold,
                candidate_policy="safe",
            )
            model_episodes.append(("pasm_ai_reranker_safe", reranked))
        if reranker_v2_model is not None:
            reranked_v2 = predict_reranked_episodes(
                record,
                pipelines[record.record_id],
                reranker_v2_model,
                threshold=reranker_v2_threshold,
                candidate_policy="pattern_v2",
                policy_config=reranker_v2_policy_config,
            )
            model_episodes.append(("pasm_ai_reranker_v2", reranked_v2))
        for model_name, episodes in model_episodes:
            diagnostics = diagnostic_rows_for_predictions(
                record,
                episodes,
                model_name=model_name,
                duration_s=float(len(record.signal)) / float(record.fs),
            )
            if len(diagnostics) > 0:
                rows.append(diagnostics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def collect_guard_removed(records, pipelines, model_specs, guarded_config):
    rows = []
    for record in records:
        for model_name, model, thresholds in model_specs:
            frame = build_pasm_feature_frame(record, split="eval", pipeline=pipelines[record.record_id])
            state_scores = apply_normal_bias(model.predict_proba(frame), guarded_config.get("normal_bias", 0.0))
            episodes = decode_pasm_episodes(
                pipelines[record.record_id]["features"],
                state_scores,
                min_confidence_by_state=thresholds,
            )
            episodes = attach_episode_support(episodes, frame)
            _, removed = guard_ml_episodes_with_report(
                episodes,
                guarded_config=guarded_config,
                record_id=record.record_id,
                model_name=model_name,
            )
            if len(removed):
                rows.append(removed)
    if not rows:
        return pd.DataFrame(columns=["record_id", "model", "type", "reason", "removed"])
    return pd.concat(rows, ignore_index=True)


def collect_reranker_removed(records, pipelines, model_specs):
    rows = []
    for record in records:
        for model_name, reranker_model, threshold, candidate_policy, policy_config in model_specs:
            candidates = predict_reranked_candidates(
                record,
                pipelines[record.record_id],
                reranker_model,
                threshold=threshold,
                candidate_policy=candidate_policy,
                policy_config=policy_config,
            )
            rejected = candidates[candidates["ai_decision"] == "reject"] if len(candidates) else pd.DataFrame()
            for _, row in rejected.iterrows():
                rows.append(
                    {
                        "record_id": record.record_id,
                        "model": model_name,
                        "type": row.get("type", ""),
                        "source": row.get("source", ""),
                        "pattern": row.get("pattern", ""),
                        "ai_accept_proba": float(row.get("ai_accept_proba", np.nan)),
                        "reason": "below_ai_threshold",
                        "removed": 1,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["record_id", "model", "type", "source", "pattern", "ai_accept_proba", "reason", "removed"]
        )
    return pd.DataFrame(rows)


def predict_reranked_candidates(
    record,
    pipeline,
    reranker_model,
    threshold=0.50,
    candidate_policy="safe",
    policy_config=None,
):
    from pasm_ai_reranker import build_episode_candidates, reranker_accept_mask

    candidates = build_episode_candidates(record, pipeline)
    if len(candidates) == 0:
        return candidates
    out = candidates.copy().reset_index(drop=True)
    proba = reranker_model.predict_accept_proba(out)
    out["ai_accept_proba"] = proba
    keep = reranker_accept_mask(
        out,
        proba,
        threshold=threshold,
        candidate_policy=candidate_policy,
        policy_config=policy_config,
    )
    out["ai_decision"] = np.where(keep, "accept", "reject")
    out["ai_top_features"] = [reranker_model.top_feature_names(row) for row in out.to_dict("records")]
    return out


def diagnostic_rows_for_predictions(record, predicted, model_name, duration_s, iou_threshold=0.30):
    predicted = normalize_episode_types(predicted).reset_index(drop=True)
    truth = normalize_episode_types(record.truth_episodes).reset_index(drop=True)
    if len(predicted) == 0:
        return pd.DataFrame(
            columns=[
                "record_id",
                "model",
                "status",
                "type",
                "start_s",
                "end_s",
                "duration_s",
                "confidence",
                "mean_sqi",
                "beats",
                "source",
                "pattern",
                "ai_accept_proba",
                "ai_decision",
                "ai_top_features",
                "best_iou",
                "failure_stage",
                "record_duration_s",
            ]
        )

    rows = []
    truth_by_type = {typ: truth[truth["type"] == typ].reset_index(drop=True) for typ in truth["type"].dropna().unique()}
    for _, pred in predicted.iterrows():
        candidates = truth_by_type.get(pred["type"], pd.DataFrame())
        best_iou = 0.0
        for _, truth_row in candidates.iterrows():
            best_iou = max(best_iou, episode_iou_for_rows(pred, truth_row))
        status = "TP" if best_iou >= iou_threshold else "FP"
        rows.append(
            {
                "record_id": record.record_id,
                "model": model_name,
                "status": status,
                "type": pred["type"],
                "start_s": float(pred["start_s"]),
                "end_s": float(pred["end_s"]),
                "duration_s": max(0.0, float(pred["end_s"]) - float(pred["start_s"])),
                "confidence": float(pred.get("confidence", np.nan)),
                "mean_sqi": float(pred.get("mean_sqi", np.nan)),
                "beats": int(pred.get("beats", 0)) if pd.notna(pred.get("beats", np.nan)) else 0,
                "source": pred.get("source", ""),
                "pattern": pred.get("pattern", ""),
                "ai_accept_proba": float(pred.get("ai_accept_proba", np.nan)),
                "ai_decision": pred.get("ai_decision", ""),
                "ai_top_features": pred.get("ai_top_features", ""),
                "best_iou": float(best_iou),
                "failure_stage": classify_prediction_failure(pred, best_iou, iou_threshold),
                "record_duration_s": float(duration_s),
            }
        )
    return pd.DataFrame(rows)


def episode_iou_for_rows(a, b):
    start = max(float(a["start_s"]), float(b["start_s"]))
    end = min(float(a["end_s"]), float(b["end_s"]))
    inter = max(0.0, end - start)
    union = max(float(a["end_s"]), float(b["end_s"])) - min(float(a["start_s"]), float(b["start_s"]))
    if union <= 0:
        return 0.0
    return inter / union


def classify_prediction_failure(pred, best_iou, iou_threshold):
    if best_iou >= iou_threshold:
        return "matched_episode"
    confidence = float(pred.get("confidence", np.nan))
    beats = float(pred.get("beats", np.nan))
    mean_sqi = float(pred.get("mean_sqi", np.nan))
    if np.isfinite(confidence) and confidence < DEFAULT_GUARDED_CONFIG["min_episode_confidence"]:
        return "decoder_low_confidence"
    if np.isfinite(beats) and beats < DEFAULT_GUARDED_CONFIG["min_beats_by_state"].get(pred.get("type"), 1):
        return "decoder_short_episode"
    if np.isfinite(mean_sqi) and mean_sqi < DEFAULT_GUARDED_CONFIG["min_episode_sqi"]:
        return "decoder_low_sqi"
    return "beat_state_scoring"


def write_ml_validation_report(path, result):
    path = Path(path)
    dataset = result["dataset"]
    train_counts = _label_counts(dataset[dataset["split"] == "train"])
    holdout_counts = _label_counts(dataset[dataset["split"] == "holdout"])
    train_records = pd.DataFrame({"record_id": result["train_record_ids"]})
    holdout_records = pd.DataFrame({"record_id": result["holdout_record_ids"]})
    requested_train_records = pd.DataFrame({"record_id": result["requested_train_record_ids"]})
    requested_holdout_records = pd.DataFrame({"record_id": result["requested_holdout_record_ids"]})
    threshold_rows = pd.DataFrame(
        [{"state": state, "min_confidence": value} for state, value in sorted(result["thresholds"].items())]
    )
    fpaware_threshold_rows = pd.DataFrame(
        [{"state": state, "min_confidence": value} for state, value in sorted(result["fpaware_thresholds"].items())]
    )
    guarded_rows = guarded_config_table(result["guarded_config"])
    holdout_predictions = result["holdout_predictions"]
    reranker_threshold_rows = pd.DataFrame(
        [
            {"model": "pasm_ai_reranker_safe", "parameter": "accept_threshold", "value": result["reranker_safe_threshold"]},
            {"model": "pasm_ai_reranker_v2", "parameter": "accept_threshold", "value": result["reranker_v2_threshold"]},
        ]
        + [
            {"model": "pasm_ai_reranker_v2", "parameter": key, "value": value}
            for key, value in sorted(result["reranker_v2_policy_config"].items())
        ]
    )

    lines = [
        "# PASM-Rhythm ML Validation",
        "",
        "This report evaluates lightweight NumPy softmax decoders and a candidate-level PASM-AI episode reranker.",
        "It is a research checkpoint, not clinical validation.",
        "",
        f"Preset: `{result['preset']}`",
        "",
        "## Patient-Wise Split",
        "",
        "Requested train records:",
        "",
        _df_to_markdown(requested_train_records),
        "",
        "Requested holdout records:",
        "",
        _df_to_markdown(requested_holdout_records),
        "",
        "Informative train records:",
        "",
        _df_to_markdown(train_records),
        "",
        "Informative holdout records:",
        "",
        _df_to_markdown(holdout_records),
        "",
        "Skipped records:",
        "",
        _df_to_markdown(result["skipped_records"]),
        "",
        "## Beat Label Counts",
        "",
        "Train:",
        "",
        _df_to_markdown(train_counts),
        "",
        "Holdout:",
        "",
        _df_to_markdown(holdout_counts),
        "",
        "## Tuned Raw ML Decoder Thresholds",
        "",
        _df_to_markdown(threshold_rows),
        "",
        "## Tuned FP-Aware Decoder Thresholds",
        "",
        _df_to_markdown(fpaware_threshold_rows),
        "",
        "## Hard Negative Training",
        "",
        _df_to_markdown(
            pd.DataFrame(
                [
                    {
                        "hard_negative_beats": result["hard_negative_count"],
                        "hard_negative_boost": result["hard_negative_boost"],
                    }
                ]
            )
        ),
        "",
        "## Tuned Guard Config",
        "",
        _df_to_markdown(guarded_rows),
        "",
        "## PASM-AI Episode Reranker",
        "",
        "The reranker is a lightweight NumPy logistic model trained on candidate episodes labelled by episode IoU.",
        "`safe` keeps relaxed ectopy diagnostic-only; `v2` uses pattern-aware relaxed rescue.",
        "",
        _df_to_markdown(reranker_threshold_rows),
        "",
        "Episode hard negatives:",
        "",
        _df_to_markdown(
            pd.DataFrame(
                [
                    {
                        "hard_negative_candidates": result["reranker_hard_negative_count"],
                        "hard_negative_boost": result["reranker_hard_negative_boost"],
                    }
                ]
            )
        ),
        "",
        "Candidate labels:",
        "",
        _df_to_markdown(reranker_candidate_label_counts(result["reranker_train_df"], result["reranker_holdout_df"])),
        "",
        "Accepted candidates by source:",
        "",
        _df_to_markdown(reranker_source_summary(result["reranker_train_df"], result["reranker_holdout_df"])),
        "",
        "Uncovered truth episodes:",
        "",
        _df_to_markdown(reranker_uncovered_summary(result["reranker_train_uncovered"], result["reranker_holdout_uncovered"])),
        "",
        "Top coefficients:",
        "",
        _df_to_markdown(top_reranker_coefficients(result["reranker_v2_model"])),
        "",
        "AI rescued vs rejected:",
        "",
        _df_to_markdown(ai_rescue_reject_summary(result)),
        "",
        "MITDB leave-one-record-out reranker v2:",
        "",
        _df_to_markdown(result["reranker_loro"]),
        "",
        "## Train Summary",
        "",
        _df_to_markdown(result["train_summary"]),
        "",
        "## Holdout Summary",
        "",
        _df_to_markdown(result["holdout_summary"]),
        "",
        "## Holdout Per-Record Metrics",
        "",
        _df_to_markdown(_per_record_macro(result["holdout_metrics"])),
        "",
        "## Holdout FP/h By Record",
        "",
        _df_to_markdown(false_alarms_per_record(result["holdout_metrics"])),
        "",
        "## Holdout False Positives By Type",
        "",
        _df_to_markdown(false_positives_by_type(holdout_predictions)),
        "",
        "## Holdout False Positives By Failure Stage",
        "",
        _df_to_markdown(false_positives_by_failure_stage(holdout_predictions)),
        "",
        "## FP Removed By Guard Reason",
        "",
        _df_to_markdown(fp_removed_by_guard_reason(result["guard_removed"])),
        "",
        "## FP Removed By PASM-AI",
        "",
        _df_to_markdown(fp_removed_by_reranker(result["reranker_removed"])),
        "",
        "## Top Holdout False-Positive Episodes",
        "",
        _df_to_markdown(top_false_positive_episodes(holdout_predictions)),
        "",
        "## Interpretation",
        "",
        "- `pasm_physionet` is the deterministic PASM baseline with PhysioNet evidence postprocessing.",
        "- `pasm_ml_decoder` is the first learned PASM scorer: softmax regression on PASM feature tables.",
        "- `pasm_ml_decoder_guarded` adds normal bias and conservative episode filters to reduce false positives.",
        "- `pasm_ml_decoder_fpaware` retrains with capped class weights and hard-negative normal beats from train false positives.",
        "- `pasm_ai_reranker_safe` is the conservative candidate-level explainable AI layer.",
        "- `pasm_ai_reranker_v2` adds pattern-aware relaxed ectopy rescue with episode-level hard negatives.",
        "- If learned variants underperform the baseline, keep them as experimental scaffolds and expand patient-wise data before moving to TCN/Transformer.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def guarded_config_table(config):
    rows = [
        {"parameter": "normal_bias", "value": config.get("normal_bias", 0.0)},
        {"parameter": "min_episode_confidence", "value": config.get("min_episode_confidence", 0.0)},
        {"parameter": "min_episode_sqi", "value": config.get("min_episode_sqi", 0.0)},
        {"parameter": "ectopy_min_morph_z", "value": config.get("ectopy_min_morph_z", 0.0)},
        {"parameter": "ectopy_min_delta_rr_z_abs", "value": config.get("ectopy_min_delta_rr_z_abs", 0.0)},
        {"parameter": "ectopy_min_score", "value": config.get("ectopy_min_score", 0.0)},
    ]
    for state, beats in sorted(config.get("min_beats_by_state", {}).items()):
        rows.append({"parameter": f"min_beats_{state}", "value": beats})
    return pd.DataFrame(rows)


def reranker_candidate_label_counts(train_df, holdout_df):
    frames = []
    for split, frame in [("train", train_df), ("holdout", holdout_df)]:
        if frame is None or len(frame) == 0:
            continue
        tmp = frame.copy()
        tmp["split"] = split
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["split", "accepted", "candidates"])
    df = pd.concat(frames, ignore_index=True)
    return (
        df.groupby(["split", "accepted"])
        .size()
        .reset_index(name="candidates")
        .sort_values(["split", "accepted"])
        .reset_index(drop=True)
    )


def reranker_source_summary(train_df, holdout_df):
    frames = []
    for split, frame in [("train", train_df), ("holdout", holdout_df)]:
        if frame is None or len(frame) == 0:
            continue
        tmp = frame.copy()
        tmp["split"] = split
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["split", "source", "pattern", "accepted", "candidates"])
    df = pd.concat(frames, ignore_index=True)
    return (
        df.groupby(["split", "source", "pattern", "accepted"])
        .size()
        .reset_index(name="candidates")
        .sort_values(["split", "source", "pattern", "accepted"])
        .reset_index(drop=True)
    )


def reranker_uncovered_summary(train_uncovered, holdout_uncovered):
    frames = []
    for split, frame in [("train", train_uncovered), ("holdout", holdout_uncovered)]:
        if frame is None or len(frame) == 0:
            continue
        tmp = frame.copy()
        tmp["split"] = split
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=["split", "type", "uncovered_truth"])
    df = pd.concat(frames, ignore_index=True)
    return (
        df.groupby(["split", "type"])
        .size()
        .reset_index(name="uncovered_truth")
        .sort_values(["split", "type"])
        .reset_index(drop=True)
    )


def top_reranker_coefficients(model, n=12):
    if model is None:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    return model.coefficients().head(n).reset_index(drop=True)


def ai_rescue_reject_summary(result):
    predictions = result.get("holdout_predictions", pd.DataFrame())
    removed = result.get("reranker_removed", pd.DataFrame())
    uncovered = result.get("reranker_holdout_uncovered", pd.DataFrame())
    rows = []
    if predictions is not None and len(predictions):
        baseline_tp = predictions[(predictions["model"] == "pasm_physionet") & (predictions["status"] == "TP")]
        v2_relaxed_tp = predictions[
            (predictions["model"] == "pasm_ai_reranker_v2")
            & (predictions["status"] == "TP")
            & (predictions.get("source", "") == "relaxed_ectopy")
        ]
        rows.append({"metric": "baseline_tp", "count": int(len(baseline_tp))})
        rows.append({"metric": "relaxed_tp_rescued", "count": int(len(v2_relaxed_tp))})
    else:
        rows.extend(
            [
                {"metric": "baseline_tp", "count": 0},
                {"metric": "relaxed_tp_rescued", "count": 0},
            ]
        )
    if removed is not None and len(removed):
        rejected = removed[(removed["model"] == "pasm_ai_reranker_v2") & (removed["source"] == "relaxed_ectopy")]
        rows.append({"metric": "relaxed_candidates_rejected", "count": int(rejected["removed"].sum())})
    else:
        rows.append({"metric": "relaxed_candidates_rejected", "count": 0})
    rows.append({"metric": "uncovered_truth", "count": int(len(uncovered)) if uncovered is not None else 0})
    return pd.DataFrame(rows)


def false_alarms_per_record(metrics):
    faph = metrics[metrics["type"] == "false_alarms_per_hour"].copy()
    if len(faph) == 0:
        return pd.DataFrame(columns=["record_id", "model", "false_alarms_per_hour"])
    return (
        faph[["record_id", "model", "f1"]]
        .rename(columns={"f1": "false_alarms_per_hour"})
        .sort_values(["record_id", "model"])
        .reset_index(drop=True)
    )


def false_positives_by_type(predictions):
    fp = _false_positive_rows(predictions)
    if len(fp) == 0:
        return pd.DataFrame(columns=["model", "type", "false_positives"])
    return (
        fp.groupby(["model", "type"])
        .size()
        .reset_index(name="false_positives")
        .sort_values(["model", "false_positives", "type"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def false_positives_by_failure_stage(predictions):
    fp = _false_positive_rows(predictions)
    if len(fp) == 0:
        return pd.DataFrame(columns=["model", "failure_stage", "false_positives"])
    return (
        fp.groupby(["model", "failure_stage"])
        .size()
        .reset_index(name="false_positives")
        .sort_values(["model", "false_positives", "failure_stage"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def fp_removed_by_guard_reason(removed):
    if removed is None or len(removed) == 0:
        return pd.DataFrame(columns=["model", "type", "reason", "removed"])
    return (
        removed.groupby(["model", "type", "reason"])["removed"]
        .sum()
        .reset_index()
        .sort_values(["model", "removed", "type", "reason"], ascending=[True, False, True, True])
        .reset_index(drop=True)
    )


def fp_removed_by_reranker(removed):
    if removed is None or len(removed) == 0:
        return pd.DataFrame(columns=["model", "type", "source", "pattern", "reason", "removed"])
    return (
        removed.groupby(["model", "type", "source", "pattern", "reason"])["removed"]
        .sum()
        .reset_index()
        .sort_values(["model", "removed", "type", "source"], ascending=[True, False, True, True])
        .reset_index(drop=True)
    )


def top_false_positive_episodes(predictions, n=12):
    fp = _false_positive_rows(predictions)
    cols = [
        "record_id",
        "model",
        "type",
        "start_s",
        "end_s",
        "duration_s",
        "confidence",
        "mean_sqi",
        "beats",
        "best_iou",
        "failure_stage",
    ]
    if len(fp) == 0:
        return pd.DataFrame(columns=cols)
    return (
        fp.sort_values(["confidence", "duration_s"], ascending=[False, False])
        .head(n)[cols]
        .reset_index(drop=True)
    )


def _false_positive_rows(predictions):
    if predictions is None or len(predictions) == 0:
        return pd.DataFrame()
    return predictions[predictions["status"] == "FP"].copy()


def _label_counts(frame):
    if len(frame) == 0:
        return pd.DataFrame(columns=["label", "beats"])
    return (
        frame["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="beats")
        .sort_values("label")
        .reset_index(drop=True)
    )


def _per_record_macro(metrics):
    macro = metrics[metrics["type"] == "macro"].copy()
    faph = metrics[metrics["type"] == "false_alarms_per_hour"][["record_id", "model", "f1"]].rename(
        columns={"f1": "false_alarms_per_hour"}
    )
    out = macro.merge(faph, on=["record_id", "model"], how="left")
    cols = ["record_id", "model", "tp", "fp", "fn", "precision", "recall", "f1", "false_alarms_per_hour"]
    return out[cols].sort_values(["record_id", "model"]).reset_index(drop=True)


def _df_to_markdown(df):
    if df is None or len(df) == 0:
        return "_No rows._"
    cols = list(df.columns)

    def fmt(value):
        if isinstance(value, float) or isinstance(value, np.floating):
            if np.isnan(value):
                return ""
            return f"{value:.3f}"
        return str(value)

    rows = [[fmt(v) for v in row] for row in df.to_numpy()]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run PASM-Rhythm patient-wise ML validation.")
    parser.add_argument("--preset", choices=sorted(ML_BENCHMARK_PRESETS), default="tiny")
    parser.add_argument("--out", default="PASM_ML_VALIDATION.md")
    parser.add_argument("--train-records", nargs="+", default=None)
    parser.add_argument("--holdout-records", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--normal-bias", type=float, default=DEFAULT_GUARDED_CONFIG["normal_bias"])
    parser.add_argument("--min-episode-confidence", type=float, default=DEFAULT_GUARDED_CONFIG["min_episode_confidence"])
    parser.add_argument("--min-episode-sqi", type=float, default=DEFAULT_GUARDED_CONFIG["min_episode_sqi"])
    parser.add_argument("--model-out", default=None)
    args = parser.parse_args(argv)

    preset = args.preset
    train_record_ids = args.train_records
    holdout_record_ids = args.holdout_records
    if train_record_ids is not None or holdout_record_ids is not None:
        if train_record_ids is None or holdout_record_ids is None:
            raise ValueError("Both --train-records and --holdout-records are required for a custom split.")
        preset = None
    else:
        train_record_ids, holdout_record_ids = resolve_ml_preset(preset)

    guarded_config = dict(DEFAULT_GUARDED_CONFIG)
    guarded_config["normal_bias"] = args.normal_bias
    guarded_config["min_episode_confidence"] = args.min_episode_confidence
    guarded_config["min_episode_sqi"] = args.min_episode_sqi
    result = run_ml_validation(
        train_record_ids=train_record_ids,
        holdout_record_ids=holdout_record_ids,
        preset=preset,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        guarded_config=guarded_config,
    )
    write_ml_validation_report(args.out, result)
    if args.model_out:
        result["model"].save_npz(args.model_out)
    print(result["holdout_summary"].to_string(index=False))
    print(f"Wrote {args.out}")
    if args.model_out:
        print(f"Wrote {args.model_out}")


if __name__ == "__main__":
    main()
