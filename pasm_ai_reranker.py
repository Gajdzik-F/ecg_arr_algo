from dataclasses import dataclass

import numpy as np
import pandas as pd

from pasm_physionet import (
    detect_ectopy_candidates,
    filter_predictions_for_annotation_scope,
    merge_close_same_type_episodes,
    score_ectopy_candidates,
)
from pasm_validation import episode_iou, normalize_episode_types


RERANKER_PATTERN_VALUES = (
    "baseline",
    "short_coupled_run",
    "premature_plus_pause",
    "morphology_cluster",
    "rr_irregular_burst",
    "state_score_segment",
)
RERANKER_TYPE_VALUES = (
    "af_like",
    "ectopic_like",
    "noise_uncertain",
    "sinus_tachy",
    "sinus_brady",
)
RERANKER_FEATURE_COLUMNS = (
    "duration_s",
    "beats",
    "confidence",
    "mean_state_score",
    "max_state_score",
    "rr_support",
    "pause_support",
    "morph_support",
    "density_support",
    "mean_rr_prev",
    "min_rr_prev",
    "local_cv",
    "local_rmssd",
    "mean_morph_z",
    "mean_sqi",
    "rr_pause_product",
    "morph_density_product",
    "short_episode_flag",
    "long_cluster_flag",
    "baseline_candidate_flag",
) + tuple(f"type_{value}" for value in RERANKER_TYPE_VALUES) + tuple(
    f"pattern_{value}" for value in RERANKER_PATTERN_VALUES
)


@dataclass
class EpisodeReranker:
    feature_columns: tuple
    fill_values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_accept_proba(self, candidate_df):
        x = _prepare_features(candidate_df, self.feature_columns, self.fill_values, self.mean, self.scale)
        logits = x @ self.weights + float(self.bias)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    def top_feature_names(self, candidate_row, n=3):
        if candidate_row is None or len(self.feature_columns) == 0:
            return ""
        frame = pd.DataFrame([candidate_row])
        x = _prepare_features(frame, self.feature_columns, self.fill_values, self.mean, self.scale)[0]
        contrib = x * self.weights
        order = np.argsort(np.abs(contrib))[::-1][: int(n)]
        return ",".join(str(self.feature_columns[i]) for i in order if np.isfinite(contrib[i]))

    def coefficients(self):
        return pd.DataFrame(
            {
                "feature": list(self.feature_columns),
                "coefficient": self.weights.astype(float),
                "abs_coefficient": np.abs(self.weights.astype(float)),
            }
        ).sort_values("abs_coefficient", ascending=False)

    def save_npz(self, path):
        np.savez(
            path,
            feature_columns=np.asarray(self.feature_columns, dtype=object),
            fill_values=self.fill_values,
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            bias=np.asarray([self.bias], dtype=float),
        )

    @classmethod
    def load_npz(cls, path):
        data = np.load(path, allow_pickle=True)
        return cls(
            feature_columns=tuple(data["feature_columns"].tolist()),
            fill_values=np.asarray(data["fill_values"], dtype=float),
            mean=np.asarray(data["mean"], dtype=float),
            scale=np.asarray(data["scale"], dtype=float),
            weights=np.asarray(data["weights"], dtype=float),
            bias=float(np.asarray(data["bias"], dtype=float)[0]),
        )


def fit_episode_reranker(
    candidate_df,
    feature_columns=None,
    label_column="accepted",
    epochs=800,
    lr=0.05,
    l2=1e-3,
    seed=2026,
    max_class_weight=8.0,
    sample_weight_boost=None,
):
    feature_columns = tuple(feature_columns or RERANKER_FEATURE_COLUMNS)
    if candidate_df is None or len(candidate_df) == 0:
        raise ValueError("Cannot train PASM-AI reranker on an empty candidate frame.")
    if label_column not in candidate_df:
        raise ValueError(f"Missing reranker label column {label_column!r}.")
    y = candidate_df[label_column].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("PASM-AI reranker needs both accepted and rejected candidates.")

    fill_values, mean, scale = _fit_normalizer(candidate_df, feature_columns)
    x = _prepare_features(candidate_df, feature_columns, fill_values, mean, scale)
    sample_weight = _balanced_binary_weights(y, max_class_weight=max_class_weight)
    if sample_weight_boost is not None:
        boost = np.asarray(sample_weight_boost, dtype=float)
        if len(boost) != len(candidate_df):
            raise ValueError("sample_weight_boost must align 1:1 with candidate_df.")
        sample_weight = sample_weight * np.where(np.isfinite(boost) & (boost > 0.0), boost, 1.0)
    weight_norm = max(float(sample_weight.sum()), 1e-12)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=x.shape[1])
    bias = 0.0
    for _ in range(int(epochs)):
        logits = x @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        err = (probs - y.astype(float)) * (sample_weight / weight_norm)
        grad_w = x.T @ err + float(l2) * weights
        grad_b = float(err.sum())
        weights -= float(lr) * grad_w
        bias -= float(lr) * grad_b

    return EpisodeReranker(
        feature_columns=feature_columns,
        fill_values=fill_values,
        mean=mean,
        scale=scale,
        weights=weights.astype(float),
        bias=float(bias),
    )


def build_episode_candidate_dataset(records, pipelines, split="train", iou_threshold=0.30):
    rows = []
    uncovered_rows = []
    for record in records:
        pipeline = pipelines[record.record_id]
        candidates = build_episode_candidates(record, pipeline)
        labeled, uncovered = label_episode_candidates(
            candidates,
            record.truth_episodes,
            record_id=record.record_id,
            split=split,
            iou_threshold=iou_threshold,
        )
        if len(labeled):
            rows.append(labeled)
        if len(uncovered):
            uncovered_rows.append(uncovered)
    dataset = pd.concat(rows, ignore_index=True) if rows else _empty_candidate_frame()
    uncovered = pd.concat(uncovered_rows, ignore_index=True) if uncovered_rows else _empty_uncovered_frame()
    return dataset, uncovered


def build_episode_candidates(record, pipeline, state_score_min=0.54, min_segment_beats=3):
    frames = []
    baseline = _candidate_rows_from_episodes(
        pipeline.get("episodes"),
        record=record,
        pipeline=pipeline,
        source="baseline",
        default_pattern="baseline",
    )
    if len(baseline):
        frames.append(baseline)

    if record.record_id.startswith("mitdb/"):
        relaxed = detect_ectopy_candidates(
            pipeline["features"],
            beats=pipeline.get("beats"),
            patient_memory=pipeline.get("patient_memory"),
            config={
                "recording_duration_s": _record_duration_s(record),
                "min_pause_ratio": 1.18,
                "dense_morph_z": 0.55,
                "dense_min_short": 1,
            },
        )
        relaxed = score_ectopy_candidates(relaxed, config={"recording_duration_s": _record_duration_s(record)})
        if len(relaxed):
            frames.append(
                _candidate_rows_from_episodes(relaxed, record=record, pipeline=pipeline, source="relaxed_ectopy")
            )

    state_segments = _state_score_candidates(pipeline, state_score_min=state_score_min, min_segment_beats=min_segment_beats)
    if len(state_segments):
        frames.append(
            _candidate_rows_from_episodes(
                state_segments,
                record=record,
                pipeline=pipeline,
                source="state_score",
                default_pattern="state_score_segment",
            )
        )

    if not frames:
        return _empty_candidate_frame()
    out = pd.concat(frames, ignore_index=True)
    out = filter_predictions_for_annotation_scope(record, out)
    dedupe_cols = ["record_id", "type", "start_s", "end_s", "pattern", "source"]
    return out.sort_values(["start_s", "end_s", "source"]).drop_duplicates(dedupe_cols).reset_index(drop=True)


def label_episode_candidates(candidates, truth_episodes, record_id="", split="", iou_threshold=0.30):
    candidates = normalize_episode_types(candidates).reset_index(drop=True)
    truth = normalize_episode_types(truth_episodes).reset_index(drop=True)
    if len(candidates) == 0:
        return _empty_candidate_frame(), _uncovered_truth_rows(truth, record_id, split)

    accepted = []
    matched_truth = set()
    best_ious = []
    best_truth = []
    for _, cand in candidates.iterrows():
        best_iou = 0.0
        best_idx = ""
        for ti, tr in truth.iterrows():
            if cand.get("type") != tr.get("type"):
                continue
            iou = episode_iou(cand, tr)
            if iou > best_iou:
                best_iou = float(iou)
                best_idx = int(ti)
        ok = best_iou >= float(iou_threshold)
        accepted.append(1 if ok else 0)
        best_ious.append(best_iou)
        best_truth.append(best_idx if ok else "")
        if ok:
            matched_truth.add(best_idx)

    out = candidates.copy()
    out["split"] = split
    out["accepted"] = accepted
    out["best_iou"] = best_ious
    out["matched_truth_index"] = best_truth
    uncovered = truth.loc[[i for i in truth.index if i not in matched_truth]].copy()
    return out, _uncovered_truth_rows(uncovered, record_id, split)


def predict_reranked_episodes(record, pipeline, reranker, threshold=0.50, candidate_policy="safe", policy_config=None):
    candidates = build_episode_candidates(record, pipeline)
    if len(candidates) == 0:
        return candidates
    out = candidates.copy().reset_index(drop=True)
    proba = reranker.predict_accept_proba(out)
    keep = reranker_accept_mask(
        out,
        proba,
        threshold=threshold,
        candidate_policy=candidate_policy,
        policy_config=policy_config,
    )
    out["ai_accept_proba"] = proba
    out["ai_decision"] = np.where(keep, "accept", "reject")
    out["ai_top_features"] = [reranker.top_feature_names(row) for row in out.to_dict("records")]
    kept = out.loc[keep].copy()
    if len(kept) == 0:
        return kept.reset_index(drop=True)
    kept["confidence"] = np.maximum(kept["confidence"].fillna(0.0).astype(float), kept["ai_accept_proba"].astype(float))
    kept = merge_close_same_type_episodes(kept, merge_gap_s=1.0)
    return filter_predictions_for_annotation_scope(record, kept.reset_index(drop=True))


def reranker_accept_mask(candidates, proba, threshold=0.50, candidate_policy="safe", policy_config=None):
    out = candidates.copy().reset_index(drop=True)
    keep = np.asarray(proba, dtype=float) >= float(threshold)
    if candidate_policy is None:
        return keep
    if candidate_policy == "balanced":
        candidate_policy = "safe"
    if candidate_policy not in {"safe", "pattern_v2"}:
        raise ValueError(f"Unknown reranker candidate policy: {candidate_policy!r}")

    typ = out.get("type", pd.Series([""] * len(out))).astype(str).to_numpy()
    source = out.get("source", pd.Series([""] * len(out))).astype(str).to_numpy()
    pattern = out.get("pattern", pd.Series([""] * len(out))).astype(str).to_numpy()
    duration = out.get("duration_s", pd.Series(np.zeros(len(out)))).fillna(0.0).to_numpy(dtype=float)
    beats = out.get("beats", pd.Series(np.zeros(len(out)))).fillna(0.0).to_numpy(dtype=float)
    rr = out.get("rr_support", pd.Series(np.zeros(len(out)))).fillna(0.0).to_numpy(dtype=float)
    pause = out.get("pause_support", pd.Series(np.zeros(len(out)))).fillna(0.0).to_numpy(dtype=float)
    morph = out.get("morph_support", pd.Series(np.zeros(len(out)))).fillna(0.0).to_numpy(dtype=float)
    density = out.get("density_support", pd.Series(np.zeros(len(out)))).fillna(0.0).to_numpy(dtype=float)

    relaxed_ectopy = (typ == "ectopic_like") & (source == "relaxed_ectopy")
    baseline_candidate = source == "baseline"
    keep |= baseline_candidate
    if candidate_policy == "safe":
        # V1 trains on broad relaxed candidates, but the safe inference policy
        # keeps every non-baseline candidate diagnostic-only.
        return baseline_candidate.copy()

    cfg = dict(policy_config or {})
    relaxed_threshold = np.asarray(proba, dtype=float) >= max(
        float(threshold),
        float(cfg.get("relaxed_min_proba", 0.82)),
    )
    short_run_ok = (
        (pattern == "short_coupled_run")
        & (duration <= float(cfg.get("short_coupled_max_duration_s", 1.20)))
        & (beats <= float(cfg.get("short_coupled_max_beats", 3)))
        & (rr >= float(cfg.get("short_coupled_min_rr_support", 0.26)))
    )
    pause_ok = (
        (pattern == "premature_plus_pause")
        & (duration <= float(cfg.get("pause_max_duration_s", 1.80)))
        & (beats <= float(cfg.get("pause_max_beats", 4)))
        & (rr >= float(cfg.get("pause_min_rr_support", 0.18)))
        & (pause >= float(cfg.get("pause_min_pause_support", 0.30)))
    )
    morph_cluster_ok = (
        (pattern == "morphology_cluster")
        & (duration <= float(cfg.get("morph_cluster_max_duration_s", 2.40)))
        & (beats <= float(cfg.get("morph_cluster_max_beats", 6)))
        & (morph >= float(cfg.get("morph_cluster_min_morph_support", 0.95)))
        & (density >= float(cfg.get("morph_cluster_min_density_support", 0.85)))
    )
    relaxed_ok = relaxed_ectopy & relaxed_threshold & (short_run_ok | pause_ok | morph_cluster_ok)
    keep &= ~relaxed_ectopy
    keep |= relaxed_ok
    keep |= baseline_candidate

    state_score_mitdb_ectopy = (typ == "ectopic_like") & (source == "state_score")
    keep &= ~state_score_mitdb_ectopy
    return keep


def tune_reranker_threshold(
    records,
    pipelines,
    reranker,
    thresholds=None,
    fp_per_hour_limit=9.0,
    candidate_policy="safe",
    policy_config=None,
    model_name="pasm_ai_reranker_safe",
):
    from pasm_validation import evaluate_episodes, summarize_benchmark

    thresholds = tuple(thresholds or (0.30, 0.40, 0.50, 0.60, 0.70, 0.80))
    best_threshold = thresholds[0]
    best_score = -np.inf
    best_metrics = None
    for threshold in thresholds:
        rows = []
        for record in records:
            episodes = predict_reranked_episodes(
                record,
                pipelines[record.record_id],
                reranker,
                threshold=threshold,
                candidate_policy=candidate_policy,
                policy_config=policy_config,
            )
            metrics = evaluate_episodes(
                episodes,
                normalize_episode_types(record.truth_episodes),
                duration_s=_record_duration_s(record),
            )
            metrics.insert(0, "model", model_name)
            metrics.insert(0, "record_id", record.record_id)
            rows.append(metrics)
        metrics = pd.concat(rows, ignore_index=True)
        summary = summarize_benchmark(metrics)
        row = summary.iloc[0]
        f1 = float(row["episode_f1_mean"])
        faph = float(row["false_alarms_per_hour_mean"])
        fp_penalty = max(0.0, faph - float(fp_per_hour_limit))
        score = f1 - 0.004 * faph - 0.05 * fp_penalty
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def tune_pattern_policy(records, pipelines, reranker, target_fp_per_hour=9.0):
    thresholds = (0.70,)
    relaxed_min = (0.90,)
    short_rr = (0.26,)
    pause_min = (0.35,)
    best = {
        "threshold": 0.80,
        "policy_config": {"relaxed_min_proba": 0.88},
        "metrics": None,
        "score": -np.inf,
    }
    from pasm_validation import summarize_benchmark

    for threshold in thresholds:
        for relaxed_proba in relaxed_min:
            for rr_min in short_rr:
                for pause_support in pause_min:
                    cfg = {
                        "relaxed_min_proba": relaxed_proba,
                        "short_coupled_min_rr_support": rr_min,
                        "pause_min_pause_support": pause_support,
                    }
                    _, metrics = tune_reranker_threshold(
                        records,
                        pipelines,
                        reranker,
                        thresholds=(threshold,),
                        fp_per_hour_limit=target_fp_per_hour,
                        candidate_policy="pattern_v2",
                        policy_config=cfg,
                        model_name="pasm_ai_reranker_v2",
                    )
                    summary = summarize_benchmark(metrics)
                    row = summary.iloc[0]
                    f1 = float(row["episode_f1_mean"])
                    faph = float(row["false_alarms_per_hour_mean"])
                    fp_penalty = max(0.0, faph - float(target_fp_per_hour))
                    score = f1 - 0.004 * faph - 0.08 * fp_penalty
                    if score > best["score"]:
                        best = {
                            "threshold": float(threshold),
                            "policy_config": cfg,
                            "metrics": metrics,
                            "score": float(score),
                        }
    return best["threshold"], best["policy_config"], best["metrics"]


def _candidate_rows_from_episodes(episodes, record, pipeline, source, default_pattern="baseline"):
    if episodes is None or len(episodes) == 0:
        return _empty_candidate_frame()
    episodes = normalize_episode_types(episodes).reset_index(drop=True)
    rows = []
    for _, episode in episodes.iterrows():
        row = _episode_feature_row(episode, record=record, pipeline=pipeline)
        row["source"] = source
        row["pattern"] = str(episode.get("pattern", "") or default_pattern)
        rows.append(row)
    return pd.DataFrame(rows)


def _episode_feature_row(episode, record, pipeline):
    start_s = float(episode.get("start_s", np.nan))
    end_s = float(episode.get("end_s", np.nan))
    typ = str(episode.get("type", ""))
    features = pipeline.get("features", pd.DataFrame())
    state_scores = pipeline.get("state_scores", pd.DataFrame())
    seg = _time_segment(features, start_s, end_s)
    score_seg = _time_segment(state_scores, start_s, end_s, times=features.get("time_s") if "time_s" in features else None)
    type_scores = score_seg[typ].astype(float) if len(score_seg) and typ in score_seg else pd.Series(dtype=float)

    row = {
        "record_id": record.record_id,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": max(0.0, end_s - start_s) if np.isfinite(start_s) and np.isfinite(end_s) else 0.0,
        "type": typ,
        "confidence": _finite_or_default(episode.get("confidence", np.nan), 0.0),
        "beats": _finite_or_default(episode.get("beats", len(seg)), len(seg)),
        "mean_state_score": float(type_scores.mean()) if len(type_scores) else 0.0,
        "max_state_score": float(type_scores.max()) if len(type_scores) else 0.0,
        "rr_support": _finite_or_default(episode.get("rr_support", np.nan), 0.0),
        "pause_support": _finite_or_default(episode.get("pause_support", np.nan), 0.0),
        "morph_support": _finite_or_default(episode.get("morph_support", np.nan), 0.0),
        "density_support": _finite_or_default(episode.get("density_support", np.nan), 0.0),
        "mean_rr_prev": _feature_mean(seg, "rr_prev"),
        "min_rr_prev": _feature_min(seg, "rr_prev"),
        "local_cv": _feature_mean(seg, "local_cv"),
        "local_rmssd": _feature_mean(seg, "local_rmssd"),
        "mean_morph_z": _finite_or_default(episode.get("mean_morph_z", np.nan), 0.0),
        "mean_sqi": _finite_or_default(episode.get("mean_sqi", np.nan), _feature_mean(seg, "sqi", default=1.0)),
    }
    pattern = str(episode.get("pattern", "") or "baseline")
    row["rr_pause_product"] = float(row["rr_support"] * row["pause_support"])
    row["morph_density_product"] = float(row["morph_support"] * row["density_support"])
    row["short_episode_flag"] = 1.0 if row["duration_s"] <= 1.5 and row["beats"] <= 4 else 0.0
    row["long_cluster_flag"] = 1.0 if pattern == "morphology_cluster" and (row["duration_s"] > 2.5 or row["beats"] > 6) else 0.0
    row["baseline_candidate_flag"] = 1.0 if pattern == "baseline" else 0.0
    for value in RERANKER_TYPE_VALUES:
        row[f"type_{value}"] = 1.0 if typ == value else 0.0
    for value in RERANKER_PATTERN_VALUES:
        row[f"pattern_{value}"] = 1.0 if pattern == value else 0.0
    return row


def _state_score_candidates(pipeline, state_score_min=0.54, min_segment_beats=3):
    features = pipeline.get("features", pd.DataFrame())
    state_scores = pipeline.get("state_scores", pd.DataFrame())
    if len(features) == 0 or len(state_scores) == 0 or "time_s" not in features:
        return pd.DataFrame()
    times = features["time_s"].to_numpy(dtype=float)
    rows = []
    for typ in RERANKER_TYPE_VALUES:
        if typ not in state_scores:
            continue
        scores = state_scores[typ].fillna(0.0).to_numpy(dtype=float)
        flags = scores >= float(state_score_min)
        start = None
        for i, ok in enumerate(flags):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                _append_state_segment(rows, typ, start, i - 1, times, scores, min_segment_beats)
                start = None
        if start is not None:
            _append_state_segment(rows, typ, start, len(flags) - 1, times, scores, min_segment_beats)
    return pd.DataFrame(rows)


def _append_state_segment(rows, typ, start, end, times, scores, min_segment_beats):
    if end - start + 1 < int(min_segment_beats):
        return
    rows.append(
        {
            "start_s": float(times[start]),
            "end_s": float(times[end]),
            "type": typ,
            "confidence": float(np.mean(scores[start : end + 1])),
            "beats": int(end - start + 1),
            "mean_sqi": 1.0,
            "pattern": "state_score_segment",
            "reason": "state score candidate",
        }
    )


def _time_segment(frame, start_s, end_s, times=None):
    if frame is None or len(frame) == 0 or not np.isfinite(start_s) or not np.isfinite(end_s):
        return frame.iloc[0:0] if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if times is None:
        if "time_s" not in frame:
            return frame.iloc[0:0]
        times = frame["time_s"]
    times = np.asarray(times, dtype=float)
    mask = (times >= float(start_s)) & (times <= float(end_s))
    return frame.loc[mask]


def _feature_mean(frame, column, default=0.0):
    if frame is None or len(frame) == 0 or column not in frame:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else float(default)


def _feature_min(frame, column, default=0.0):
    if frame is None or len(frame) == 0 or column not in frame:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.min(values)) if len(values) else float(default)


def _finite_or_default(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _fit_normalizer(frame, feature_columns):
    x = _raw_feature_matrix(frame, feature_columns)
    fill_values = np.nanmedian(x, axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    filled = np.where(np.isfinite(x), x, fill_values[None, :])
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return fill_values.astype(float), mean.astype(float), scale.astype(float)


def _prepare_features(frame, feature_columns, fill_values, mean, scale):
    x = _raw_feature_matrix(frame, feature_columns)
    x = np.where(np.isfinite(x), x, fill_values[None, :])
    return (x - mean[None, :]) / scale[None, :]


def _raw_feature_matrix(frame, feature_columns):
    out = pd.DataFrame(index=frame.index if frame is not None else range(0))
    for col in feature_columns:
        out[col] = pd.to_numeric(frame[col], errors="coerce") if frame is not None and col in frame else np.nan
    return out.loc[:, list(feature_columns)].astype(float).to_numpy()


def _balanced_binary_weights(y, max_class_weight=8.0):
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights_by_class = np.zeros(2, dtype=float)
    present = counts > 0
    weights_by_class[present] = len(y) / (float(present.sum()) * counts[present])
    if max_class_weight is not None:
        weights_by_class[present] = np.minimum(weights_by_class[present], float(max_class_weight))
    return weights_by_class[y]


def _uncovered_truth_rows(truth, record_id, split):
    if truth is None or len(truth) == 0:
        return _empty_uncovered_frame()
    rows = []
    for idx, row in truth.iterrows():
        rows.append(
            {
                "record_id": record_id,
                "split": split,
                "truth_index": int(idx),
                "type": row.get("type", ""),
                "start_s": float(row.get("start_s", np.nan)),
                "end_s": float(row.get("end_s", np.nan)),
                "status": "uncovered_truth",
            }
        )
    return pd.DataFrame(rows)


def _empty_candidate_frame():
    cols = [
        "record_id",
        "split",
        "source",
        "start_s",
        "end_s",
        "duration_s",
        "type",
        "pattern",
        "accepted",
        "best_iou",
        "matched_truth_index",
    ] + list(RERANKER_FEATURE_COLUMNS)
    return pd.DataFrame(columns=list(dict.fromkeys(cols)))


def _empty_uncovered_frame():
    return pd.DataFrame(columns=["record_id", "split", "truth_index", "type", "start_s", "end_s", "status"])


def _record_duration_s(record):
    if hasattr(record, "signal") and hasattr(record, "fs"):
        return float(len(record.signal)) / float(record.fs)
    if hasattr(record, "r_times") and len(record.r_times):
        return float(record.r_times[-1] - record.r_times[0])
    return 0.0
