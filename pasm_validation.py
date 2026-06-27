import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pasm_rhythm import DEFAULT_MIN_CONFIDENCE_BY_STATE, run_pasm_rhythm


CANONICAL_TYPES = (
    "sinus_tachy",
    "sinus_brady",
    "af_like",
    "ectopic_like",
    "noise_uncertain",
)


TYPE_ALIASES = {
    "tachy": "sinus_tachy",
    "brady": "sinus_brady",
    "ectopic_or_outlier_run": "ectopic_like",
}


@dataclass
class SyntheticRecord:
    record_id: str
    r_times: np.ndarray
    rr_prev: np.ndarray
    rr_next: np.ndarray
    beats: np.ndarray
    sqi_at_r: np.ndarray
    rpeak_uncertainty: np.ndarray
    beat_labels: np.ndarray
    truth_episodes: pd.DataFrame


def make_synthetic_record(seed, record_id=None):
    """Generate one patient-like beat sequence with labeled rhythm episodes."""
    rng = np.random.default_rng(seed)
    record_id = record_id or f"synthetic_{seed:03d}"

    base_rr = rng.uniform(0.72, 0.98)
    normal_jitter = rng.uniform(0.008, 0.025)
    beat_len = 96
    t = np.linspace(-1.0, 1.0, beat_len)
    qrs = np.exp(-(t * rng.uniform(7.0, 9.0)) ** 2)
    p_wave = 0.08 * np.exp(-((t + 0.32) * 9.0) ** 2)
    twave = 0.20 * np.exp(-((t - 0.38) * 4.0) ** 2)
    prototype = qrs + p_wave + twave

    rr_parts = []
    labels = []
    morphology_offsets = []
    sqi_parts = []
    uncertainty_parts = []

    def add_segment(label, n, rr_values, sqi=0.92, uncertainty=0.03, morph_offset=0.0):
        rr_values = np.asarray(rr_values, dtype=float)
        rr_parts.extend(rr_values.tolist())
        labels.extend([label] * n)
        morphology_offsets.extend([morph_offset] * n)
        sqi_parts.extend(np.full(n, sqi, dtype=float).tolist())
        uncertainty_parts.extend(np.full(n, uncertainty, dtype=float).tolist())

    add_segment(
        "normal",
        90,
        base_rr + rng.normal(0.0, normal_jitter, 90),
        sqi=rng.uniform(0.88, 0.98),
        uncertainty=rng.uniform(0.0, 0.05),
    )

    event_order = ["sinus_tachy", "af_like", "ectopic_like", "noise_uncertain"]
    rng.shuffle(event_order)
    for event in event_order:
        gap_n = int(rng.integers(8, 18))
        add_segment("normal", gap_n, base_rr + rng.normal(0.0, normal_jitter, gap_n))

        if event == "sinus_tachy":
            n = int(rng.integers(18, 38))
            rr = rng.uniform(0.39, 0.52) + rng.normal(0.0, 0.010, n)
            add_segment(event, n, rr, sqi=0.92, uncertainty=0.03)
        elif event == "af_like":
            n = int(rng.integers(28, 58))
            rr = base_rr + rng.normal(0.0, rng.uniform(0.11, 0.19), n)
            rr = np.clip(rr, 0.42, 1.35)
            add_segment(event, n, rr, sqi=0.90, uncertainty=0.04)
        elif event == "ectopic_like":
            n = int(rng.integers(15, 33))
            rr = base_rr + rng.normal(0.0, normal_jitter, n)
            rr[::3] = np.clip(base_rr * rng.uniform(0.48, 0.65), 0.36, 0.70)
            rr[1::3] = np.clip(base_rr * rng.uniform(1.25, 1.55), 0.85, 1.55)
            add_segment(event, n, rr, sqi=0.91, uncertainty=0.04, morph_offset=rng.uniform(0.28, 0.50))
        elif event == "noise_uncertain":
            n = int(rng.integers(10, 24))
            rr = base_rr + rng.normal(0.0, normal_jitter * 1.5, n)
            add_segment(event, n, rr, sqi=rng.uniform(0.12, 0.32), uncertainty=rng.uniform(0.45, 0.75))

    add_segment("normal", 50, base_rr + rng.normal(0.0, normal_jitter, 50))

    rr = np.asarray(rr_parts, dtype=float)
    rr = np.clip(rr, 0.30, 1.80)
    labels = np.asarray(labels, dtype=object)
    sqi = np.asarray(sqi_parts, dtype=float)
    uncertainty = np.asarray(uncertainty_parts, dtype=float)
    morphology_offsets = np.asarray(morphology_offsets, dtype=float)

    r_times = np.cumsum(rr)
    rr_prev = rr.copy()
    rr_prev[0] = np.nan
    rr_next = np.r_[rr[1:], np.nan]

    beats = np.tile(prototype, (len(rr), 1))
    beats += rng.normal(0.0, 0.012, beats.shape)
    ect_shape = np.exp(-((t - rng.uniform(0.12, 0.23)) * 8.0) ** 2)
    beats += morphology_offsets[:, None] * ect_shape[None, :]

    truth = labels_to_episodes(r_times, labels)
    return SyntheticRecord(
        record_id=record_id,
        r_times=r_times,
        rr_prev=rr_prev,
        rr_next=rr_next,
        beats=beats,
        sqi_at_r=sqi,
        rpeak_uncertainty=uncertainty,
        beat_labels=labels,
        truth_episodes=truth,
    )


def labels_to_episodes(r_times, labels, ignored_label="normal"):
    r_times = np.asarray(r_times, dtype=float)
    labels = np.asarray(labels, dtype=object)
    episodes = []
    start = None
    current = None
    for i, label in enumerate(labels):
        if label != ignored_label and start is None:
            start = i
            current = label
        elif start is not None and label != current:
            episodes.append(
                {"start_s": float(r_times[start]), "end_s": float(r_times[i - 1]), "type": current}
            )
            start = i if label != ignored_label else None
            current = label if label != ignored_label else None
    if start is not None:
        episodes.append({"start_s": float(r_times[start]), "end_s": float(r_times[-1]), "type": current})
    return pd.DataFrame(episodes)


def normalize_episode_types(episodes):
    if episodes is None or len(episodes) == 0:
        return pd.DataFrame(columns=["start_s", "end_s", "type"])
    out = episodes.copy()
    out["type"] = out["type"].map(lambda x: TYPE_ALIASES.get(x, x))
    return out


def episode_iou(a, b):
    start = max(float(a["start_s"]), float(b["start_s"]))
    end = min(float(a["end_s"]), float(b["end_s"]))
    inter = max(0.0, end - start)
    union = max(float(a["end_s"]), float(b["end_s"])) - min(float(a["start_s"]), float(b["start_s"]))
    if union <= 0:
        return 0.0
    return inter / union


def evaluate_episodes(predicted, truth, duration_s, types=CANONICAL_TYPES, iou_threshold=0.30):
    predicted = normalize_episode_types(predicted)
    truth = normalize_episode_types(truth)

    rows = []
    matched_pred_global = set()
    matched_truth_global = set()
    for typ in types:
        pred_idx = predicted.index[predicted["type"] == typ].tolist()
        truth_idx = truth.index[truth["type"] == typ].tolist()
        pairs = []
        for pi in pred_idx:
            for ti in truth_idx:
                iou = episode_iou(predicted.loc[pi], truth.loc[ti])
                if iou >= iou_threshold:
                    pairs.append((iou, pi, ti))
        pairs.sort(reverse=True)

        matched_pred = set()
        matched_truth = set()
        ious = []
        for iou, pi, ti in pairs:
            if pi in matched_pred or ti in matched_truth:
                continue
            matched_pred.add(pi)
            matched_truth.add(ti)
            matched_pred_global.add(pi)
            matched_truth_global.add(ti)
            ious.append(iou)

        tp = len(matched_pred)
        fp = len(pred_idx) - tp
        fn = len(truth_idx) - tp
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "type": typ,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_iou": float(np.mean(ious)) if ious else np.nan,
            }
        )

    total_tp = len(matched_pred_global)
    total_fp = len(predicted) - total_tp
    total_fn = len(truth) - total_tp
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hours = max(float(duration_s) / 3600.0, 1e-12)
    rows.append(
        {
            "type": "macro",
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": np.nan,
        }
    )
    rows.append(
        {
            "type": "false_alarms_per_hour",
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": np.nan,
            "recall": np.nan,
            "f1": total_fp / hours,
            "mean_iou": np.nan,
        }
    )
    return pd.DataFrame(rows)


def pasm_predict(record, thresholds=None):
    return run_pasm_rhythm(
        record.r_times,
        record.rr_prev,
        rr_next=record.rr_next,
        beats=record.beats,
        sqi_at_r=record.sqi_at_r,
        rpeak_uncertainty=record.rpeak_uncertainty,
        win_beats=10,
        memory_warmup_beats=80,
        min_confidence_by_state=thresholds,
    )


def make_synthetic_cohort(n_records, seed, prefix="synthetic"):
    return [make_synthetic_record(seed + i, record_id=f"{prefix}_{i:03d}") for i in range(n_records)]


def threshold_grid():
    base = dict(DEFAULT_MIN_CONFIDENCE_BY_STATE)
    for tachy in [0.24, 0.28, 0.30, 0.34]:
        for af_like in [0.36, 0.40, 0.42, 0.46, 0.50]:
            for ectopic in [0.28, 0.30, 0.34]:
                cfg = dict(base)
                cfg["sinus_tachy"] = tachy
                cfg["sinus_brady"] = tachy
                cfg["af_like"] = af_like
                cfg["ectopic_like"] = ectopic
                yield cfg


def evaluate_pasm_on_records(records, thresholds=None, model_name="pasm"):
    rows = []
    for record in records:
        pasm = pasm_predict(record, thresholds=thresholds)
        metrics = evaluate_episodes(
            pasm["episodes"],
            record.truth_episodes,
            duration_s=float(record.r_times[-1] - record.r_times[0]),
        )
        metrics.insert(0, "model", model_name)
        metrics.insert(0, "record_id", record.record_id)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def tune_pasm_thresholds(records):
    best_cfg = None
    best_score = -np.inf
    best_summary = None
    for cfg in threshold_grid():
        metrics = evaluate_pasm_on_records(records, thresholds=cfg, model_name="pasm_tuned")
        summary = summarize_benchmark(metrics)
        row = summary.iloc[0]
        f1 = float(row["episode_f1_mean"])
        faph = float(row["false_alarms_per_hour_mean"])
        score = f1 - 0.001 * faph
        if score > best_score:
            best_score = score
            best_cfg = dict(cfg)
            best_summary = summary
    return best_cfg, best_summary


def run_synthetic_benchmark(n_records=30, seed=2026, thresholds=None):
    records = make_synthetic_cohort(n_records, seed)
    return evaluate_pasm_on_records(records, thresholds=thresholds, model_name="pasm"), records


def run_train_holdout_benchmark(train_records=30, holdout_records=30, seed=2026):
    train = make_synthetic_cohort(train_records, seed, prefix="train")
    holdout = make_synthetic_cohort(holdout_records, seed + 10000, prefix="holdout")
    thresholds, train_summary = tune_pasm_thresholds(train)
    holdout_metrics = pd.concat(
        [
            evaluate_pasm_on_records(holdout, thresholds=thresholds, model_name="pasm_tuned"),
            evaluate_pasm_on_records(holdout, thresholds=None, model_name="pasm_default"),
        ],
        ignore_index=True,
    )
    return {
        "thresholds": thresholds,
        "train_summary": train_summary,
        "holdout_metrics": holdout_metrics,
        "holdout_summary": summarize_benchmark(holdout_metrics),
        "train_records": train,
        "holdout_records": holdout,
    }


def summarize_benchmark(metrics):
    summary_rows = []
    for model, group in metrics.groupby("model"):
        typed = group[group["type"].isin(CANONICAL_TYPES)]
        macro = group[group["type"] == "macro"]
        faph = group[group["type"] == "false_alarms_per_hour"]
        summary_rows.append(
            {
                "model": model,
                "episode_f1_mean": float(macro["f1"].mean()),
                "episode_precision_mean": float(macro["precision"].mean()),
                "episode_recall_mean": float(macro["recall"].mean()),
                "false_alarms_per_hour_mean": float(faph["f1"].mean()),
                "typed_f1_mean": float(typed["f1"].mean()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("episode_f1_mean", ascending=False)


def write_markdown_report(path, metrics, summary, n_records, thresholds=None, train_summary=None):
    path = Path(path)
    type_table = (
        metrics[metrics["type"].isin(CANONICAL_TYPES)]
        .groupby(["model", "type"])[["precision", "recall", "f1", "mean_iou"]]
        .mean()
        .reset_index()
    )
    lines = [
        "# PASM-Rhythm Synthetic Validation",
        "",
        "This report is generated from deterministic synthetic ECG rhythm cohorts.",
        "It validates the PASM-Rhythm code path and episode metrics, but it is not clinical validation.",
        "",
        f"Records: {n_records}",
        "",
        "## Decoder Thresholds",
        "",
        _thresholds_to_markdown(thresholds),
        "",
        "## Training Summary",
        "",
        _df_to_markdown(train_summary) if train_summary is not None else "_No threshold tuning run._",
        "",
        "## Summary",
        "",
        _df_to_markdown(summary),
        "",
        "## Per-Type Mean Metrics",
        "",
        _df_to_markdown(type_table),
        "",
        "## Interpretation",
        "",
        "- This report validates the PASM-only rhythm path on synthetic ectopic, AF-like, tachy, brady, and noise episodes.",
        "- Results are useful as a regression gate before moving to PhysioNet/MIT-BIH style validation.",
        "- The next validation stage must use patient-wise splits on real annotated ECG databases.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _df_to_markdown(df):
    if len(df) == 0:
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


def _thresholds_to_markdown(thresholds):
    if thresholds is None:
        thresholds = DEFAULT_MIN_CONFIDENCE_BY_STATE
    rows = [{"state": k, "min_confidence": v} for k, v in sorted(thresholds.items())]
    return _df_to_markdown(pd.DataFrame(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run PASM-Rhythm synthetic validation.")
    parser.add_argument("--records", type=int, default=30)
    parser.add_argument("--train-records", type=int, default=30)
    parser.add_argument("--holdout-records", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="PASM_VALIDATION.md")
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args(argv)

    if args.no_tune:
        metrics, _ = run_synthetic_benchmark(n_records=args.records, seed=args.seed)
        summary = summarize_benchmark(metrics)
        write_markdown_report(args.out, metrics, summary, args.records)
    else:
        result = run_train_holdout_benchmark(
            train_records=args.train_records,
            holdout_records=args.holdout_records,
            seed=args.seed,
        )
        metrics = result["holdout_metrics"]
        summary = result["holdout_summary"]
        write_markdown_report(
            args.out,
            metrics,
            summary,
            args.holdout_records,
            thresholds=result["thresholds"],
            train_summary=result["train_summary"],
        )
    print(summary.to_string(index=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
