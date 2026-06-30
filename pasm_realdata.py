import argparse
import json
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from pasm_ai_reranker import build_episode_candidates, label_episode_candidates
from pasm_html_report import build_realdata_tables, write_diagnostic_html_report, write_realdata_html_report
from pasm_physionet import (
    ECTOPY_FLOOD_DENSE_MORPH_Z,
    ECTOPY_FLOOD_DENSITY_WINDOW_S,
    ECTOPY_FLOOD_MIN_CONFIDENCE,
    ECTOPY_FLOOD_MIN_CANDIDATES,
    ECTOPY_FLOOD_MIN_DENSITY,
    ECTOPY_FLOOD_RATE_PER_HOUR,
    ECTOPY_FLOOD_STRONG_MORPH_Z,
    ECTOPY_MERGE_GAP_S,
    ECTOPY_RELATIVE_RR_FRACTION,
    ECTOPY_SHORT_RR_S,
    PHYSIONET_AF_MERGE_GAP_S,
    PHYSIONET_AF_TACHY_MARGIN_S,
    PHYSIONET_MIN_TACHY_DURATION_S,
    diagnose_physionet_record,
    evaluate_physionet_records,
    load_afdb_record,
    load_mitdb_record,
    match_diagnostic_episodes,
    run_pasm_physionet_pipeline,
)
from pasm_validation import episode_iou, normalize_episode_types, summarize_benchmark


PRESETS = {
    "smoke": {
        "mitdb": {"records": ["200"], "max_seconds": 900.0},
        "afdb": {"records": ["04015", "04126"], "max_seconds": 900.0},
    },
    "mini": {
        "mitdb": {"records": ["200", "201", "203", "205", "208"], "max_seconds": 900.0},
        "afdb": {"records": ["04015", "04043", "04048", "04126", "04746"], "max_seconds": 1200.0},
    },
    "afdb-smoke": {
        "afdb": {"records": ["04015", "04126"], "max_seconds": 900.0},
    },
    "afdb-mini": {
        "afdb": {"records": ["04015", "04043", "04048", "04126", "04746"], "max_seconds": 1200.0},
    },
    "mitdb-smoke": {
        "mitdb": {"records": ["200"], "max_seconds": 900.0},
    },
    "mitdb-mini": {
        "mitdb": {"records": ["200", "201", "203", "205", "208"], "max_seconds": 900.0},
    },
}


def list_presets():
    rows = []
    for preset, spec in sorted(PRESETS.items()):
        for db, cfg in spec.items():
            rows.append(
                {
                    "preset": preset,
                    "db": db,
                    "records": " ".join(cfg["records"]),
                    "max_seconds": float(cfg.get("max_seconds", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def load_records_for_preset(preset):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Available: {', '.join(sorted(PRESETS))}")

    loaded = []
    skipped = []
    spec = PRESETS[preset]
    for db, cfg in spec.items():
        loader = load_mitdb_record if db == "mitdb" else load_afdb_record
        for record_name in cfg["records"]:
            try:
                record = loader(record_name, max_seconds=cfg.get("max_seconds"))
                if len(record.truth_episodes) == 0:
                    skipped.append({"record_id": record.record_id, "reason": "empty_truth"})
                loaded.append(record)
            except Exception as exc:
                skipped.append({"record_id": f"{db}/{record_name}", "reason": f"{type(exc).__name__}: {exc}"})
    return loaded, pd.DataFrame(skipped)


def summarize_loaded_records(records):
    rows = []
    for record in records:
        truth = record.truth_episodes
        truth_duration_s = 0.0
        truth_types = ""
        if truth is not None and len(truth) > 0:
            truth_duration_s = float((truth["end_s"] - truth["start_s"]).clip(lower=0).sum())
            truth_types = " ".join(sorted(str(t) for t in truth["type"].dropna().unique()))
        rows.append(
            {
                "record_id": record.record_id,
                "duration_s": float(len(record.signal)) / float(record.fs),
                "beats": int(len(record.rpeaks)),
                "truth_episodes": int(len(truth)),
                "truth_duration_s": truth_duration_s,
                "truth_types": truth_types,
            }
        )
    return pd.DataFrame(rows)


def run_realdata_preset(preset="smoke"):
    records, skipped = load_records_for_preset(preset)
    informative = [record for record in records if len(record.truth_episodes) > 0]
    if not informative:
        raise ValueError(f"Preset {preset!r} has no informative records with ground-truth episodes.")

    metrics = evaluate_physionet_records(informative, skip_empty_truth=True)
    pipelines = {record.record_id: run_pasm_physionet_pipeline(record) for record in informative}
    candidate_metrics, fn_audit = audit_candidate_stage(informative, pipelines)
    summary = summarize_benchmark(metrics)
    return {
        "preset": preset,
        "records": records,
        "informative_records": informative,
        "pipelines": pipelines,
        "inventory": summarize_loaded_records(records),
        "skipped": skipped,
        "metrics": metrics,
        "candidate_metrics": candidate_metrics,
        "fn_audit": fn_audit,
        "summary": summary,
    }


def write_realdata_report(path, result):
    path = Path(path)
    metrics = result["metrics"]
    summary = result["summary"]
    skipped = result["skipped"]
    inventory = result["inventory"]
    type_table, macro_table = build_realdata_tables(metrics)
    candidate_metrics = result.get("candidate_metrics", pd.DataFrame())
    fn_audit = result.get("fn_audit", pd.DataFrame())

    lines = [
        "# PASM-Rhythm Real-Data Summary",
        "",
        "This report combines the configured WFDB/PhysioNet real-data preset.",
        "It is a research validation checkpoint, not clinical certification.",
        "",
        f"Preset: `{result['preset']}`",
        f"Records loaded: {len(result['records'])}",
        f"Records with truth episodes: {len(result['informative_records'])}",
        "",
        "## Loaded Record Inventory",
        "",
        _df_to_markdown(inventory),
        "",
        "## Summary",
        "",
        _df_to_markdown(summary),
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
        f"- Ectopy flood strong morph z: {ECTOPY_FLOOD_STRONG_MORPH_Z:.2f}",
        f"- Ectopy flood dense morph z: {ECTOPY_FLOOD_DENSE_MORPH_Z:.2f}",
        "",
        "## Per-Record Type Metrics",
        "",
        _df_to_markdown(type_table),
        "",
        "## Per-Record Episode Metrics",
        "",
        _df_to_markdown(macro_table),
        "",
        "## Candidate-Level Metrics",
        "",
        "These rows evaluate the physiology-guided candidate generator before final deterministic filtering or AI/reranker acceptance.",
        "",
        _df_to_markdown(candidate_metrics),
        "",
        "## False-Negative Stage Audit",
        "",
        "Each FN is assigned to the earliest visible stage that can explain the miss.",
        "",
        _df_to_markdown(fn_audit),
        "",
        "## Skipped Or Empty Records",
        "",
        _df_to_markdown(skipped) if skipped is not None and len(skipped) else "_None._",
        "",
        "## Diagnostic Sidecars",
        "",
        "When generated through the CLI, per-record TP/FP/FN diagnostics are written under `reports/diagnostics/`.",
        "",
        "## Current Interpretation",
        "",
        "- The preset exercises both AFDB rhythm annotations and MITDB short ectopic runs.",
        "- AFDB false alarms are much lower after PhysioNet evidence postprocessing; validate these parameters on a broader patient-wise split.",
        "- The pipeline is reported as four separate stages: physiology-guided candidate generator, deterministic PASM evidence/fallback, PASM-AI reranker/acceptor, and final decision fusion.",
        "- MITDB candidate coverage should be interpreted before final F1; the reranker cannot recover truth episodes that were never proposed.",
        "- MITDB ectopy has stricter patient-relative short-RR evidence, but `mitdb/203` remains the main false-alarm stress case.",
        "- This is still a research checkpoint; the next step is patient-wise train/holdout validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def audit_candidate_stage(records, pipelines=None, iou_threshold=0.30):
    pipelines = pipelines or {record.record_id: run_pasm_physionet_pipeline(record) for record in records}
    metric_rows = []
    fn_rows = []
    for record in records:
        pipeline = pipelines[record.record_id]
        candidates = build_episode_candidates(record, pipeline)
        labeled, uncovered = label_episode_candidates(
            candidates,
            record.truth_episodes,
            record_id=record.record_id,
            split="candidate",
            iou_threshold=iou_threshold,
        )
        truth_count = int(len(normalize_episode_types(record.truth_episodes)))
        proposed_true = max(0, truth_count - int(len(uncovered)))
        accepted_candidates = int(labeled["accepted"].sum()) if len(labeled) and "accepted" in labeled else 0
        candidate_count = int(len(labeled))
        metric_rows.append(
            {
                "record_id": record.record_id,
                "candidate_count": candidate_count,
                "candidate_tp_rows": accepted_candidates,
                "candidate_fp_rows": max(0, candidate_count - accepted_candidates),
                "truth_episodes": truth_count,
                "candidate_recalled_truth": proposed_true,
                "truth_never_proposed": int(len(uncovered)),
                "candidate_precision": accepted_candidates / candidate_count if candidate_count else 1.0,
                "candidate_recall": proposed_true / truth_count if truth_count else 1.0,
            }
        )

        predicted = normalize_episode_types(pipeline.get("episodes"))
        truth = normalize_episode_types(record.truth_episodes)
        diagnostics = match_diagnostic_episodes(predicted, truth, pipeline=pipeline, iou_threshold=iou_threshold)
        for _, row in diagnostics[diagnostics["status"] == "FN"].iterrows():
            fn_rows.append(_fn_audit_row(record, row, candidates, predicted, iou_threshold))

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(fn_rows, columns=_fn_audit_columns()),
    )


def _fn_audit_row(record, fn_row, candidates, final_predictions, iou_threshold):
    truth_like = {
        "start_s": float(fn_row.get("start_s", np.nan)),
        "end_s": float(fn_row.get("end_s", np.nan)),
        "type": fn_row.get("type", ""),
    }
    best_candidate, best_candidate_iou = _best_overlap(candidates, truth_like)
    best_final, best_final_iou = _best_overlap(final_predictions, truth_like)
    generated = best_candidate is not None and best_candidate_iou >= float(iou_threshold)

    if generated:
        source = str(best_candidate.get("source", ""))
        ai_decision = str(best_candidate.get("ai_decision", ""))
        if ai_decision == "reject":
            rejected_stage = "ai_reranker_rejected"
        elif source in {"relaxed_ectopy", "state_score"}:
            rejected_stage = "deterministic_rules_rejected"
        else:
            rejected_stage = "label_matching_or_episode_boundary"
    elif best_candidate is not None and best_candidate_iou > 0.0:
        rejected_stage = "label_matching_or_episode_boundary"
    else:
        rejected_stage = "candidate_generator_miss"

    context = best_candidate if best_candidate is not None else fn_row
    return {
        "record_id": record.record_id,
        "episode_time": float(truth_like["start_s"]),
        "start_s": float(truth_like["start_s"]),
        "end_s": float(truth_like["end_s"]),
        "label_type": truth_like["type"],
        "generated_candidate_yes_no": "yes" if generated else "no",
        "rejected_stage": rejected_stage,
        "best_candidate_iou": float(best_candidate_iou),
        "best_final_iou": float(best_final_iou),
        "source": context.get("source", ""),
        "rr_pattern": context.get("pattern", ""),
        "rr_support": _first_finite(context, "rr_support", "context_rr_prev"),
        "morphology_score": _first_finite(context, "mean_morph_z"),
        "sqi": _first_finite(context, "mean_sqi"),
        "density": _first_finite(context, "density_support", "candidate_density"),
        "pause_support": _first_finite(context, "pause_support"),
        "ectopy_pattern": context.get("pattern", ""),
        "final_decision": "FN",
    }


def _best_overlap(episodes, truth_like):
    if episodes is None or len(episodes) == 0:
        return None, 0.0
    best = None
    best_iou = 0.0
    normalized = normalize_episode_types(episodes)
    for _, episode in normalized.iterrows():
        if episode.get("type") != truth_like.get("type"):
            continue
        iou = episode_iou(episode, truth_like)
        if iou > best_iou:
            best = episode
            best_iou = float(iou)
    return best, best_iou


def _first_finite(row, *keys):
    for key in keys:
        if key not in row:
            continue
        try:
            value = float(row.get(key, np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return np.nan


def _fn_audit_columns():
    return [
        "record_id",
        "episode_time",
        "start_s",
        "end_s",
        "label_type",
        "generated_candidate_yes_no",
        "rejected_stage",
        "best_candidate_iou",
        "best_final_iou",
        "source",
        "rr_pattern",
        "rr_support",
        "morphology_score",
        "sqi",
        "density",
        "pause_support",
        "ectopy_pattern",
        "final_decision",
    ]


def write_preset_diagnostic_tables(result, out_dir=None):
    out_dir = Path(out_dir or Path("reports") / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for record in result.get("informative_records", []):
        diagnostic = diagnose_physionet_record(record)
        rows = diagnostic["diagnostics"]
        safe_record = record.record_id.replace("/", "_").replace("\\", "_")
        csv_path = out_dir / f"{safe_record}_diagnostics.csv"
        md_path = out_dir / f"{safe_record}_diagnostics.md"
        rows.to_csv(csv_path, index=False)

        counts = rows["status"].value_counts().to_dict() if len(rows) else {}
        pattern_counts = (
            rows[rows["status"].isin(["TP", "FP"])]
            .get("pattern", pd.Series(dtype=object))
            .replace("", "unknown")
            .value_counts()
            .to_dict()
            if len(rows)
            else {}
        )
        lines = [
            f"# Diagnostics: {record.record_id}",
            "",
            "## Status Counts",
            "",
            _df_to_markdown(pd.DataFrame([{"status": k, "count": v} for k, v in sorted(counts.items())])),
            "",
            "## Pattern Counts",
            "",
            _df_to_markdown(pd.DataFrame([{"pattern": k, "count": v} for k, v in sorted(pattern_counts.items())])),
            "",
            "## Episodes",
            "",
            _df_to_markdown(rows),
            "",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")
        for status, count in counts.items():
            summary_rows.append({"record_id": record.record_id, "kind": "status", "name": status, "count": int(count)})
        for pattern, count in pattern_counts.items():
            summary_rows.append({"record_id": record.record_id, "kind": "pattern", "name": pattern, "count": int(count)})

    summary = pd.DataFrame(summary_rows, columns=["record_id", "kind", "name", "count"])
    summary_path = out_dir / f"{result.get('preset', 'preset')}_diagnostics_summary.md"
    summary_path.write_text(
        "\n".join(["# Diagnostic Summary", "", _df_to_markdown(summary), ""]),
        encoding="utf-8",
    )
    return summary


def write_candidate_audit_tables(result, out_dir=None):
    out_dir = Path(out_dir or Path("reports") / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = result.get("preset", "preset")
    safe_preset = str(preset).replace("/", "_").replace("\\", "_")
    candidate_metrics = result.get("candidate_metrics", pd.DataFrame())
    fn_audit = result.get("fn_audit", pd.DataFrame())

    candidate_csv = out_dir / f"{safe_preset}_candidate_metrics.csv"
    candidate_md = out_dir / f"{safe_preset}_candidate_metrics.md"
    fn_csv = out_dir / f"{safe_preset}_fn_audit.csv"
    fn_json = out_dir / f"{safe_preset}_fn_audit.json"
    fn_md = out_dir / f"{safe_preset}_fn_audit.md"

    candidate_metrics.to_csv(candidate_csv, index=False)
    candidate_md.write_text(
        "\n".join(["# Candidate-Level Metrics", "", _df_to_markdown(candidate_metrics), ""]),
        encoding="utf-8",
    )
    fn_audit.to_csv(fn_csv, index=False)
    fn_json.write_text(json.dumps(fn_audit.to_dict("records"), indent=2), encoding="utf-8")
    fn_md.write_text(
        "\n".join(["# False-Negative Stage Audit", "", _df_to_markdown(fn_audit), ""]),
        encoding="utf-8",
    )
    return {
        "candidate_csv": candidate_csv,
        "candidate_md": candidate_md,
        "fn_csv": fn_csv,
        "fn_json": fn_json,
        "fn_md": fn_md,
    }


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


def serve_report(path, host="127.0.0.1", port=8765, open_browser=False):
    path = Path(path).resolve()
    directory = Path.cwd().resolve()
    try:
        url_path = path.relative_to(directory).as_posix()
    except ValueError:
        directory = path.parent
        url_path = path.name
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
        *args,
        directory=str(directory),
        **kwargs,
    )
    server = ThreadingHTTPServer((host, int(port)), handler)
    url = f"http://{host}:{int(port)}/{url_path}"
    print(f"Serving {path} at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


def write_record_diagnostic_report(path, result, record_id):
    matches = [record for record in result["records"] if record.record_id == record_id]
    if not matches:
        available = ", ".join(record.record_id for record in result["records"])
        raise ValueError(f"Record {record_id!r} not found in preset {result['preset']!r}. Available: {available}")
    diagnostic = diagnose_physionet_record(matches[0])
    write_diagnostic_html_report(path, diagnostic)
    return diagnostic


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run combined PASM-Rhythm real-data smoke presets.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--out", default="PASM_REALDATA_SUMMARY.md")
    parser.add_argument("--html-out", default=None)
    parser.add_argument("--serve", action="store_true", help="Serve the generated HTML report on localhost.")
    parser.add_argument("--diagnose-record", default=None)
    parser.add_argument("--diagnostic-html-out", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args(argv)

    if args.list_presets:
        print(_df_to_markdown(list_presets()))
        return

    result = run_realdata_preset(args.preset)
    write_realdata_report(args.out, result)
    diagnostics_summary = write_preset_diagnostic_tables(result)
    audit_paths = write_candidate_audit_tables(result)
    html_out = args.html_out
    if args.serve and html_out is None:
        html_out = str(Path("reports") / f"pasm_{args.preset.replace('-', '_')}.html")
    if html_out is not None:
        write_realdata_html_report(html_out, result, markdown_report=args.out)
    diagnostic_html_out = args.diagnostic_html_out
    if args.diagnose_record and diagnostic_html_out is None:
        safe_record = args.diagnose_record.replace("/", "_").replace("\\", "_")
        diagnostic_html_out = str(Path("reports") / f"{safe_record}_diagnostic.html")
    if args.diagnose_record:
        diagnostic = write_record_diagnostic_report(diagnostic_html_out, result, args.diagnose_record)
    print(result["summary"].to_string(index=False))
    print(f"Wrote {args.out}")
    print(f"Wrote diagnostics for {diagnostics_summary['record_id'].nunique() if len(diagnostics_summary) else 0} records")
    print(f"Wrote candidate metrics {audit_paths['candidate_csv']}")
    print(f"Wrote FN audit {audit_paths['fn_csv']}")
    if html_out is not None:
        print(f"Wrote {html_out}")
    if args.diagnose_record:
        counts = diagnostic["diagnostics"]["status"].value_counts().to_dict()
        print(f"Wrote {diagnostic_html_out}")
        print(f"Diagnostic {args.diagnose_record}: {counts}")
    if args.serve:
        serve_report(html_out, host=args.host, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
