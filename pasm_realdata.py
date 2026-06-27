import argparse
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

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
)
from pasm_validation import summarize_benchmark


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
    summary = summarize_benchmark(metrics)
    return {
        "preset": preset,
        "records": records,
        "informative_records": informative,
        "inventory": summarize_loaded_records(records),
        "skipped": skipped,
        "metrics": metrics,
        "summary": summary,
    }


def write_realdata_report(path, result):
    path = Path(path)
    metrics = result["metrics"]
    summary = result["summary"]
    skipped = result["skipped"]
    inventory = result["inventory"]
    type_table, macro_table = build_realdata_tables(metrics)

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
        "## Skipped Or Empty Records",
        "",
        _df_to_markdown(skipped) if skipped is not None and len(skipped) else "_None._",
        "",
        "## Current Interpretation",
        "",
        "- The preset exercises both AFDB rhythm annotations and MITDB short ectopic runs.",
        "- AFDB false alarms are much lower after PhysioNet evidence postprocessing; validate these parameters on a broader patient-wise split.",
        "- MITDB ectopy has stricter patient-relative short-RR evidence, but `mitdb/203` remains the main false-alarm stress case.",
        "- This is still a research checkpoint; the next step is patient-wise train/holdout validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
