from datetime import datetime
from html import escape
import os
from pathlib import Path

import numpy as np
import pandas as pd

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
)
from pasm_validation import CANONICAL_TYPES


def write_realdata_html_report(path, result, markdown_report=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    summary = result["summary"]
    inventory = result["inventory"]
    skipped = result["skipped"]
    type_table, macro_table = build_realdata_tables(metrics)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(str(result['preset']))} PASM-Rhythm Report</title>",
            "<style>",
            _stylesheet(),
            "</style>",
            "</head>",
            "<body>",
            '<main class="shell">',
            _header(result, generated_at, path, markdown_report),
            _summary_cards(summary),
            _section("Evidence Layer", _evidence_parameters()),
            _section("Loaded Record Inventory", _inventory_table(inventory)),
            _section("Per-Record Episode Metrics", _dataframe_to_table(macro_table)),
            _section("Per-Record Type Metrics", _dataframe_to_table(type_table)),
            _section(
                "Skipped Or Empty Records",
                _dataframe_to_table(skipped) if skipped is not None and len(skipped) else '<p class="empty">None.</p>',
            ),
            _section("Current Interpretation", _interpretation()),
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    path.write_text(html, encoding="utf-8")


def write_diagnostic_html_report(path, diagnostic_result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record = diagnostic_result["record"]
    diagnostics = diagnostic_result["diagnostics"]
    predicted = diagnostic_result["predicted"]
    truth = diagnostic_result["truth"]
    duration_s = float(len(record.signal)) / float(record.fs)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_counts = diagnostics["status"].value_counts().to_dict() if len(diagnostics) else {}

    html = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(record.record_id)} Diagnostic Report</title>",
            "<style>",
            _stylesheet(),
            "</style>",
            "</head>",
            "<body>",
            '<main class="shell">',
            f"""
<header class="topbar">
  <div>
    <p class="label">PASM-Rhythm Diagnostics</p>
    <h1>Record Diagnostic: {escape(record.record_id)}</h1>
    <p class="lede">Prediction vs truth episode matching with RR, morphology, and local context.</p>
  </div>
  <div class="meta">
    <span>{escape(generated_at)}</span>
    <span>{_fmt(duration_s)} s</span>
    <span>IoU >= {_fmt(diagnostic_result["iou_threshold"])}</span>
  </div>
</header>
""",
            _diagnostic_cards(status_counts, predicted, truth),
            _section("Timeline", _timeline(predicted, truth, duration_s)),
            _section("Ectopy Evidence Parameters", _ectopy_parameters()),
            _section("Diagnostic Episodes", _dataframe_to_table(diagnostics)),
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    path.write_text(html, encoding="utf-8")


def build_realdata_tables(metrics):
    type_table = (
        metrics[metrics["type"].isin(CANONICAL_TYPES)]
        .groupby(["record_id", "type"])[["precision", "recall", "f1", "mean_iou"]]
        .mean()
        .reset_index()
    )
    macro_table = (
        metrics[metrics["type"].isin(["macro", "false_alarms_per_hour"])]
        .pivot_table(
            index="record_id",
            columns="type",
            values=["tp", "fp", "fn", "precision", "recall", "f1"],
            aggfunc="first",
        )
        .reset_index()
    )
    macro_table.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in macro_table.columns
    ]
    keep_cols = [
        "record_id",
        "tp_macro",
        "fp_macro",
        "fn_macro",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_false_alarms_per_hour",
    ]
    return type_table, macro_table[[col for col in keep_cols if col in macro_table.columns]]


def _header(result, generated_at, html_path, markdown_report):
    md_link = ""
    if markdown_report:
        md_path = Path(markdown_report)
        href = os.path.relpath(md_path.resolve(), html_path.parent.resolve()).replace("\\", "/")
        md_link = f'<a class="link" href="{escape(href)}">Markdown</a>'

    return f"""
<header class="topbar">
  <div>
    <p class="label">PASM-Rhythm Validation</p>
    <h1>Real-Data Report: {escape(str(result["preset"]))}</h1>
    <p class="lede">WFDB/PhysioNet checkpoint report for arrhythmia episode detection.</p>
  </div>
  <div class="meta">
    <span>{escape(generated_at)}</span>
    <span>{len(result["records"])} records</span>
    <span>{len(result["informative_records"])} with truth</span>
    {md_link}
  </div>
</header>
"""


def _summary_cards(summary):
    if summary is None or len(summary) == 0:
        return _section("Summary", '<p class="empty">No summary rows.</p>')

    row = summary.iloc[0]
    cards = [
        ("Episode F1", row.get("episode_f1_mean")),
        ("Precision", row.get("episode_precision_mean")),
        ("Recall", row.get("episode_recall_mean")),
        ("False Alarms / Hour", row.get("false_alarms_per_hour_mean")),
        ("Typed F1", row.get("typed_f1_mean")),
    ]
    body = ['<section class="summary-grid" aria-label="Summary metrics">']
    for label, value in cards:
        body.append(
            f"""
<article class="metric">
  <div class="metric-label">{escape(label)}</div>
  <div class="metric-value">{_fmt(value)}</div>
</article>
"""
        )
    body.append("</section>")
    return "\n".join(body)


def _evidence_parameters():
    rows = pd.DataFrame(
        [
            {"parameter": "AF merge gap", "value": f"{PHYSIONET_AF_MERGE_GAP_S:.1f} s"},
            {"parameter": "AF-adjacent tachy suppression margin", "value": f"{PHYSIONET_AF_TACHY_MARGIN_S:.1f} s"},
            {"parameter": "Minimum retained sinus tachy duration", "value": f"{PHYSIONET_MIN_TACHY_DURATION_S:.1f} s"},
            {"parameter": "Ectopy short RR", "value": f"{ECTOPY_SHORT_RR_S:.2f} s"},
            {"parameter": "Ectopy relative RR fraction", "value": f"{ECTOPY_RELATIVE_RR_FRACTION:.2f}"},
            {"parameter": "Ectopy merge gap", "value": f"{ECTOPY_MERGE_GAP_S:.1f} s"},
            {"parameter": "Ectopy flood rate threshold", "value": f"{ECTOPY_FLOOD_RATE_PER_HOUR:.1f} / h"},
            {"parameter": "Ectopy flood minimum confidence", "value": f"{ECTOPY_FLOOD_MIN_CONFIDENCE:.2f}"},
            {"parameter": "Ectopy flood density window", "value": f"{ECTOPY_FLOOD_DENSITY_WINDOW_S:.1f} s"},
            {"parameter": "Ectopy flood minimum density", "value": str(ECTOPY_FLOOD_MIN_DENSITY)},
            {"parameter": "Ectopy flood minimum candidates", "value": str(ECTOPY_FLOOD_MIN_CANDIDATES)},
            {"parameter": "Ectopy flood strong morph z", "value": f"{ECTOPY_FLOOD_STRONG_MORPH_Z:.2f}"},
            {"parameter": "Ectopy flood dense morph z", "value": f"{ECTOPY_FLOOD_DENSE_MORPH_Z:.2f}"},
        ]
    )
    return _dataframe_to_table(rows)


def _ectopy_parameters():
    rows = pd.DataFrame(
        [
            {"parameter": "short_rr_s", "value": ECTOPY_SHORT_RR_S},
            {"parameter": "relative_rr_fraction", "value": ECTOPY_RELATIVE_RR_FRACTION},
            {"parameter": "merge_gap_s", "value": ECTOPY_MERGE_GAP_S},
            {"parameter": "flood_rate_per_hour", "value": ECTOPY_FLOOD_RATE_PER_HOUR},
            {"parameter": "flood_min_confidence", "value": ECTOPY_FLOOD_MIN_CONFIDENCE},
            {"parameter": "flood_density_window_s", "value": ECTOPY_FLOOD_DENSITY_WINDOW_S},
            {"parameter": "flood_min_density", "value": ECTOPY_FLOOD_MIN_DENSITY},
            {"parameter": "flood_min_candidates", "value": ECTOPY_FLOOD_MIN_CANDIDATES},
            {"parameter": "flood_strong_morph_z", "value": ECTOPY_FLOOD_STRONG_MORPH_Z},
            {"parameter": "flood_dense_morph_z", "value": ECTOPY_FLOOD_DENSE_MORPH_Z},
        ]
    )
    return _dataframe_to_table(rows)


def _diagnostic_cards(status_counts, predicted, truth):
    cards = [
        ("TP", status_counts.get("TP", 0)),
        ("FP", status_counts.get("FP", 0)),
        ("FN", status_counts.get("FN", 0)),
        ("Predicted", len(predicted)),
        ("Truth", len(truth)),
    ]
    body = ['<section class="summary-grid" aria-label="Diagnostic counts">']
    for label, value in cards:
        body.append(
            f"""
<article class="metric">
  <div class="metric-label">{escape(label)}</div>
  <div class="metric-value">{_fmt(value)}</div>
</article>
"""
        )
    body.append("</section>")
    return "\n".join(body)


def _timeline(predicted, truth, duration_s):
    duration_s = max(float(duration_s), 1e-9)

    def lane(name, episodes, css_class):
        pieces = [f'<div class="lane"><div class="lane-label">{escape(name)}</div><div class="lane-track">']
        for _, row in episodes.iterrows():
            start = max(0.0, min(100.0, float(row["start_s"]) / duration_s * 100.0))
            end = max(start + 0.25, min(100.0, float(row["end_s"]) / duration_s * 100.0))
            width = max(0.25, end - start)
            title = f'{row.get("type", "")} {_fmt(row.get("start_s"))}-{_fmt(row.get("end_s"))}s'
            pieces.append(
                f'<span class="tick {css_class}" title="{escape(title)}" style="left:{start:.3f}%;width:{width:.3f}%"></span>'
            )
        pieces.append("</div></div>")
        return "\n".join(pieces)

    return "\n".join(
        [
            '<div class="timeline">',
            lane("Truth", truth, "truth"),
            lane("Predicted", predicted, "pred"),
            "</div>",
        ]
    )


def _inventory_table(inventory):
    if inventory is None or len(inventory) == 0:
        return '<p class="empty">No loaded records.</p>'

    rows = []
    for _, row in inventory.iterrows():
        duration = float(row.get("duration_s", 0.0) or 0.0)
        truth_duration = float(row.get("truth_duration_s", 0.0) or 0.0)
        width = 0.0 if duration <= 0 else min(100.0, max(0.0, truth_duration / duration * 100.0))
        rows.append(
            {
                "record_id": row.get("record_id", ""),
                "duration_s": row.get("duration_s", np.nan),
                "beats": row.get("beats", ""),
                "truth_episodes": row.get("truth_episodes", ""),
                "truth_duration_s": row.get("truth_duration_s", np.nan),
                "truth_types": row.get("truth_types", ""),
                "truth_coverage": f'<span class="bar"><span style="width:{width:.2f}%"></span></span>',
            }
        )
    return _dataframe_to_table(pd.DataFrame(rows), raw_columns={"truth_coverage"})


def _interpretation():
    return """
<ul class="notes">
  <li>The preset exercises both AFDB rhythm annotations and MITDB short ectopic runs.</li>
  <li>AFDB false alarms are much lower after PhysioNet evidence postprocessing; validate these parameters on a broader patient-wise split.</li>
  <li>MITDB ectopy has stricter patient-relative short-RR evidence, but mitdb/203 remains the main false-alarm stress case.</li>
  <li>This is still a research checkpoint; the next step is patient-wise train/holdout validation.</li>
</ul>
"""


def _section(title, body):
    return f"""
<section class="panel">
  <div class="section-head">
    <h2>{escape(title)}</h2>
  </div>
  {body}
</section>
"""


def _dataframe_to_table(df, raw_columns=None):
    raw_columns = raw_columns or set()
    if df is None or len(df) == 0:
        return '<p class="empty">No rows.</p>'

    cols = list(df.columns)
    out = ['<div class="table-wrap"><table>']
    out.append("<thead><tr>")
    for col in cols:
        out.append(f"<th>{escape(str(col))}</th>")
    out.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        out.append("<tr>")
        for col in cols:
            value = row[col]
            classes = []
            if col in {"fp_macro", "fn_macro", "f1_false_alarms_per_hour"}:
                classes.append("warn")
            if col == "status":
                classes.append(f"status-{str(value).lower()}")
            if col.endswith("f1") or col.startswith("f1_") or col in {"precision_macro", "recall_macro"}:
                classes.append("score")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            if col in raw_columns:
                cell = str(value)
            else:
                cell = escape(_fmt(value))
            out.append(f"<td{class_attr}>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{float(value):.3f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if pd.isna(value):
        return ""
    return str(value)


def _stylesheet():
    return """
:root {
  color-scheme: light;
  --bg: #f5f7f8;
  --surface: #ffffff;
  --surface-2: #eef3f4;
  --text: #152023;
  --muted: #5f6d72;
  --line: #d7e0e3;
  --accent: #0f766e;
  --accent-2: #b45309;
  --good: #0b6b43;
  --shadow: 0 14px 40px rgba(21, 32, 35, 0.08);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}

.shell {
  width: min(1240px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 48px;
}

.topbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 20px;
  align-items: end;
  padding: 24px 0 20px;
}

.label {
  margin: 0 0 6px;
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: clamp(2rem, 4vw, 3.4rem);
  line-height: 1.02;
  font-weight: 850;
  letter-spacing: 0;
}

.lede {
  max-width: 680px;
  margin: 12px 0 0;
  color: var(--muted);
  font-size: 1rem;
}

.meta {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
  max-width: 480px;
}

.meta span,
.link {
  display: inline-flex;
  min-height: 34px;
  align-items: center;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  padding: 0 12px;
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 700;
  text-decoration: none;
}

.link {
  color: var(--accent);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0 18px;
}

.metric,
.panel {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
  box-shadow: var(--shadow);
}

.metric {
  min-height: 118px;
  padding: 18px;
}

.metric-label {
  min-height: 34px;
  color: var(--muted);
  font-size: 0.82rem;
  font-weight: 750;
  text-transform: uppercase;
  letter-spacing: 0;
}

.metric-value {
  margin-top: 12px;
  font-size: 2.15rem;
  line-height: 1;
  font-weight: 850;
  font-variant-numeric: tabular-nums;
}

.panel {
  margin-top: 16px;
  overflow: hidden;
}

.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--line);
  background: var(--surface-2);
  padding: 14px 18px;
}

h2 {
  margin: 0;
  font-size: 1rem;
  line-height: 1.2;
  font-weight: 850;
  letter-spacing: 0;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

th,
td {
  border-bottom: 1px solid var(--line);
  padding: 11px 12px;
  text-align: left;
  white-space: nowrap;
}

th {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 850;
  text-transform: uppercase;
  letter-spacing: 0;
}

tbody tr:hover {
  background: #f8fbfb;
}

td.score {
  color: var(--good);
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}

td.warn {
  color: var(--accent-2);
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}

td.status-tp {
  color: var(--good);
  font-weight: 850;
}

td.status-fp {
  color: var(--accent-2);
  font-weight: 850;
}

td.status-fn {
  color: #9f1239;
  font-weight: 850;
}

.timeline {
  display: grid;
  gap: 16px;
  padding: 18px;
}

.lane {
  display: grid;
  grid-template-columns: 92px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
}

.lane-label {
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 850;
  text-transform: uppercase;
}

.lane-track {
  position: relative;
  height: 26px;
  overflow: hidden;
  border-radius: 8px;
  background: #dfe7e9;
}

.tick {
  position: absolute;
  top: 5px;
  height: 16px;
  border-radius: 6px;
}

.tick.truth {
  background: #0f766e;
}

.tick.pred {
  background: #b45309;
}

.bar {
  display: block;
  width: 160px;
  height: 9px;
  overflow: hidden;
  border-radius: 8px;
  background: #dfe7e9;
}

.bar span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--accent), #22a398);
}

.empty,
.notes {
  margin: 0;
  padding: 18px;
  color: var(--muted);
}

.notes {
  padding-left: 38px;
}

.notes li + li {
  margin-top: 8px;
}

@media (max-width: 920px) {
  .topbar {
    grid-template-columns: 1fr;
    align-items: start;
  }

  .meta {
    justify-content: flex-start;
  }

  .summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .shell {
    width: min(100% - 20px, 1240px);
    padding-top: 14px;
  }

  h1 {
    font-size: 2rem;
  }

  .summary-grid {
    grid-template-columns: 1fr;
  }

  .metric {
    min-height: 98px;
  }
}
"""
