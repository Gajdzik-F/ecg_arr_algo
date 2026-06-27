# ECG Arrhythmia Toolkit

Lightweight Python modules for PASM-Rhythm ECG arrhythmia research. The code
loads ECG CSV files or PhysioNet records, preprocesses signals, estimates signal
quality, extracts R-peak-aligned beats, runs the PASM rhythm-state algorithm,
and writes Markdown/HTML validation reports.

## Features

- CSV loading with simple time/value column detection.
- Resampling, band-pass filtering, notch filtering, and robust normalization.
- Signal Quality Index (SQI) estimates for sliding ECG windows.
- R-peak-aligned beat extraction and RR interval features.
- PASM-Rhythm prototype with patient memory, rhythm context features, typed
  beat-state graph evidence, uncertainty-aware state scoring, and episode
  decoding.
- PhysioNet evidence layers for AFDB fast-irregular AF and MITDB short-coupled
  ectopy.
- Matplotlib previews for R-peaks and SQI.

## Project Layout

```text
beats.py        Beat extraction and RR feature helpers
ecgio.py        ECG CSV loading and sampling-rate estimation
pasm_rhythm.py  Patient-adaptive rhythm memory, state graph, and episode decoder
pasm_dataset.py Per-beat PASM feature dataset builder
pasm_ml_decoder.py Lightweight NumPy softmax state decoder
pasm_ml_validation.py Patient-wise ML decoder validation report
pasm_validation.py Synthetic cohort generation and PASM episode metrics
pasm_physionet.py Optional WFDB/PhysioNet validation harness
pasm_realdata.py Combined PhysioNet smoke presets and summary report
pasm_html_report.py Static HTML report renderer for real-data validation results
preprocess.py   Resampling, filtering, and normalization
sqi.py          Signal quality index calculation
viz.py          Plotting helpers
```

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Minimal Usage

```python
from ecgio import load_ecg_csv, estimate_fs_from_time
from preprocess import bandpass_filter, robust_normalize
from sqi import compute_sqi

time_s, x = load_ecg_csv("ecg.csv")
fs = estimate_fs_from_time(time_s)
x_filt = robust_normalize(bandpass_filter(x, fs))
sqi_t, sqi, features = compute_sqi(time_s, x_filt, fs)
```

R-peak detection is expected to happen outside this toolkit. Once you have
`rpeaks` sample indices, use `beats.py` and `pasm_rhythm.py` to build the PASM
pipeline.

## PASM-Rhythm Prototype

The first PASM-Rhythm implementation can run from R-peak-aligned rhythm inputs:

```python
from beats import compute_rr_times
from pasm_rhythm import run_pasm_rhythm

r_times, rr_prev, rr_next = compute_rr_times(rpeaks, fs)
result = run_pasm_rhythm(
    r_times,
    rr_prev,
    rr_next=rr_next,
    beats=beats,
    sqi_at_r=sqi_at_r,
)

episodes = result["episodes"]
patient_memory = result["patient_memory"]
rhythm_graph = result["graph"]
```

This is a deterministic baseline for the PASM-Rhythm direction: patient-specific
memory, multi-beat rhythm context, SQI-aware uncertainty, and episode-level
output. It is not yet the final learned model or clinical validation.

## Validation

Run the unit tests with Python 3.11:

```bash
py -3.11 -m unittest discover -s tests -v
```

Run the deterministic synthetic cohort benchmark:

```bash
py -3.11 pasm_validation.py --train-records 30 --holdout-records 30 --seed 2026 --out PASM_VALIDATION.md
```

Current train/holdout synthetic benchmark summary:

```text
model         episode_f1_mean  precision  recall  false_alarms_per_hour
pasm_tuned    0.834            0.773      0.917   17.581
pasm_default  0.749            0.733      0.792   19.256
```

This benchmark validates the code path and episode-level behavior on controlled
synthetic rhythm cohorts. Decoder thresholds are tuned only on the synthetic
training cohort and reported on a separate holdout cohort. The next validation
step is patient-wise evaluation on real annotated ECG databases.

Run the PASM v0.4 FP-aware learned-decoder benchmark suite:

```bash
py -3.11 pasm_ml_validation.py --preset tiny --out PASM_ML_VALIDATION.md
py -3.11 pasm_ml_validation.py --preset mini --out PASM_ML_BENCHMARK.md
```

This trains lightweight NumPy softmax decoders on PASM feature tables using a
patient-wise split, then compares deterministic `pasm_physionet`, raw
`pasm_ml_decoder`, guarded `pasm_ml_decoder_guarded`, and FP-aware
`pasm_ml_decoder_fpaware` on holdout records. The report includes per-record
FP/h, false positives by type, hard-negative counts, guard removals, and top
false-positive episodes. It intentionally adds no heavy ML dependencies.

Current PASM learned-decoder tiny holdout summary:

```text
pasm_physionet            episode F1 0.500, precision 0.500, recall 0.500, false alarms/hour 12.000
pasm_ml_decoder           episode F1 0.222, precision 0.500, recall 0.333, false alarms/hour 4.333
pasm_ml_decoder_fpaware   episode F1 0.200, precision 0.810, recall 0.333, false alarms/hour 4.000
pasm_ml_decoder_guarded   episode F1 0.200, precision 0.810, recall 0.333, false alarms/hour 4.000
```

The learned variants now beat the deterministic baseline on false alarms/hour,
but they still lose recall and episode F1. The next tuning target is recall
recovery without returning to high FP/h.

Optional PhysioNet smoke tests require `wfdb` and internet access:

```bash
py -3.11 pasm_realdata.py --preset smoke --out PASM_REALDATA_SMOKE.md
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md
py -3.11 pasm_physionet.py --db mitdb --records 200 --max-seconds 900 --out PASM_PHYSIONET_VALIDATION.md
py -3.11 pasm_physionet.py --db afdb --records 04015 04126 --max-seconds 900 --out PASM_AFDB_VALIDATION.md
```

Generate static HTML reports next to the Markdown reports:

```bash
py -3.11 pasm_realdata.py --preset smoke --out PASM_REALDATA_SMOKE.md --html-out reports/pasm_smoke.html
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md --html-out reports/pasm_mini.html
```

Serve a generated HTML report on localhost:

```bash
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md --serve --port 8765
```

Generate a per-record diagnostic HTML report:

```bash
py -3.11 pasm_realdata.py --preset mini --out PASM_REALDATA_SUMMARY.md --diagnose-record mitdb/203 --diagnostic-html-out reports/mitdb_203_diagnostic.html
```

The PhysioNet harness currently validates data loading, annotation mapping, beat
extraction, PASM execution, and episode metrics. MITDB/203 false positives
remain the next algorithmic target. Current short-record smoke tests are
intentionally tracked as regression evidence:

```text
Combined smoke preset: episode F1 0.822, precision 1.000, recall 0.722, false alarms/hour 0.000
Combined mini preset: episode F1 0.710, precision 0.786, recall 0.667, false alarms/hour 5.143
MITDB 200, first 900 s: episode F1 0.667, precision 1.000, recall 0.500, false alarms/hour 0.000
AFDB 04015 + 04126, first 900 s: episode F1 0.900, precision 1.000, recall 0.833, false alarms/hour 0.000
```

## Notes

This project is intended for research and prototyping. It is not a medical
device and should not be used as the sole basis for clinical decisions.
