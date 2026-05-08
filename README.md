# ECG Arrhythmia Toolkit

Lightweight Python modules for exploratory ECG arrhythmia analysis. The code
loads ECG CSV files, preprocesses signals, estimates signal quality, extracts
beats, learns Beat2Vec-style embeddings, clusters beats, detects coarse rhythm
episodes, and writes diagnostic plots.

## Features

- CSV loading with simple time/value column detection.
- Resampling, band-pass filtering, notch filtering, and robust normalization.
- Signal Quality Index (SQI) estimates for sliding ECG windows.
- R-peak-aligned beat extraction and RR interval features.
- Self-supervised 1D convolutional beat embeddings with PyTorch.
- HDBSCAN/DBSCAN clustering and outlier scoring.
- Heuristic tachycardia, bradycardia, ectopic/outlier run, and AF-like episode
  detection.
- Matplotlib previews for R-peaks, SQI, UMAP embeddings, timelines, and cluster
  prototypes.

## Project Layout

```text
beats.py        Beat extraction and RR feature helpers
beat2vec.py     Self-supervised beat embedding model
clustering.py   Hybrid feature building, clustering, and outlier scoring
ecgio.py        ECG CSV loading and sampling-rate estimation
episodes.py     Coarse rhythm episode detection
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
`rpeaks` sample indices, use `beats.py`, `beat2vec.py`, `clustering.py`, and
`episodes.py` to build the rest of the exploratory pipeline.

## Notes

This project is intended for research and prototyping. It is not a medical
device and should not be used as the sole basis for clinical decisions.
