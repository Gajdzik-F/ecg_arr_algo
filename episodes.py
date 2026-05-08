import numpy as np
import pandas as pd

def detect_episodes(r_times, rr_prev, labels, outlier_score, sqi_at_r,
                    tachy_hr=110, brady_hr=50,
                    ectopic_run_len=3,
                    sqi_min=0.4):
    """
    Produces coarse episodes:
      - tachy, brady based on HR
      - ectopic_run: consecutive non-normal clusters or high outlier score
      - af_like: high RR irregularity over window + not-too-bad SQI (heuristic)
    """
    n = len(r_times)
    hr = np.full(n, np.nan)
    valid = np.isfinite(rr_prev) & (rr_prev > 0)
    hr[valid] = 60.0 / rr_prev[valid]

    # mark reliable beats
    reliable = sqi_at_r >= sqi_min

    episodes = []

    def add_episode(t0, t1, typ, note):
        episodes.append({"start_s": float(t0), "end_s": float(t1), "type": typ, "note": note})

    # tachy/brady episodes: contiguous stretches
    add_episode_runs(episodes, r_times, reliable & (hr > tachy_hr), "tachy", f"HR>{tachy_hr}")
    add_episode_runs(episodes, r_times, reliable & (hr < brady_hr), "brady", f"HR<{brady_hr}")

    # ectopic-like run: labels != normal OR outlier_score high
    # we don't know normal label here; we treat -1 as outlier and also high outlier_score
    ect = reliable & ((labels == -1) | (outlier_score > 0.7))
    add_episode_runs(episodes, r_times, ect, "ectopic_or_outlier_run", "labels=-1 or outlier_score>0.7", min_len=ectopic_run_len)

    # AF-like: RR irregularity in sliding window (very heuristic)
    # High RMSSD over 10 beats + not too many missing RR
    af_flags = af_like_flags(rr_prev, reliable, win_beats=10, rmssd_thr=0.12)  # 120ms RMSSD-ish
    add_episode_runs(episodes, r_times, af_flags, "af_like", "high RR irregularity (RMSSD)")

    return pd.DataFrame(episodes)

def add_episode_runs(episodes, r_times, mask, typ, note, min_len=5):
    # mask is per beat. episode duration based on r_times.
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
    # group consecutive indices
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            if (prev - start + 1) >= min_len:
                episodes.append({"start_s": float(r_times[start]), "end_s": float(r_times[prev]), "type": typ, "note": note})
            start = i
            prev = i
    if (prev - start + 1) >= min_len:
        episodes.append({"start_s": float(r_times[start]), "end_s": float(r_times[prev]), "type": typ, "note": note})

def rmssd(rr):
    d = np.diff(rr)
    return np.sqrt(np.mean(d*d))

def af_like_flags(rr_prev, reliable, win_beats=10, rmssd_thr=0.12):
    flags = np.zeros_like(rr_prev, dtype=bool)
    rr = rr_prev.copy()
    rr[~np.isfinite(rr)] = np.nan

    for i in range(len(rr)):
        a = max(0, i - win_beats + 1)
        seg = rr[a:i+1]
        rel = reliable[a:i+1]
        seg = seg[rel & np.isfinite(seg)]
        if len(seg) >= max(6, win_beats//2):  # enough beats
            if rmssd(seg) > rmssd_thr:
                flags[i] = True
    return flags
