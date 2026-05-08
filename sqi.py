import numpy as np
from scipy import signal

def _bandpower(x, fs, f1, f2):
    f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs*2)))
    mask = (f >= f1) & (f < f2)
    return float(np.trapz(pxx[mask], f[mask]) + 1e-12)

def compute_sqi(time_s, x, fs, win_s=5.0, step_s=1.0):
    """
    Returns:
      sqi_t: time centers of windows
      sqi: quality in [0,1] (higher=better)
      features dict with arrays
    """
    n = len(x)
    win = int(win_s * fs)
    step = int(step_s * fs)
    if win < 10:
        raise ValueError("SQI window is too short.")

    sqi_vals = []
    t_centers = []
    baseline_ratio = []
    hf_ratio = []
    flat_frac = []
    clip_frac = []

    # eps for flatline detection
    x_absmax = np.max(np.abs(x)) + 1e-12
    eps = 1e-6 * x_absmax

    for start in range(0, n - win + 1, step):
        seg = x[start:start+win]
        # spectral ratios
        p_total = _bandpower(seg, fs, 0.5, 40.0)
        p_bl = _bandpower(seg, fs, 0.0, 0.5)
        p_hf = _bandpower(seg, fs, 20.0, 40.0)
        br = p_bl / p_total
        hfr = p_hf / p_total

        # flatline fraction
        d = np.abs(np.diff(seg))
        flat = np.mean(d < eps)

        # clip fraction heuristic: values near 0.5% and 99.5% quantiles
        lo = np.quantile(seg, 0.005)
        hi = np.quantile(seg, 0.995)
        # if distribution is too tight, avoid false "clip"
        if (hi - lo) < 1e-9:
            clip = 1.0
        else:
            clip = np.mean((seg <= lo) | (seg >= hi))

        # normalize penalties with soft thresholds
        # baseline: good <0.2, bad >0.5
        p1 = _ramp(br, lo=0.20, hi=0.50)
        # hf: good <0.15, bad >0.35
        p2 = _ramp(hfr, lo=0.15, hi=0.35)
        # flat: good <0.05, bad >0.25
        p3 = _ramp(flat, lo=0.05, hi=0.25)
        # clip: good <0.01, bad >0.08
        p4 = _ramp(clip, lo=0.01, hi=0.08)

        # combine (flat + clip heavier)
        penalty = 0.25*p1 + 0.25*p2 + 0.30*p3 + 0.20*p4
        q = float(np.clip(1.0 - penalty, 0.0, 1.0))

        sqi_vals.append(q)
        baseline_ratio.append(br)
        hf_ratio.append(hfr)
        flat_frac.append(flat)
        clip_frac.append(clip)

        t = (start + win//2) / fs if time_s is None else time_s[start + win//2]
        t_centers.append(float(t))

    feats = {
        "baseline_ratio": np.array(baseline_ratio),
        "hf_ratio": np.array(hf_ratio),
        "flat_frac": np.array(flat_frac),
        "clip_frac": np.array(clip_frac),
    }
    return np.array(t_centers), np.array(sqi_vals), feats

def _ramp(x, lo, hi):
    # 0 below lo, 1 above hi, linear in-between
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / (hi - lo)

def sqi_at_times(sqi_t, sqi, query_t):
    """
    Interpolate SQI to arbitrary times (e.g., R-peak times).
    """
    if len(sqi_t) < 2:
        return np.ones_like(query_t, dtype=float)
    return np.interp(query_t, sqi_t, sqi, left=sqi[0], right=sqi[-1])
