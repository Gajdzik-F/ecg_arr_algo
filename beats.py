import numpy as np

def extract_beats(x_filt, rpeaks, fs, pre_s=0.25, post_s=0.45):
    pre = int(pre_s * fs)
    post = int(post_s * fs)
    L = pre + post

    beats = []
    beat_r = []
    valid_idx = []
    for i, r in enumerate(rpeaks):
        a = r - pre
        b = r + post
        if a < 0 or b > len(x_filt):
            continue
        seg = x_filt[a:b].copy()
        beats.append(seg)
        beat_r.append(r)
        valid_idx.append(i)

    beats = np.array(beats, dtype=float)  # [N, L]
    beat_r = np.array(beat_r, dtype=int)

    # robust per-beat normalization
    beats = _normalize_beats(beats)

    return beats, beat_r

def _normalize_beats(beats):
    # subtract median and divide by MAD per beat
    med = np.median(beats, axis=1, keepdims=True)
    mad = np.median(np.abs(beats - med), axis=1, keepdims=True) + 1e-12
    beats_n = (beats - med) / (1.4826 * mad)
    return beats_n

def compute_rr_times(rpeaks, fs):
    # RR in seconds aligned to each R (RR_prev, RR_next)
    t = rpeaks / fs
    rr_prev = np.full_like(t, fill_value=np.nan, dtype=float)
    rr_next = np.full_like(t, fill_value=np.nan, dtype=float)
    if len(t) >= 2:
        rr = np.diff(t)
        rr_prev[1:] = rr
        rr_next[:-1] = rr
    return t, rr_prev, rr_next

def hr_from_rr(rr_s):
    # HR = 60/RR
    hr = 60.0 / rr_s
    return hr

def select_rr_for_beats(beat_r, rpeaks, rr_prev, rr_next):
    """
    Align RR features to beats retained by `extract_beats`.

    Returns RR arrays for the kept beats and their indices in the original
    `rpeaks` array.
    """
    beat_r = np.asarray(beat_r, dtype=int)
    rpeaks = np.asarray(rpeaks, dtype=int)
    rr_prev = np.asarray(rr_prev, dtype=float)
    rr_next = np.asarray(rr_next, dtype=float)

    if rr_prev.shape != rpeaks.shape or rr_next.shape != rpeaks.shape:
        raise ValueError("rr_prev and rr_next must have the same shape as rpeaks.")

    idx = np.searchsorted(rpeaks, beat_r)
    matched = (idx < len(rpeaks)) & (rpeaks[idx] == beat_r)
    if not np.all(matched):
        raise ValueError("Could not match all retained beats to rpeaks.")

    return rr_prev[idx], rr_next[idx], idx
