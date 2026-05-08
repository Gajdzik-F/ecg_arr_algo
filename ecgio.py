import pandas as pd
import numpy as np

def load_ecg_csv(path: str):
    """
    Loads CSV that contains either:
      - columns: time, value
      - or single column value
    Returns: time (np.ndarray or None), x (np.ndarray)
    """
    df = pd.read_csv(path)

    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    time = None
    if "time" in cols and "value" in cols:
        time = df["time"].to_numpy(dtype=float)
        x = df["value"].to_numpy(dtype=float)
    elif "t" in cols and "value" in cols:
        time = df["t"].to_numpy(dtype=float)
        x = df["value"].to_numpy(dtype=float)
    elif "ecg" in cols:
        x = df["ecg"].to_numpy(dtype=float)
    elif "value" in cols:
        x = df["value"].to_numpy(dtype=float)
    else:
        # fallback: take first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            raise ValueError("CSV does not contain numeric columns.")
        x = df[num_cols[0]].to_numpy(dtype=float)

    # time unit guess (ms vs s) - safe heuristic
    if time is not None:
        dt_med = np.median(np.diff(time))
        # if dt seems like milliseconds steps (e.g., 4, 2, 1 ms)
        # treat as ms when dt_med > 0.5 and values look like ms-scale
        if dt_med > 0.5:  # likely ms
            time = time / 1000.0

    return time, x


def estimate_fs_from_time(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) < 10:
        raise ValueError("Not enough valid time samples to estimate fs.")
    fs = 1.0 / np.median(dt)
    return float(fs)
