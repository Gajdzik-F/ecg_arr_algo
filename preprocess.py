import numpy as np
from scipy import signal

def resample_to_fs(time_s, x, fs_in, fs_target):
    if fs_in == fs_target:
        if time_s is None:
            t = np.arange(len(x)) / fs_target
        else:
            t = time_s
        return t, x

    # Use polyphase resampling with rational approximation
    ratio = fs_target / fs_in
    # approximate ratio with integers
    up, down = _rational_approx(ratio, max_den=1000)
    x_rs = signal.resample_poly(x, up, down)
    t_rs = np.arange(len(x_rs)) / fs_target
    return t_rs, x_rs

def _rational_approx(x, max_den=1000):
    # simple continued fraction approximation
    from fractions import Fraction
    f = Fraction(x).limit_denominator(max_den)
    return f.numerator, f.denominator

def bandpass_filter(x, fs, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    lowc = low / nyq
    highc = high / nyq
    b, a = signal.butter(order, [lowc, highc], btype="band")
    return signal.filtfilt(b, a, x)

def notch_filter(x, fs, f0=50.0, q=30.0):
    # IIR notch
    b, a = signal.iirnotch(w0=f0, Q=q, fs=fs)
    return signal.filtfilt(b, a, x)

def robust_normalize(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return (x - med) / (1.4826 * mad)
