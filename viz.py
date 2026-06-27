import matplotlib.pyplot as plt

def save_rpeaks_preview(outpath, t, x, r_idx, fs, seconds=10):
    # plot first N seconds
    n = min(len(x), int(seconds * fs))
    fig = plt.figure()
    plt.plot(t[:n], x[:n], linewidth=1.0)
    r_in = r_idx[(r_idx >= 0) & (r_idx < n)]
    plt.scatter(t[r_in], x[r_in], s=25)
    plt.xlabel("Time [s]")
    plt.ylabel("ECG (filtered)")
    plt.title("R-peaks preview")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
def save_sqi(outpath, sqi_t, sqi, feats=None):
    fig = plt.figure()
    plt.plot(sqi_t, sqi, linewidth=2.0)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Time [s]")
    plt.ylabel("SQI [0..1]")
    plt.title("Signal Quality Index")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
