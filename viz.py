import numpy as np
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

def save_umap(outpath, emb2d, labels):
    fig = plt.figure()
    # color by labels; -1 are outliers
    plt.scatter(emb2d[:,0], emb2d[:,1], s=6)
    plt.title("UMAP of beat embeddings (colors not shown in this simple scatter)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_timeline(outpath, r_times, labels, outlier_score):
    fig = plt.figure()
    plt.scatter(r_times, labels, s=8)
    plt.plot(r_times, outlier_score * (np.nanmax(labels) - np.nanmin(labels) + 1) + np.nanmin(labels), linewidth=1.0)
    plt.xlabel("Time [s]")
    plt.ylabel("Cluster label")
    plt.title("Clusters over time + outlier score (scaled)")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def save_cluster_prototypes(outpath, beats, labels, max_clusters=6):
    # plot prototypes (median beat) for most frequent clusters (excluding -1)
    uniq, cnt = np.unique(labels[labels >= 0], return_counts=True)
    if len(uniq) == 0:
        return
    order = uniq[np.argsort(-cnt)]
    order = order[:max_clusters]

    fig = plt.figure(figsize=(10, 6))
    for i, c in enumerate(order, start=1):
        segs = beats[labels == c]
        proto = np.median(segs, axis=0)
        plt.plot(proto, label=f"cluster {int(c)} (n={segs.shape[0]})")
    plt.title("Cluster prototypes (median beat)")
    plt.xlabel("Samples in beat window")
    plt.ylabel("Normalized amplitude")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
