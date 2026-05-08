import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def build_hybrid_features(emb, rr_prev, rr_next):
    """
    Final feature vector:
      [Beat2Vec embedding] + [z_rr_prev, z_rr_next]
    """
    emb = np.asarray(emb, dtype=float)
    if emb.ndim != 2:
        raise ValueError("emb must have shape [N, D].")

    rr_prev = np.asarray(rr_prev, dtype=float)
    rr_next = np.asarray(rr_next, dtype=float)
    if len(rr_prev) != len(emb) or len(rr_next) != len(emb):
        raise ValueError("rr_prev and rr_next must align 1:1 with embeddings.")

    rr_feats = np.column_stack([
        _zscore_rr(rr_prev),
        _zscore_rr(rr_next),
    ])
    return np.hstack([emb, rr_feats])

def cluster_embeddings(emb, method="hdbscan", min_cluster_size=30, min_samples=10,
                       rr_feature_dims=0, rr_weight=0.75):
    emb2 = StandardScaler().fit_transform(emb)
    if rr_feature_dims:
        if rr_feature_dims < 0 or rr_feature_dims > emb2.shape[1]:
            raise ValueError("rr_feature_dims must indicate the number of trailing RR dimensions.")
        # Weight RR after standardization so their influence is not canceled out.
        emb2[:, -rr_feature_dims:] *= rr_weight

    if method == "hdbscan":
        try:
            import hdbscan
            cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = cl.fit_predict(emb2)
            # outlier score: higher = more outlier
            outlier_scores = getattr(cl, "outlier_scores_", None)
            if outlier_scores is None:
                outlier_scores = np.zeros(len(labels))
            # normalize outlier score to [0,1]
            os = outlier_scores.astype(float)
            if np.max(os) > 0:
                os = os / np.max(os)
            return labels, os
        except Exception as e:
            print("[WARN] HDBSCAN failed; falling back to DBSCAN. Reason:", e)

    # fallback: DBSCAN
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.8, min_samples=min_samples)
    labels = db.fit_predict(emb2)
    # simple outlier score: distance to kNN mean
    os = knn_outlier_score(emb2, k=10)
    return labels, os

def knn_outlier_score(emb, k=10):
    nn = NearestNeighbors(n_neighbors=min(k+1, len(emb)), metric="euclidean").fit(emb)
    d, _ = nn.kneighbors(emb)
    # d[:,0] is 0 (self); take mean of others
    score = np.mean(d[:, 1:], axis=1)
    score = (score - score.min()) / (score.max() - score.min() + 1e-12)
    return score

def get_cluster_stats(labels):
    uniq, cnt = np.unique(labels, return_counts=True)
    return dict(zip(uniq.tolist(), cnt.tolist()))

def choose_normal_cluster(labels):
    # choose largest non-negative label cluster as "normal-like"
    uniq, cnt = np.unique(labels[labels >= 0], return_counts=True)
    if len(uniq) == 0:
        return None
    return int(uniq[np.argmax(cnt)])

def _zscore_rr(rr):
    rr = np.asarray(rr, dtype=float)
    z = np.zeros_like(rr, dtype=float)
    valid = np.isfinite(rr) & (rr > 0)
    if not np.any(valid):
        return z

    mu = np.mean(rr[valid])
    sigma = np.std(rr[valid]) + 1e-12
    z[valid] = (rr[valid] - mu) / sigma
    return z
