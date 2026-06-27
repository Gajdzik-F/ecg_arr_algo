from dataclasses import dataclass

import numpy as np
import pandas as pd

from pasm_dataset import FEATURE_COLUMNS
from pasm_rhythm import PASM_STATES


@dataclass
class SoftmaxPASMDecoder:
    classes: tuple
    feature_columns: tuple
    fill_values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    def predict_proba(self, frame):
        x = _prepare_features(frame, self.feature_columns, self.fill_values, self.mean, self.scale)
        probs = _softmax(x @ self.weights + self.bias[None, :])
        return pd.DataFrame(probs, columns=list(self.classes), index=frame.index)

    def save_npz(self, path):
        np.savez(
            path,
            classes=np.asarray(self.classes, dtype=object),
            feature_columns=np.asarray(self.feature_columns, dtype=object),
            fill_values=self.fill_values,
            mean=self.mean,
            scale=self.scale,
            weights=self.weights,
            bias=self.bias,
        )


def fit_softmax_decoder(
    train_df,
    feature_columns=None,
    label_column="label",
    classes=PASM_STATES,
    epochs=800,
    lr=0.05,
    l2=1e-3,
    seed=2026,
    max_class_weight=8.0,
    sample_weight_boost=None,
):
    feature_columns = tuple(feature_columns or FEATURE_COLUMNS)
    classes = tuple(classes)
    if len(train_df) == 0:
        raise ValueError("Cannot train PASM ML decoder on an empty frame.")

    labels = train_df[label_column].astype(str).to_numpy()
    class_to_idx = {label: i for i, label in enumerate(classes)}
    unknown = sorted(set(labels) - set(classes))
    if unknown:
        raise ValueError(f"Unknown labels for PASM decoder: {unknown}")

    fill_values, mean, scale = _fit_normalizer(train_df, feature_columns)
    x = _prepare_features(train_df, feature_columns, fill_values, mean, scale)
    y = np.asarray([class_to_idx[label] for label in labels], dtype=int)
    sample_weight = _balanced_sample_weights(y, len(classes), max_class_weight=max_class_weight)
    if sample_weight_boost is not None:
        boost = np.asarray(sample_weight_boost, dtype=float)
        if len(boost) != len(train_df):
            raise ValueError("sample_weight_boost must align 1:1 with train_df.")
        sample_weight = sample_weight * np.where(np.isfinite(boost) & (boost > 0.0), boost, 1.0)

    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(x.shape[1], len(classes)))
    bias = np.zeros(len(classes), dtype=float)
    y_onehot = np.eye(len(classes), dtype=float)[y]
    weight_norm = max(float(sample_weight.sum()), 1e-12)

    for _ in range(int(epochs)):
        logits = x @ weights + bias[None, :]
        probs = _softmax(logits)
        err = (probs - y_onehot) * (sample_weight[:, None] / weight_norm)
        grad_w = x.T @ err + float(l2) * weights
        grad_b = err.sum(axis=0)
        weights -= float(lr) * grad_w
        bias -= float(lr) * grad_b

    return SoftmaxPASMDecoder(
        classes=classes,
        feature_columns=feature_columns,
        fill_values=fill_values,
        mean=mean,
        scale=scale,
        weights=weights,
        bias=bias,
    )


def _fit_normalizer(frame, feature_columns):
    x = frame.loc[:, list(feature_columns)].astype(float).to_numpy()
    fill_values = np.nanmedian(x, axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0)
    filled = np.where(np.isfinite(x), x, fill_values[None, :])
    mean = filled.mean(axis=0)
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return fill_values.astype(float), mean.astype(float), scale.astype(float)


def _prepare_features(frame, feature_columns, fill_values, mean, scale):
    x = frame.loc[:, list(feature_columns)].astype(float).to_numpy()
    x = np.where(np.isfinite(x), x, fill_values[None, :])
    return (x - mean[None, :]) / scale[None, :]


def _balanced_sample_weights(y, n_classes, max_class_weight=8.0):
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights_by_class = np.zeros(n_classes, dtype=float)
    present = counts > 0
    weights_by_class[present] = len(y) / (float(present.sum()) * counts[present])
    if max_class_weight is not None:
        weights_by_class[present] = np.minimum(weights_by_class[present], float(max_class_weight))
    return weights_by_class[y]


def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)
