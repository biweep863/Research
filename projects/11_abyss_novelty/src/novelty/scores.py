"""Novelty (open-set) scores: higher = more likely an unknown species.

* msp     : 1 - max softmax probability (Hendrycks & Gimpel 2017)
* energy  : -logsumexp(logits)             (Liu et al. 2020)
* knn     : distance to the k-th nearest known training embedding (Sun et al. 2022)
* maha    : min class-conditional Mahalanobis distance with a shared covariance (Lee et al. 2018)
* ae      : autoencoder reconstruction error
* pca_msp : 1 - max prob of the PCA + logistic baseline
Metrics: AUROC, AUPR (unknown positive), FPR at 95 % TPR.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def msp(logits: np.ndarray) -> np.ndarray:
    return 1.0 - softmax(logits, axis=1).max(axis=1)


def energy(logits: np.ndarray) -> np.ndarray:
    return -logsumexp(logits, axis=1)


def knn(z_train: np.ndarray, z: np.ndarray, k: int = 5) -> np.ndarray:
    zt = z_train / (np.linalg.norm(z_train, axis=1, keepdims=True) + 1e-9)
    zq = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    d = np.sqrt(np.maximum(2 - 2 * zq @ zt.T, 0.0))
    d.sort(axis=1)
    return d[:, min(k, d.shape[1]) - 1]


def mahalanobis(z_train: np.ndarray, y_train: np.ndarray, z: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    mus = np.stack([z_train[y_train == c].mean(axis=0) for c in classes])
    resid = np.concatenate([z_train[y_train == c] - mus[i] for i, c in enumerate(classes)])
    cov = resid.T @ resid / len(resid) + 1e-3 * np.eye(z.shape[1])
    P = np.linalg.inv(cov)
    d = np.stack([np.einsum("ij,jk,ik->i", z - mu, P, z - mu) for mu in mus], axis=1)
    return d.min(axis=1)


def metrics(score: np.ndarray, is_unknown: np.ndarray) -> dict:
    is_unknown = np.asarray(is_unknown, dtype=int)
    if is_unknown.min() == is_unknown.max():
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    auroc = roc_auc_score(is_unknown, score)
    aupr = average_precision_score(is_unknown, score)
    fpr, tpr, _ = roc_curve(is_unknown, score)
    fpr95 = float(fpr[np.searchsorted(tpr, 0.95, side="left").clip(0, len(fpr) - 1)])
    return {"auroc": float(auroc), "aupr": float(aupr), "fpr95": fpr95}
