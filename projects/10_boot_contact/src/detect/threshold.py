"""The connected-ball baseline: an impact is a sudden change of the
acceleration vector; its kind is read off two numbers.

    |a|_peak > tau          -> a contact happened (vs. gait background)
    width_half < w_split    -> ball (short, stiff) else player (long, soft)
    a second peak later     -> both

The two thresholds are fitted on training windows by a grid search on macro-F1,
so the baseline is as strong as a threshold rule can be.
"""
from __future__ import annotations

import numpy as np

from .features import featurize

PEAK, WIDTH, NPEAKS = 0, 2, 15


class ThresholdDetector:
    def __init__(self, tau: float = 20.0, w_split_ms: float = 15.0, n_split: float = 1.5):
        self.tau, self.w_split_ms, self.n_split = tau, w_split_ms, n_split

    def fit(self, X: np.ndarray, y: np.ndarray, fs: float) -> "ThresholdDetector":
        F = featurize(X, fs)
        best = (-1.0, self.tau, self.w_split_ms)
        taus = np.quantile(F[:, PEAK], np.linspace(0.05, 0.95, 37))
        widths = np.linspace(4.0, 60.0, 29)
        for tau in taus:
            for w in widths:
                pred = self._predict_F(F, tau, w)
                f1 = macro_f1(y, pred)
                if f1 > best[0]:
                    best = (f1, float(tau), float(w))
        _, self.tau, self.w_split_ms = best
        return self

    def _predict_F(self, F, tau, w):
        contact = F[:, PEAK] > tau
        short = F[:, WIDTH] < w
        multi = F[:, NPEAKS] > self.n_split
        pred = np.zeros(len(F), dtype=int)
        pred[contact & short] = 1
        pred[contact & ~short] = 2
        pred[contact & multi] = 3
        return pred

    def predict(self, X: np.ndarray, fs: float) -> np.ndarray:
        return self._predict_F(featurize(X, fs), self.tau, self.w_split_ms)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int = 4) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def confusion(y: np.ndarray, pred: np.ndarray, n_classes: int = 4) -> np.ndarray:
    M = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y, pred):
        M[a, b] += 1
    return M
