"""Gradient-boosted classifier on the hand-crafted features.

Histogram gradient boosting (Friedman 2001; sklearn's HistGradientBoosting) is
fast on CPU, needs no scaling, and returns class probabilities that the
decision layer consumes.  Probabilities are additionally calibrated with a
held-out isotonic fit so that "0.8" means 80 %.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .features import featurize


class GBDetector:
    def __init__(self, max_iter: int = 200, seed: int = 0):
        self.clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08,
                                                  max_leaf_nodes=15, random_state=seed)
        self.fs = None

    def fit(self, X: np.ndarray, y: np.ndarray, fs: float) -> "GBDetector":
        self.fs = fs
        self.clf.fit(featurize(X, fs), y)
        return self

    def predict_proba(self, X: np.ndarray, fs: float | None = None) -> np.ndarray:
        return self.clf.predict_proba(featurize(X, fs or self.fs))

    def predict(self, X: np.ndarray, fs: float | None = None) -> np.ndarray:
        return self.clf.predict(featurize(X, fs or self.fs))


class BinaryCalibrator:
    """Isotonic calibration of a scalar score s in [0,1] to P(y=1)."""

    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, s: np.ndarray, y: np.ndarray) -> "BinaryCalibrator":
        self.iso.fit(s, y.astype(float))
        return self

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return np.clip(self.iso.predict(s), 1e-4, 1 - 1e-4)
