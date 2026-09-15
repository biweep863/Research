"""Probabilistic decision: did the boot play the ball first, or go straight to
the player?

Evidence is combined in log-odds (a naive-Bayes fusion, each source assumed
conditionally independent given the truth):

    logit P(ball-first | e) = logit(prior[situation])
                            + LR_boot(e_boot)      classifier posterior
                            + LR_ball(e_ball)      ball-sensor timing vs boot onset
                            + LR_sep(e_sep)        two-pulse separator order

The prior is a *knowledge base*: base rates of ball-first outcomes for the
kind of challenge (standing tackle, sliding tackle, 50/50, late challenge).
Each likelihood ratio is learned on a calibration split as an isotonic map
from the source's raw score to P(ball-first), converted to a log-likelihood
ratio against the split's base rate.  The output is a calibrated posterior;
the decision rule minimises expected cost with asymmetric costs and a
review band (send to video review when the posterior is near 0.5).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..detect.ml import BinaryCalibrator


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def timing_score(t_ball_hat, t_boot_onset, sigma_ms: float) -> float:
    """Score in [0,1] from the ball sensor: how consistent is the ball's impact
    time with the boot's first contact?  If the ball reports nothing, the boot
    did not strike it (score 0).  If it fired within the timing error of the
    boot onset, the first contact was the ball; if the ball fired clearly
    *after* the boot's onset, something else was hit first."""
    if t_ball_hat is None:
        return 0.0
    d_ms = 1000.0 * (t_ball_hat - t_boot_onset)
    # P(ball contact not later than boot onset), with combined timing error.
    return float(1.0 - norm.cdf(d_ms / (sigma_ms * np.sqrt(2.0))))


class Fusion:
    """Fit per-source calibrators on one split, fuse on another."""

    def __init__(self, sources: list[str]):
        self.sources = sources
        self.cal: dict[str, BinaryCalibrator] = {}
        self.base = 0.5

    def fit(self, scores: dict[str, np.ndarray], y: np.ndarray) -> "Fusion":
        self.base = float(np.mean(y))
        for s in self.sources:
            self.cal[s] = BinaryCalibrator().fit(scores[s], y)
        return self

    def posterior(self, scores: dict[str, np.ndarray], prior: np.ndarray,
                  use: list[str] | None = None) -> np.ndarray:
        use = use or self.sources
        lo = logit(prior)
        for s in use:
            lo = lo + logit(self.cal[s](scores[s])) - logit(self.base)
        return sigmoid(lo)


def decide(p_ball_first: np.ndarray, cost_false_foul: float, cost_missed_foul: float,
           review_band: float = 0.0) -> np.ndarray:
    """0 = foul (player first), 1 = no foul (ball first), 2 = review.
    Bayes threshold on P(ball-first): call 'no foul' iff p > c_FN/(c_FP+c_FN)
    where c_FP = cost of a false foul call and c_FN = cost of a missed foul."""
    thr = cost_missed_foul / (cost_false_foul + cost_missed_foul)
    out = np.where(p_ball_first > thr, 1, 0)
    if review_band > 0:
        out = np.where(np.abs(p_ball_first - thr) < review_band, 2, out)
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error and the reliability-diagram bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    tot, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        tot += m.mean() * abs(conf - acc)
        rows.append({"bin": b, "n": int(m.sum()), "conf": conf, "acc": acc})
    return float(tot), rows
