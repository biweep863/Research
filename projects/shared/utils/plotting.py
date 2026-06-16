"""Matplotlib defaults tuned for a two-column LaTeX paper.

NeurIPS-style columnwidth ≈ 3.25 inches, textwidth ≈ 6.75 inches.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

COL_W = 3.25
TEXT_W = 6.75
GOLDEN = (5 ** 0.5 - 1) / 2


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def col_fig(height_ratio: float = GOLDEN):
    apply_style()
    return plt.subplots(figsize=(COL_W, COL_W * height_ratio))


def text_fig(height_ratio: float = 0.45):
    apply_style()
    return plt.subplots(figsize=(TEXT_W, TEXT_W * height_ratio))
