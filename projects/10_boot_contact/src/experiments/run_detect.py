"""Experiment 1 -- contact classification.

For each background (synthetic gait / real PAMAP2 ankle IMU), sensor grade
(impact-rated 200 g / phone-grade 16 g) and sampling rate, generate labelled
windows {none, ball, player, both}, fit the connected-ball-style threshold
rule and the gradient-boosting classifier on a train split, and score both on
a held-out split (accuracy, macro-F1, confusion).  Also stores one example
window per class for the paper's signature figure.  Writes results/detect.json.
"""
from __future__ import annotations

import time

import numpy as np

from .common import dump, load_cfg, project_root
from src.data.pamap2 import load_pamap2_ankle
from src.detect.ml import GBDetector
from src.detect.threshold import ThresholdDetector, confusion, macro_f1
from src.sim.dataset import make_windows
from src.sim.events import CLASSES


def main():
    cfg, out_dir = load_cfg()
    d = cfg["detect"]
    sim, sensors = cfg["sim"], cfg["sensors"]
    seed = cfg.get("seed", 0)
    real = load_pamap2_ankle(project_root())
    print(f"[detect] PAMAP2 background: {'available (%d segments)' % len(real) if real else 'NOT available -> synthetic only'}")

    rows = []
    for bg in d["backgrounds"]:
        if bg == "pamap2" and not real:
            continue
        fs_list = [d["pamap2_fs"]] if bg == "pamap2" else d["fs_grid"]
        for sensor in d["sensors"]:
            for fs in fs_list:
                rng = np.random.default_rng(seed + fs)
                n = d["n_per_class"]
                n_tr = int(round(n * (1 - d["test_frac"])))
                tr = make_windows(n_tr, fs, sensors[sensor], sim, rng, background=bg, real_segments=real)
                te = make_windows(n - n_tr, fs, sensors[sensor], sim, rng, background=bg, real_segments=real)
                t0 = time.time()
                thr = ThresholdDetector().fit(tr.X, tr.y, fs)
                p_thr = thr.predict(te.X, fs)
                gb = GBDetector(max_iter=d["gb_max_iter"], seed=seed).fit(tr.X, tr.y, fs)
                p_gb = gb.predict(te.X)
                row = {
                    "background": bg, "sensor": sensor, "fs": fs,
                    "n_train": len(tr.y), "n_test": len(te.y),
                    "threshold": {"acc": float(np.mean(p_thr == te.y)), "f1": macro_f1(te.y, p_thr),
                                  "tau_g": thr.tau, "w_split_ms": thr.w_split_ms,
                                  "confusion": confusion(te.y, p_thr).tolist()},
                    "gb": {"acc": float(np.mean(p_gb == te.y)), "f1": macro_f1(te.y, p_gb),
                           "confusion": confusion(te.y, p_gb).tolist()},
                    "clip_frac": float(np.mean(np.abs(te.X[:, :, :3]).max(axis=(1, 2)) >= sensors[sensor]["range_g"] - 1e-6)),
                    "time_s": time.time() - t0,
                }
                rows.append(row)
                print(f"[detect] bg={bg:9s} sensor={sensor:8s} fs={fs:5d}  "
                      f"thr F1={row['threshold']['f1']:.3f}  gb F1={row['gb']['f1']:.3f}  "
                      f"clipped={100*row['clip_frac']:.0f}%")

    # Example windows (sport sensor, 1 kHz, synthetic) for the signature figure.
    rng = np.random.default_rng(seed + 999)
    fs_ex = max(d["fs_grid"])
    ex = make_windows(1, fs_ex, sensors["sport"], sim, rng, classes=CLASSES)
    examples = {CLASSES[i]: {"acc": ex.X[i, :, :3].tolist(),
                             "t_ball": ex.truths[i].t_ball, "t_player": ex.truths[i].t_player}
                for i in range(len(CLASSES))}

    dump(out_dir, "detect.json", {"classes": CLASSES, "rows": rows, "fs_example": fs_ex,
                                  "examples": examples, "pamap2_available": bool(real)})


if __name__ == "__main__":
    main()
