"""Experiment 3 -- the anomalous overlap: ball and opponent hit within one
pulse width of each other.

For a grid of signed gaps (player onset minus ball onset; negative = the
player was hit first) generate 'both' windows and ask three questions:
  1. Does the classifier even notice there were two contacts?  (fraction of
     windows the gradient-boosting model labels 'both')
  2. Does the two-template separator detect two pulses, and does it get the
     order right?
  3. How precisely does it recover the gap?  (RMSE of gap estimate)
Writes results/overlap.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from src.detect.ml import GBDetector
from src.detect.separate import separate
from src.sim.dataset import clean_templates, make_windows
from src.sim.events import CLASS_ID


def main():
    cfg, out_dir = load_cfg()
    o, sim, sensors = cfg["overlap"], cfg["sim"], cfg["sensors"]
    seed = cfg.get("seed", 0)
    fs = o["fs"]
    scfg = sensors[o["sensor"]]
    rng = np.random.default_rng(seed + 33)

    # Classifier trained on the general distribution (gaps from config).
    tr = make_windows(cfg["detect"]["n_per_class"], fs, scfg, sim, rng)
    gb = GBDetector(max_iter=cfg["detect"]["gb_max_iter"], seed=seed).fit(tr.X, tr.y, fs)
    tpl = clean_templates(fs, sim, rng)

    rows = []
    for gap in o["gap_grid_ms"]:
        ws = make_windows(o["n_per_gap"], fs, scfg, sim, rng, classes=["both"], gap_ms=float(gap))
        pred = gb.predict(ws.X)
        p_both = float(np.mean(pred == CLASS_ID["both"]))
        p_ball = float(np.mean(pred == CLASS_ID["ball"]))
        p_player = float(np.mean(pred == CLASS_ID["player"]))
        two, order_ok, gap_err = 0, 0, []
        for w, tr_ in zip(ws.X, ws.truths):
            r = separate(w, fs, tpl, step_ms=o["grid_step_ms"])
            if r["model"] == "both":
                two += 1
                gap_err.append(r["gap_ms"] - tr_.gap_ms)
            order_ok += int((r["order"] == "ball_first") == tr_.ball_first)
        n = len(ws.y)
        rows.append({"gap_ms": gap, "n": n,
                     "gb_frac_both": p_both, "gb_frac_ball": p_ball, "gb_frac_player": p_player,
                     "sep_frac_two": two / n, "sep_order_acc": order_ok / n,
                     "sep_gap_rmse_ms": float(np.sqrt(np.mean(np.square(gap_err)))) if gap_err else None,
                     "sep_gap_bias_ms": float(np.mean(gap_err)) if gap_err else None})
        print(f"[overlap] gap={gap:+4d} ms  gb both={p_both:.2f}  sep two={two/n:.2f}  "
              f"order acc={order_ok/n:.2f}  gap rmse={rows[-1]['sep_gap_rmse_ms']}")

    dump(out_dir, "overlap.json", {"fs": fs, "sensor": o["sensor"], "rows": rows,
                                   "ball_dur_ms": sim["ball"]["dur_ms"],
                                   "player_dur_ms": sim["player"]["dur_ms"]})


if __name__ == "__main__":
    main()
