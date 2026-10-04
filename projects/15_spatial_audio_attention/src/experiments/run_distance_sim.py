"""Can acoustic cues give the talker's distance? Features vs distance, CV error.

For every (distance, angle, RT60, seed) clip we pool speech frames and silence
frames separately (Localizer.clip_summary) and extract cdr/level/snr/tilt.
Evaluation:
- Spearman rho of each feature with log-distance (direction + strength),
- leave-one-angle-out CV MAE (m) of single-feature and all-feature models,
- cross-room transfer: leave one RT60 out, with/without RT60 as a covariate
  (on hardware the firmware's RT60 estimate would supply it),
- the inverse-square prior (no training) as a baseline.
The model on ``distance.features`` fitted on everything is saved as the
simulation prior ``distance.model_path`` used by the live radar until
hardware calibration.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..analysis.distance import FEATURES, DistanceModel
from ..analysis.localizer import analyze
from ..analysis.metrics import spearman
from .common import Timer, parse_config, talker_clip, write_json


def _cv(rows, feats, key, ridge):
    groups = sorted({r[key] for r in rows})
    errs = []
    for g in groups:
        tr = [r for r in rows if r[key] != g]
        te = [r for r in rows if r[key] == g]
        m = DistanceModel.fit([r["features"] for r in tr], [r["dist_m"] for r in tr], feats, ridge)
        errs += [abs(m.predict(r["features"]) - r["dist_m"]) for r in te]
    return float(np.mean(errs))


def main():
    _, cfg = parse_config(__doc__)
    sw, dcfg = cfg["distance_sweep"], cfg["distance"]
    prior = DistanceModel(ref_level_db=dcfg["ref_level_db"], max_range=dcfg["max_range_m"])
    rows = []
    with Timer() as tm:
        for d, az, rt60, seed in itertools.product(
            sw["distances_m"], sw["angles_deg"], sw["rt60_s"], range(sw["seeds"])
        ):
            clip = talker_clip(cfg, az, d, rt60, seed=2000 + 7 * seed + az, noise_dbfs=sw["noise_dbfs"],
                               level_jitter_db=sw["level_jitter_db"])
            loc, _ = analyze(clip["mix"], cfg, prior)
            s = loc.clip_summary()
            if s["n_speech_frames"] < 5:
                print(f"skip d={d} az={az} rt60={rt60}: {s['n_speech_frames']} speech frames")
                continue
            # rt60 as a room covariate: on hardware it comes from the firmware RT60 estimate
            rows.append({"dist_m": d, "az_deg": az, "rt60_s": rt60, "seed": seed,
                         "features": {**s["features"], "rt60_s": rt60}, "prior_range_m": s["range_m"],
                         "n_speech_frames": s["n_speech_frames"]})
            print(f"d={d:.1f} az={az:3d} rt60={rt60:.1f} " + " ".join(f"{k}={v:6.1f}" for k, v in s["features"].items()))

    ld = np.log([r["dist_m"] for r in rows])
    rho = {k: spearman([r["features"][k] for r in rows], ld) for k in FEATURES}
    # within-room correlation (what the distance cue can do once the room is known)
    rho_by_room = {
        str(rt): {k: spearman([r["features"][k] for r in rows if r["rt60_s"] == rt],
                              np.log([r["dist_m"] for r in rows if r["rt60_s"] == rt])) for k in FEATURES}
        for rt in sw["rt60_s"]
    }
    ridge = dcfg["ridge"]
    cv = {k: _cv(rows, [k], "az_deg", ridge) for k in FEATURES}
    subsets = {"all": FEATURES, "cdr+level": ["cdr_db", "level_db"],
               "cdr+level+rt60": ["cdr_db", "level_db", "rt60_s"], "cdr+rt60": ["cdr_db", "rt60_s"]}
    for name, feats in subsets.items():
        cv[name] = _cv(rows, feats, "az_deg", ridge)
    # cross-room: leave one RT60 out (needs >= 3 rooms for the rt60 covariate to vary in training)
    cross_room = {name: _cv(rows, feats, "rt60_s", ridge) for name, feats in subsets.items()}         if len(sw["rt60_s"]) > 2 else None
    prior_mae = float(np.mean([abs(r["prior_range_m"] - r["dist_m"]) for r in rows]))
    mean_baseline = float(np.mean([abs(np.mean([q["dist_m"] for q in rows]) - r["dist_m"]) for r in rows]))

    model = DistanceModel.fit([r["features"] for r in rows], [r["dist_m"] for r in rows], dcfg["features"], ridge,
                              max_range=dcfg["max_range_m"], source="simulation prior (run_distance_sim)")
    model.to_json(dcfg["model_path"])
    model_rt60 = DistanceModel.fit([r["features"] for r in rows], [r["dist_m"] for r in rows],
                                   dcfg["features_rt60"], ridge, max_range=dcfg["max_range_m"],
                                   source="simulation prior with RT60 covariate (run_distance_sim)")
    model_rt60.to_json(dcfg["model_rt60_path"])
    print("spearman:", {k: round(v, 2) for k, v in rho.items()})
    print("CV MAE (m):", {k: round(v, 2) for k, v in cv.items()}, "cross-room:", cross_room,
          "prior:", round(prior_mae, 2), "predict-mean:", round(mean_baseline, 2))
    write_json(cfg, "distance_sim.json", {
        "rows": rows, "spearman_log_distance": rho, "spearman_by_rt60": rho_by_room,
        "cv_mae_leave_angle_out_m": cv, "cross_room_mae_m": cross_room,
        "prior_inverse_square_mae_m": prior_mae, "predict_mean_mae_m": mean_baseline,
        "model": model.__dict__, "model_rt60": model_rt60.__dict__, "wall_s": tm.s,
    })


if __name__ == "__main__":
    main()
