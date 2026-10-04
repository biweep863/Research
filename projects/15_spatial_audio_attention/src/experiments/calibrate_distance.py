"""Calibrate and evaluate on labelled hardware recordings (record_session.py).

    python -m src.experiments.calibrate_distance --dir data/recordings

1. Orientation: fits the array->robot mapping (azimuth_offset_deg, mirror)
   from our DOA vs the labelled angle, and the same for the firmware DOA.
   Paste the suggestion into configs/*.yaml.
2. DOA accuracy after calibration (ours vs firmware), leave-one-out so the
   offset is not fitted on the clip it is scored on.
3. Distance: per-feature Spearman, leave-one-recording-out MAE for
   (a) the simulation prior, (b) a 1-parameter inverse-square fit,
   (c) ridge models (CDR only, ``distance.features``, all features);
   saves the ``distance.features`` model as distance.hw_model_path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from ..analysis.distance import FEATURES, DistanceModel
from ..analysis.geometry import array_to_robot_deg
from ..analysis.localizer import analyze
from ..analysis.metrics import angular_error_deg, spearman
from ..analysis.tracker import circular_mean_deg
from ..data.respeaker_io import load_labels, load_wav, select_mics
from .common import write_json
from .live_radar import load_model


def fit_orientation(est_deg, true_deg):
    """Best (offset, mirror) mapping array azimuths onto labelled robot azimuths."""
    best = None
    for mirror in (False, True):
        mapped = array_to_robot_deg(est_deg, 0.0, mirror)
        offset = circular_mean_deg(np.asarray(true_deg) - mapped)
        err = float(np.median(angular_error_deg(array_to_robot_deg(est_deg, offset, mirror), true_deg)))
        if best is None or err < best[2]:
            best = (offset, mirror, err)
    return best


def loo_orientation_errors(est, true):
    errs = []
    for k in range(len(est)):
        idx = [i for i in range(len(est)) if i != k]
        off, mir, _ = fit_orientation(np.asarray(est)[idx], np.asarray(true)[idx])
        errs.append(float(angular_error_deg(array_to_robot_deg(est[k], off, mir), true[k])))
    return errs


def loo_mae(rows, fit_fn):
    errs = []
    for k in range(len(rows)):
        m = fit_fn([r for i, r in enumerate(rows) if i != k])
        errs.append(abs(m.predict(rows[k]["features"]) - rows[k]["distance_m"]))
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/local_small.yaml")
    ap.add_argument("--dir", default="data/recordings")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    a, dcfg = cfg["audio"], cfg["distance"]
    sim_model = load_model(cfg)

    rows = []
    for wav in sorted(Path(args.dir).glob("*.wav")):
        lab = load_labels(wav)
        if "angle_deg" not in lab:
            print(f"skip {wav.name}: no label json")
            continue
        x, fs = load_wav(wav)
        assert fs == a["fs"], f"{wav.name}: fs {fs} != {a['fs']}"
        loc, _ = analyze(select_mics(x, a["mic_channels"]), cfg, sim_model)
        s = loc.clip_summary()
        if s["n_speech_frames"] < 5:
            print(f"skip {wav.name}: only {s['n_speech_frames']} speech frames (VAD gate too high?)")
            continue
        fw_log = lab.get("firmware_log") or []
        talk = [d for t, d, v in fw_log if t > lab.get("silence_s", 0)]
        rows.append({
            "file": wav.name, "angle_deg": lab["angle_deg"], "distance_m": lab["distance_m"],
            "speaker": lab.get("speaker", ""), "note": lab.get("note", ""),
            "az_array_deg": s["az_deg_pooled"], "doa_conf": s["conf_pooled"], "f0_hz": s["f0_median_hz"],
            "fw_doa_deg": circular_mean_deg(talk) if talk else None,
            "features": s["features"], "sim_prior_range_m": s["range_m"],
            "n_speech_frames": s["n_speech_frames"], "n_noise_frames": s["n_noise_frames"],
        })
        print(f"{wav.name}: label az={lab['angle_deg']:5.1f} d={lab['distance_m']:.2f} | array az={s['az_deg_pooled']:6.1f} "
              f"f0={s['f0_median_hz']:4.0f} " + " ".join(f"{k}={v:6.1f}" for k, v in s["features"].items()))
    if len(rows) < 2:
        raise SystemExit("need >= 2 labelled recordings")

    out = {"n_recordings": len(rows), "rows": rows}
    true = np.array([r["angle_deg"] for r in rows])
    est = np.array([r["az_array_deg"] for r in rows])
    off, mir, med = fit_orientation(est, true)
    out["orientation"] = {"azimuth_offset_deg": off, "mirror": mir, "median_err_in_sample_deg": med}
    if len(rows) >= 3:
        out["doa_loo_median_err_deg"] = float(np.median(loo_orientation_errors(est, true)))
    fw_rows = [r for r in rows if r["fw_doa_deg"] is not None]
    if len(fw_rows) >= 3:
        fe = np.array([r["fw_doa_deg"] for r in fw_rows])
        ft = np.array([r["angle_deg"] for r in fw_rows])
        fo, fm, _ = fit_orientation(fe, ft)
        out["firmware"] = {"azimuth_offset_deg": fo, "mirror": fm,
                           "loo_median_err_deg": float(np.median(loo_orientation_errors(fe, ft)))}
    print(f"\nSuggested config:  azimuth_offset_deg: {off:.1f}   mirror: {str(mir).lower()}"
          f"   (in-sample median error {med:.1f} deg)")
    if "doa_loo_median_err_deg" in out:
        print(f"DOA leave-one-out median error: ours {out['doa_loo_median_err_deg']:.1f} deg"
              + (f", firmware {out['firmware']['loo_median_err_deg']:.1f} deg" if "firmware" in out else ""))

    dists = sorted({r["distance_m"] for r in rows})
    if len(dists) >= 2 and len(rows) >= 4:
        ld = np.log([r["distance_m"] for r in rows])
        out["spearman_log_distance"] = {k: spearman([r["features"][k] for r in rows], ld) for k in FEATURES}

        def fit_inv_sq(tr):
            ref = float(np.mean([r["features"]["level_db"] + 20 * np.log10(r["distance_m"]) for r in tr]))
            return DistanceModel(ref_level_db=ref, max_range=dcfg["max_range_m"])

        def fit_ridge(tr, feats=tuple(dcfg["features"])):
            return DistanceModel.fit([r["features"] for r in tr], [r["distance_m"] for r in tr], feats,
                                     dcfg["ridge"], max_range=dcfg["max_range_m"])

        out["loo_mae_m"] = {
            "sim_prior": float(np.mean([abs(r["sim_prior_range_m"] - r["distance_m"]) for r in rows])),
            "inverse_square_fitted": loo_mae(rows, fit_inv_sq),
            "ridge_cdr": loo_mae(rows, lambda tr: fit_ridge(tr, ["cdr_db"])),
            "ridge_configured": loo_mae(rows, fit_ridge),
            "ridge_all": loo_mae(rows, lambda tr: fit_ridge(tr, FEATURES)),
            "predict_mean": float(np.mean([abs(np.mean([q["distance_m"] for q in rows]) - r["distance_m"])
                                           for r in rows])),
        }
        model = fit_ridge(rows)
        model.source = f"hardware calibration ({len(rows)} clips, {args.dir})"
        model.to_json(dcfg["hw_model_path"])
        print("Spearman vs log-distance:", {k: round(v, 2) for k, v in out["spearman_log_distance"].items()})
        print("LOO MAE (m):", {k: round(v, 2) for k, v in out["loo_mae_m"].items()})
        print(f"saved {dcfg['hw_model_path']} (live_radar will prefer it)")
    else:
        print("distance: need >= 2 distinct distances and >= 4 clips to calibrate")
    write_json(cfg, "hw_calibration.json", out)


if __name__ == "__main__":
    main()
