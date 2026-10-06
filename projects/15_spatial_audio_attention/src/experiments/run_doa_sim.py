"""Angle-of-arrival accuracy vs SNR and reverberation, SRP-PHAT variants vs GCC-PHAT.

Each clip: lead silence + one utterance from (angle, distance). Two error
measures per variant:
- pooled: one estimate from all VAD-speech frames of the clip,
- framewise: every streaming estimate (what the radar shows), median error.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np

from ..analysis.doa import gcc_phat_doa
from ..analysis.localizer import analyze
from ..analysis.metrics import angular_error_deg
from .common import Timer, parse_config, talker_clip, write_json


def _variant_cfg(cfg, variant):
    c = copy.deepcopy(cfg)
    c["audio"]["harmonic_weighting"] = variant == "srp_harmonic"
    c["audio"]["srp_noise_subtract"] = variant not in ("srp_no_nsub", "srp_plain")
    c["audio"]["srp_snr_weighting"] = variant not in ("srp_no_snrw", "srp_plain")
    return c


def main():
    _, cfg = parse_config(__doc__)
    sw = cfg["doa_sweep"]
    rows = []
    with Timer() as tm:
        for noise, snr, rt60, az, seed in itertools.product(
            sw["noise_types"], sw["snr_db"], sw["rt60_s"], sw["angles_deg"], range(sw["seeds"])
        ):
            clip = talker_clip(cfg, az, sw["distance_m"], rt60, seed=1000 + seed + az, snr_db=snr,
                               fan=noise == "fan")
            for variant in sw["variants"]:
                vcfg = _variant_cfg(cfg, "srp" if variant == "gcc" else variant)
                loc, frames = analyze(clip["mix"], vcfg)
                summ = loc.clip_summary()
                if summ["n_speech_frames"] == 0:
                    rows.append({"noise": noise, "snr_db": snr, "rt60_s": rt60, "az_deg": az, "seed": seed,
                                 "variant": variant, "detected": False})
                    continue
                if variant == "gcc":
                    tot = loc.totals
                    R = tot.Phi_x / tot.n_x
                    if tot.n_n:
                        R = R - tot.Phi_n / tot.n_n
                    est = np.rad2deg(gcc_phat_doa(R, loc.freqs, loc.mics, loc.nfft, loc.fs, loc.band, loc.c))
                    fw = float("nan")
                else:
                    est = summ["az_deg_pooled"]
                    fa = np.array([f.az_deg for f in frames if np.isfinite(f.az_deg)])
                    fw = float(np.median(angular_error_deg(fa, az))) if len(fa) else float("nan")
                rows.append({
                    "noise": noise, "snr_db": snr, "rt60_s": rt60, "az_deg": az, "seed": seed, "variant": variant,
                    "detected": True, "est_deg": float(est),
                    "err_pooled_deg": float(angular_error_deg(est, az)),
                    "err_framewise_median_deg": fw,
                    "n_speech_frames": summ["n_speech_frames"],
                })
    summary = {}
    for noise, variant, snr, rt60 in itertools.product(
        sw["noise_types"], sw["variants"], sw["snr_db"], sw["rt60_s"]
    ):
        key = f"{noise}|{variant}|{snr}|{rt60}"
        sel = [r for r in rows if r["variant"] == variant and r["snr_db"] == snr
               and r["rt60_s"] == rt60 and r["noise"] == noise and r["detected"]]
        e = np.array([r["err_pooled_deg"] for r in sel])
        fw = np.array([r["err_framewise_median_deg"] for r in sel])
        summary[key] = {
            "noise": noise, "variant": variant, "snr_db": snr, "rt60_s": rt60, "n": len(sel),
            "mae_pooled_deg": float(e.mean()) if len(e) else None,
            "median_framewise_deg": float(np.nanmedian(fw)) if len(sel) and np.isfinite(fw).any() else None,
            "acc_10deg": float((e <= 10).mean()) if len(e) else None,
        }
        mae = summary[key]["mae_pooled_deg"]
        print(f"{noise:8s} {variant:13s} snr={snr:>3} rt60={rt60:.1f}  MAE="
              + (f"{mae:6.1f}" if mae is not None else "  n/a") + f"  n={len(sel)}")
    write_json(cfg, "doa_sim.json", {"rows": rows, "summary": list(summary.values()), "wall_s": tm.s})


if __name__ == "__main__":
    main()
