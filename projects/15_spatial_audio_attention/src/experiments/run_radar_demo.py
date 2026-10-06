"""Moving-talker demo: writes a simulated ReSpeaker 6-ch WAV + ground truth and
scores the streaming tracker on it (what the radar would show).

Two passes over the same audio: the metric distance model (relative range off)
for the azimuth / metres track, and the online near/mid/far bands
(``distance.relative``) scored against the band of the true distance.

Outputs:
  data/demo_moving_speaker.wav (+ .json waypoints)  -> replay with live_radar --wav
  results/radar_demo.json                           -> per-frame track vs truth
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..analysis.localizer import analyze
from ..analysis.metrics import angular_error_deg
from ..data.respeaker_io import save_wav, select_mics
from ..data.simulate import moving_speaker_demo
from .common import Timer, parse_config, write_json
from .live_radar import load_model


def main():
    _, cfg = parse_config(__doc__)
    with Timer() as tm:
        x6, fs, wps = moving_speaker_demo(cfg, seed=cfg["seed"])
        wav = Path("data/demo_moving_speaker.wav")
        save_wav(wav, x6, fs)
        wav.with_suffix(".json").write_text(json.dumps({"fs": fs, "waypoints": wps}, indent=2))
        print(f"wrote {wav} ({x6.shape[1] / fs:.1f} s)")

        mics = select_mics(x6, cfg["audio"]["mic_channels"])
        metric_cfg = copy.deepcopy(cfg)
        metric_cfg["distance"].setdefault("relative", {})["enabled"] = False
        loc, frames = analyze(mics, metric_cfg, load_model(cfg))
        rel_cfg = cfg["distance"].get("relative") or {}
        _, rel_frames = analyze(mics, cfg, load_model(cfg)) if rel_cfg.get("enabled") else (None, None)
    track = []
    for f in frames:
        wp = next((w for w in wps if w["t0"] <= f.t <= w["t1"]), None)
        track.append({
            "t": f.t, "is_speech": f.is_speech, "az_meas": f.az_deg, "track_az": f.track_az_deg,
            "track_range": f.track_range_m, "track_active": f.track_active,
            "true_az": wp["az_deg"] if wp else None, "true_range": wp["dist_m"] if wp else None,
        })
    # score only frames inside an utterance and after a 0.5 s settling time
    scored = [r for r, f in zip(track, frames) if r["true_az"] is not None and r["track_active"]
              and any(w["t0"] + 0.5 <= f.t <= w["t1"] for w in wps)]
    az_err = angular_error_deg([r["track_az"] for r in scored], [r["true_az"] for r in scored])
    rng_err = np.array([abs(r["track_range"] - r["true_range"]) for r in scored if np.isfinite(r["track_range"])])
    per_wp = []
    for w in wps:
        sel = [r for r in scored if w["t0"] + 0.5 <= r["t"] <= w["t1"]]
        if sel:
            per_wp.append({**w, "median_az_err": float(np.median(angular_error_deg(
                [r["track_az"] for r in sel], w["az_deg"]))),
                "median_range": float(np.nanmedian([r["track_range"] for r in sel]))})
    bands = None
    if rel_frames is not None:
        # true band: split at the geometric means of the radar radii
        r = rel_cfg.get("band_radius_m", [0.75, 1.75, 3.0])
        edges = (np.sqrt(r[0] * r[1]), np.sqrt(r[1] * r[2]))
        true_band = lambda d: "near" if d < edges[0] else "mid" if d < edges[1] else "far"  # noqa: E731
        per = []
        for w in wps:
            sel = [f.range_band for f in rel_frames if w["t0"] + 0.5 <= f.t <= w["t1"] and f.range_band]
            if sel:
                per.append({"dist_m": w["dist_m"], "true": true_band(w["dist_m"]),
                            "pred": max(set(sel), key=sel.count), "frac_correct":
                            float(np.mean([b == true_band(w["dist_m"]) for b in sel]))})
        bands = {"cue": rel_cfg.get("cue", "speech_db"), "edges_m": list(edges), "per_waypoint": per,
                 "frame_accuracy": float(np.mean([p["frac_correct"] for p in per])) if per else None}
        for b in per:
            print(f"  band d={b['dist_m']:.1f} true={b['true']:4s} pred={b['pred']:4s} ({b['frac_correct']:.0%} frames)")
    summary = {
        "n_scored_frames": len(scored),
        "median_track_az_err_deg": float(np.median(az_err)) if len(az_err) else None,
        "p90_track_az_err_deg": float(np.percentile(az_err, 90)) if len(az_err) else None,
        "median_track_range_err_m": float(np.median(rng_err)) if len(rng_err) else None,
        "per_waypoint": per_wp,
        "relative_bands": bands,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_waypoint"}, indent=1))
    for w in per_wp:
        print(f"  az={w['az_deg']:4d} d={w['dist_m']:.1f}  az_err={w['median_az_err']:5.1f}  r_est={w['median_range']:.2f}")
    write_json(cfg, "radar_demo.json", {"summary": summary, "track": track, "wall_s": tm.s})


if __name__ == "__main__":
    main()
