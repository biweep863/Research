"""Moving-talker demo: writes a simulated ReSpeaker 6-ch WAV + ground truth and
scores the streaming tracker on it (what the radar would show).

Outputs:
  data/demo_moving_speaker.wav (+ .json waypoints)  -> replay with live_radar --wav
  results/radar_demo.json                           -> per-frame track vs truth
"""

from __future__ import annotations

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

        loc, frames = analyze(select_mics(x6, cfg["audio"]["mic_channels"]), cfg, load_model(cfg))
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
    summary = {
        "n_scored_frames": len(scored),
        "median_track_az_err_deg": float(np.median(az_err)) if len(az_err) else None,
        "p90_track_az_err_deg": float(np.percentile(az_err, 90)) if len(az_err) else None,
        "median_track_range_err_m": float(np.median(rng_err)) if len(rng_err) else None,
        "per_waypoint": per_wp,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_waypoint"}, indent=1))
    for w in per_wp:
        print(f"  az={w['az_deg']:4d} d={w['dist_m']:.1f}  az_err={w['median_az_err']:5.1f}  r_est={w['median_range']:.2f}")
    write_json(cfg, "radar_demo.json", {"summary": summary, "track": track, "wall_s": tm.s})


if __name__ == "__main__":
    main()
