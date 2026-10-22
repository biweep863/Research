"""Guided recording session (Part A: project 15 labelled DOA/distance; Part B: noise takes).

    python -m src.experiments.record_session calibrate --level-db -35
    python -m src.experiments.record_session noise --noise pink
    python -m src.experiments.record_session speech --angle 90 --text t3
    python -m src.experiments.record_session live --angle 90 --noise talker --text t9
    python -m src.experiments.record_session silence

Angles: degrees counter-clockwise (seen from above) from the array's 0 deg mark.
Talk takes: wait for the beep (2 s after the first chirp), read, stay silent
until the second chirp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from ..data.session import TakeSpec, run_take
from ..data.texts import TEXTS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["calibrate", "noise", "speech", "live", "silence"])
    ap.add_argument("--config", default="configs/local_small.yaml")
    ap.add_argument("--angle", type=float, help="talker azimuth (deg)")
    ap.add_argument("--dist", type=float, help="talker distance (m)")
    ap.add_argument("--noise", help="interferer type")
    ap.add_argument("--noise-angle", type=float)
    ap.add_argument("--noise-dist", type=float)
    ap.add_argument("--level-db", type=float, help="interferer playback RMS (dBFS)")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--text", help="prompt id (src/data/texts.py)")
    ap.add_argument("--effort", default="normal", choices=["soft", "normal", "loud"])
    ap.add_argument("--note", default="")
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channel", choices=["L", "R", "LR"], help="speaker for chirps + interferer")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    s = cfg["session"]
    talk = args.kind in ("speech", "live")
    noisy = args.kind in ("noise", "live", "calibrate")
    kind = "noise" if args.kind == "calibrate" else args.kind
    seconds = args.seconds or (3.0 if args.kind == "calibrate" else
                               s["speech_seconds"] if talk else s["noise_seconds"] if noisy else 15.0)
    spec = TakeSpec(
        kind=kind, seconds=seconds,
        talker_angle=args.angle if talk else None,
        talker_dist=(args.dist or s["talker_dist"]) if talk else None,
        noise=(args.noise or "pink") if noisy else None,
        noise_angle=(args.noise_angle if args.noise_angle is not None else s["noise_angle"]) if noisy else None,
        noise_dist=(args.noise_dist or s["noise_dist"]) if noisy else None,
        level_db=args.level_db if args.level_db is not None else s["noise_level_db"],
        text_id=args.text, effort=args.effort, note=args.note, seed=args.seed,
        channel=args.channel or s["noise_channel"],
    )
    if noisy and args.noise_angle is None:
        spec.noise_angle = s["speaker_angles"][spec.channel]
    if talk and args.angle is None:
        raise SystemExit("--angle is required for talk takes")
    if args.text:
        print(f"\nREAD ({args.text}):\n  {TEXTS[args.text]}\n")
    name = args.name
    if args.kind == "calibrate":
        name = name or "calib"
    out = run_take(spec, cfg, Path(s["dir"]), name=name)
    if args.kind == "calibrate":
        from ..data.wavio import load_take

        t = load_take(out)
        fs = t["fs"]
        tl = t["meta"]["timeline"]
        # content window in array samples (rough: streams start ~0.3 s before playback)
        a = int((tl["content"] / tl["fs"] + 0.3 + 0.5) * fs)
        b = int((tl["content_end"] / tl["fs"] + 0.3 - 0.3) * fs)
        raw = t["x"][1:5, a:b]
        rms = 20 * np.log10(np.sqrt(np.mean(raw**2)) + 1e-12)
        target = -32.0
        print(f"\ncalibration: playback {spec.level_db:.1f} dBFS -> raw mics {rms:.1f} dBFS RMS. "
              f"For {target:.0f} dBFS at the array use --level-db {spec.level_db + target - rms:.1f}")


if __name__ == "__main__":
    main()
