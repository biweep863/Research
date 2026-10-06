"""Record a labelled 6-ch ReSpeaker clip for calibration / evaluation.

    python -m src.experiments.record_session --angle 90 --distance 1.5
    python -m src.experiments.record_session --angle 0 --distance 1 --seconds 10 --note "fan on"

Protocol (printed again at runtime):
  1. Mark the robot front on the floor; measure the angle CCW from the front
     (same convention as the radar) and the horizontal distance mouth->array.
  2. Stay SILENT for the first ``--silence`` seconds: this is the noise model.
  3. Talk normally (read a text, count) for the rest of the clip.
Writes data/recordings/<stamp>_a<angle>_d<dist>.wav (+ .json with labels and
the firmware DOA/VAD samples when the USB control interface is reachable).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..data.respeaker_io import FirmwarePoller, LiveSource, find_device, save_wav


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/local_small.yaml")
    ap.add_argument("--angle", type=float, required=True, help="talker azimuth, deg CCW from robot front")
    ap.add_argument("--distance", type=float, required=True, help="horizontal distance (m)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--silence", type=float, default=2.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--speaker", default="", help="who is talking (vocal-effort differences)")
    ap.add_argument("--device")
    ap.add_argument("--out", default="data/recordings")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    a = cfg["audio"]
    dev = int(args.device) if args.device and args.device.isdigit() else args.device
    if dev is None:
        dev = find_device(a["device_name"], a["device_channels"])
        if dev is None:
            raise SystemExit("ReSpeaker (6 ch) not found; use --device (see live_radar --list-devices)")

    fw = FirmwarePoller()
    chunks = []
    print(__doc__.split("Protocol")[1].split("Writes")[0])
    for k in (3, 2, 1):
        print(f"starting in {k}...")
        time.sleep(1)
    with LiveSource(dev, a["device_channels"], a["fs"], a["hop"] * 2) as src:
        t0 = time.monotonic()
        said_talk = False
        print(f"[0.0 s] STAY SILENT ({args.silence:.0f} s)")
        while time.monotonic() - t0 < args.seconds:
            b = src.read()
            if b is not None:
                chunks.append(b)
            if not said_talk and time.monotonic() - t0 > args.silence:
                print(f"[{args.silence:.1f} s] >>> TALK NOW <<<")
                said_talk = True
            time.sleep(0.01)
        overflows = src.overflows
    fw.stop()
    x = np.concatenate(chunks, axis=1)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_a{args.angle:g}_d{args.distance:g}"
    wav = Path(args.out) / f"{name}.wav"
    save_wav(wav, x, a["fs"])
    meta = {
        "angle_deg": args.angle, "distance_m": args.distance, "fs": a["fs"], "channels": x.shape[0],
        "silence_s": args.silence, "seconds": x.shape[1] / a["fs"], "note": args.note,
        "speaker": args.speaker, "overflows": overflows, "firmware_error": fw.error,
        "firmware_rt60": fw.rt60, "firmware_log": fw.log,
    }
    wav.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    peak = 20 * np.log10(np.abs(x[1:5]).max() + 1e-12)
    print(f"wrote {wav}  ({meta['seconds']:.1f} s, raw-mic peak {peak:.1f} dBFS, overflows {overflows})")
    if peak > -1:
        print("WARNING: raw mics clipped - move further or talk softer")


if __name__ == "__main__":
    main()
