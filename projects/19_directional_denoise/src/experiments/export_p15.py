"""Export this session's talk takes as project 15 labelled clips.

Project 15's tools (``make calibrate``, ``make hw``) expect 6-ch WAVs that
start with ``silence_s`` of talker silence followed by speech, and a JSON
sidecar with ``angle_deg`` / ``distance_m``. Our takes also contain the
alignment chirps and the beep, which would enter p15's silence model, so the
export keeps only [talker silence | utterance] (the same cut as the scenes).
No high-pass: p15 applies its own 200 Hz VAD high-pass.

    python -m src.experiments.export_p15 [--out ../15_spatial_audio_attention/data/recordings]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.scenes import load_speech
from ..data.wavio import save_wav
from .common import ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="data/session")
    ap.add_argument("--out", default="../15_spatial_audio_attention/data/recordings")
    ap.add_argument("--silence", type=float, default=2.0)
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    n = 0
    for p in sorted((ROOT / args.session).glob("*.json")):
        meta = json.loads(p.read_text())
        if meta["spec"]["kind"] != "speech":
            continue
        sp = load_speech(p.with_suffix(".wav"), args.silence, 30.0)
        name = f"20261018_{meta['name']}"
        save_wav(out / f"{name}.wav", sp.x, 16000)
        s = meta["spec"]
        lab = {"angle_deg": s["talker_angle"], "distance_m": s["talker_dist"], "fs": 16000,
               "channels": int(sp.x.shape[0]), "silence_s": sp.lead / 16000, "seconds": sp.x.shape[1] / 16000,
               "note": f"exported from project 19 session ({meta['name']}); {s.get('note', '')}".strip(),
               "speaker": "luis", "effort": s.get("effort", "normal"), "overflows": meta["array"]["overflows"],
               "firmware_error": "not read: Windows, no libusb driver", "firmware_rt60": None, "firmware_log": [],
               "angle_frame": "talker's own marks: 0 deg = USB port, approximate (+-20 deg)"}
        (out / f"{name}.json").write_text(json.dumps(lab, indent=1))
        n += 1
    print(f"exported {n} clips to {out}")


if __name__ == "__main__":
    main()
