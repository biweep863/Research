"""VAD high-pass cut-off vs DOA at 0 dB SNR (simulation side of the hardware fix).

On hardware the raw mics carry strong sub-200 Hz room noise, and a 200 Hz
high-pass on the VAD input was needed (analyze_hw_session.py). This checks what
the same filter costs where that noise does not exist: the 0 dB cells of the
DOA sweep (8 angles x RT60 x {diffuse, fan}), full SRP-PHAT, one clip per angle.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np

from ..analysis.localizer import analyze
from ..analysis.metrics import angular_error_deg
from .common import Timer, parse_config, talker_clip, write_json

CUTOFFS = [None, 100, 150, 200]


def main():
    _, cfg = parse_config(__doc__)
    sw = cfg["doa_sweep"]
    rows = []
    with Timer() as tm:
        for noise, rt60 in itertools.product(sw["noise_types"], sw["rt60_s"]):
            clips = [(az, talker_clip(cfg, az, sw["distance_m"], rt60, seed=1000 + az, snr_db=0,
                                      fan=noise == "fan")) for az in sw["angles_deg"]]
            for hp in CUTOFFS:
                c = copy.deepcopy(cfg)
                c["vad"]["highpass_hz"] = hp
                errs, n_speech = [], []
                for az, clip in clips:
                    loc, _ = analyze(clip["mix"], c)
                    s = loc.clip_summary()
                    n_speech.append(s["n_speech_frames"])
                    if s["n_speech_frames"]:
                        errs.append(float(angular_error_deg(s["az_deg_pooled"], az)))
                rows.append({"noise": noise, "rt60_s": rt60, "highpass_hz": hp, "n_detected": len(errs),
                             "mae_deg": float(np.mean(errs)), "median_deg": float(np.median(errs)),
                             "max_deg": float(np.max(errs)), "mean_speech_frames": float(np.mean(n_speech))})
                r = rows[-1]
                print(f"{noise:7s} rt60={rt60} hp={str(hp):4s} MAE {r['mae_deg']:5.1f} median {r['median_deg']:4.1f} "
                      f"max {r['max_deg']:5.1f} detected {r['n_detected']}/{len(clips)}")
    write_json(cfg, "vad_highpass_sim.json", {"rows": rows, "wall_s": tm.s})


if __name__ == "__main__":
    main()
