"""Summarise the first hardware session (ReSpeaker on a MacBook, 2026-10-05).

    python -m src.experiments.analyze_hw_session

Inputs (git-ignored raw data, kept locally):
  data/recordings/<clip>.wav  one labelled clip (0 deg, 0.5 m): 2 s silence, then talking
  data/hw_live.csv            live radar log, VAD without high-pass (as shipped from simulation)
  data/hw_rel.csv             live radar log, 150 Hz VAD high-pass + online relative range
                              (protocol: ~1 m, then near, then ~3 m, then back; no labels)
Outputs: results/hw_session.json (all numbers below), read by the figure script.

1. Room noise on the raw mics (silence of the clip): power per band, level after high-pass.
2. VAD ablation on the clip: high-pass cut-off vs speech coverage in the talking part and
   false "speech" in the silence part.
3. The two live logs: speech / ranged frame counts; per-utterance cues of the second log
   and an offline replay of the relative-range classifier with each cue (speech_db, cdr_db).
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
from pathlib import Path

import numpy as np
import scipy.signal

from ..analysis.localizer import analyze
from ..analysis.metrics import spearman
from ..analysis.relrange import RelativeRange
from ..data.respeaker_io import load_labels, load_wav, select_mics
from .common import parse_config, write_json

BANDS_HZ = [(0, 60), (60, 100), (100, 200), (200, 500), (500, 1000), (1000, 4000), (4000, 8000)]
UTT_GAP_S = 0.6  # same as audio.reset_after_silence_s


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        col = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v not in ("", "nan") else np.nan for v in col])
        except ValueError:
            out[k] = np.array(col)
    return out


def noise_spectrum(x: np.ndarray, fs: int) -> dict:
    mono = x.mean(axis=0)
    f, p = scipy.signal.welch(mono, fs, nperseg=4096)
    tot = p.sum()
    out = {"broadband_dbfs": float(10 * np.log10(np.mean(mono**2))),
           "band_power_frac": {f"{lo}-{hi}": float(p[(f >= lo) & (f < hi)].sum() / tot) for lo, hi in BANDS_HZ},
           "psd_hz": f[f <= 1000].tolist(), "psd_db": (10 * np.log10(p[f <= 1000] + 1e-20)).tolist()}
    for fc in (100, 150, 200):
        y = scipy.signal.sosfilt(scipy.signal.butter(4, fc, "highpass", fs=fs, output="sos"), mono)
        out[f"highpass_{fc}_dbfs"] = float(10 * np.log10(np.mean(y**2)))
    return out


def vad_ablation(mics: np.ndarray, cfg: dict, silence_s: float) -> list[dict]:
    rows = []
    for hp in (None, 150, 200):
        c = copy.deepcopy(cfg)
        c["vad"]["highpass_hz"] = hp
        with contextlib.redirect_stdout(io.StringIO()):
            _, frames = analyze(mics, c)
        talk = [f for f in frames if f.t > silence_s + 0.3]
        sil = [f for f in frames if 0.8 < f.t < silence_s - 0.1]  # after the noise-floor warm-up
        rows.append({"highpass_hz": hp, "talk_frames": len(talk), "speech_in_talk": sum(f.is_speech for f in talk),
                     "silence_frames": len(sil), "speech_in_silence": sum(f.is_speech for f in sil),
                     "ranged_frames": sum(f.features is not None for f in frames)})
    return rows


def utterances(log: dict) -> list[dict]:
    sel = np.isfinite(log["speech_db"])
    t = log["t"][sel]
    if not len(t):
        return []
    utt = np.r_[0, np.cumsum(np.diff(t) > UTT_GAP_S)]
    out = []
    for u in np.unique(utt):
        k = utt == u
        g = {key: log[key][sel][k] for key in ("speech_db", "snr_db", "cdr_db", "model_range_m", "track_az_robot", "fw_doa")}
        out.append({"t0": float(t[k][0]), "t1": float(t[k][-1]), "n": int(k.sum()),
                    **{f"{key}_med": float(np.nanmedian(v)) if np.isfinite(v).any() else None for key, v in g.items()},
                    "_frames": {"speech_db": g["speech_db"].tolist(), "snr_db": g["snr_db"].tolist(),
                                "cdr_db": g["cdr_db"].tolist()}})
    return out


def replay_relative(utts: list[dict], cue: str, rcfg: dict) -> list[str | None]:
    rr = RelativeRange.from_config(rcfg)
    if cue != "speech_db":
        rr.prior = []
    bands = []
    for u in utts:
        fr = u["_frames"]
        b = None
        for v, snr in zip(fr[cue], fr["snr_db"]):
            b, _ = rr.update(v, snr if cue == "speech_db" else 99.0)  # CDR: no SNR gate
        rr.end_utterance()
        bands.append(b)
    return bands


def live_counts(log: dict) -> dict:
    return {"frames": int(len(log["t"])), "seconds": float(log["t"][-1]),
            "speech_frames": int(np.nansum(log["is_speech"])),
            "ranged_frames": int(np.isfinite(log["range_meas"]).sum())}


def main():
    def extra(ap):
        ap.add_argument("--clip", default=None, help="labelled clip (default: newest in data/recordings)")
        ap.add_argument("--live", default="data/hw_live.csv")
        ap.add_argument("--rel", default="data/hw_rel.csv")

    args, cfg = parse_config(__doc__, extra)
    a = cfg["audio"]
    clip = Path(args.clip) if args.clip else sorted(Path("data/recordings").glob("*.wav"))[-1]
    lab = load_labels(clip)
    x, fs = load_wav(clip)
    mics = select_mics(x, a["mic_channels"])
    sil = float(lab.get("silence_s", 2.0))
    talk_db = float(10 * np.log10(np.mean(mics[:, int((sil + 0.3) * fs):] ** 2)))

    with contextlib.redirect_stdout(io.StringIO()):
        loc, _ = analyze(mics, cfg)
    summ = loc.clip_summary()
    fw_talk = [d for t, d, v in lab.get("firmware_log", []) if t > sil]
    out = {
        "device": {"name": "ReSpeaker 4 Mic Array (UAC1.0), USB v2.0, 6-ch firmware", "fs": fs,
                   "channels": int(x.shape[0]), "host": "macOS (MacBook Air), sounddevice/PortAudio",
                   "firmware_usb": lab.get("firmware_error") is None, "firmware_rt60_s": lab.get("firmware_rt60")},
        "clip": {"file": clip.name, "angle_deg": lab.get("angle_deg"), "distance_m": lab.get("distance_m"),
                 "talk_broadband_dbfs": talk_db, "features": summ.get("features"),
                 "model_range_m": summ.get("range_m"), "az_array_deg": summ.get("az_deg_pooled"),
                 "fw_doa_median_deg": float(np.median(fw_talk)) if fw_talk else None},
        "noise": noise_spectrum(mics[:, int(0.2 * fs): int((sil - 0.1) * fs)], fs),
        "vad_ablation": vad_ablation(mics, cfg, sil),
    }
    if Path(args.live).exists():
        out["live_no_highpass"] = live_counts(_read_csv(Path(args.live)))
    if Path(args.rel).exists():
        log = _read_csv(Path(args.rel))
        utts = utterances(log)
        rcfg = cfg["distance"]["relative"]
        replay = {cue: replay_relative(utts, cue, rcfg) for cue in ("speech_db", "cdr_db")}
        # no distance labels in this log: report agreement between the two cues and band behaviour
        big = [u for u in utts if u["n"] >= 5 and u["snr_db_med"] is not None and u["snr_db_med"] > 3]
        agree = {"n_utterances_ge5_frames": len(big),
                 "spearman_speech_vs_cdr": spearman([u["speech_db_med"] for u in big], [u["cdr_db_med"] for u in big]),
                 "speech_db_iqr": np.percentile([u["speech_db_med"] for u in big], [25, 75]).tolist(),
                 "cdr_db_iqr": np.percentile([u["cdr_db_med"] for u in big], [25, 75]).tolist()}
        for c, b in replay.items():
            b = [x for x in b if x]
            agree[f"bands_{c}"] = {k: b.count(k) / len(b) for k in ("near", "mid", "far")} if b else None
            agree[f"switch_rate_{c}"] = float(np.mean([x != y for x, y in zip(b, b[1:])])) if len(b) > 1 else None
        both = [(x, y) for x, y in zip(replay["speech_db"], replay["cdr_db"]) if x and y]
        agree["band_agreement_speech_vs_cdr"] = float(np.mean([x == y for x, y in both])) if both else None
        out["live_relative"] = {**live_counts(log), "n_utterances": len(utts), "cue_agreement": agree,
                                "utterances": [{k: v for k, v in u.items() if k != "_frames"}
                                               | {f"band_{c}": replay[c][i] for c in replay}
                                               for i, u in enumerate(utts)]}
    write_json(cfg, "hw_session.json", out)

    n = out["noise"]
    print(f"noise {n['broadband_dbfs']:.1f} dBFS; <100 Hz {100 * (n['band_power_frac']['0-60'] + n['band_power_frac']['60-100']):.0f}% "
          f"of power; after 200 Hz high-pass {n['highpass_200_dbfs']:.1f} dBFS")
    for r in out["vad_ablation"]:
        print(f"VAD hp={r['highpass_hz']}: speech {r['speech_in_talk']}/{r['talk_frames']} talk frames, "
              f"false {r['speech_in_silence']}/{r['silence_frames']} silence frames")
    for key in ("live_no_highpass", "live_relative"):
        if key in out:
            print(f"{key}: {out[key]['speech_frames']} speech / {out[key]['frames']} frames, "
                  f"{out[key]['ranged_frames']} ranged")
    if "live_relative" in out:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out["live_relative"]["cue_agreement"].items()})


if __name__ == "__main__":
    main()
