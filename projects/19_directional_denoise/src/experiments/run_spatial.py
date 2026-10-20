"""Stage 1: every multichannel front end on every mixed scene (exact output SNR).

Scenes = 8 utterances (talker around the array) x 5 interferers x 2 speakers
x SNRs. Writes results/spatial.json (one row per scene x method) and, for the
DFN / ASR stages, 16 kHz mono WAVs of each method's output in the work dir.
"""

from __future__ import annotations

import numpy as np

from ..analysis import spatial
from ..analysis.doa import cross_spectra, peak_azimuth, srp_phat
from ..analysis.geometry import respeaker_v2_positions
from ..analysis.metrics import si_sdr, snr_db
from ..analysis.stft import istft, stft
from ..data.scenes import load_session, make_scene
from ..data.wavio import save_wav
from .common import ROOT, circ_diff, setup, work_dir, write


def clean_doa(x4, fs, a, mask=None, band=(300.0, 4000.0)):
    """High-SNR direction of a source from its own clean take (the label we trust)."""
    f, X = stft(x4, fs, a["nfft"], a["hop"])
    if mask is not None:
        X = X[..., np.resize(mask[:: a["hop"]], X.shape[-1])]
    grid = np.deg2rad(np.arange(0, 360, 1.0))
    P = srp_phat(cross_spectra(X), f, respeaker_v2_positions(a["mic_radius"]), grid, band=band)
    az, conf = peak_azimuth(P, grid)
    return float(np.rad2deg(az)), conf


def main():
    cfg, out = setup(__doc__)
    a, an = cfg["audio"], cfg["analysis"]
    fs, nfft, hop = a["fs"], a["nfft"], a["hop"]
    mics = respeaker_v2_positions(a["mic_radius"])
    speech, noise = load_session(ROOT / cfg["session"]["dir"], an["lead_s"], an["max_s"], an["highpass_hz"])
    speech = [s for s in speech if not an.get("utterances") or s.name in an["utterances"]]
    noise = [n for n in noise if n.kind in an["noises"]]
    print(f"{len(speech)} utterances x {len(noise)} interferer takes x {len(an['snrs'])} SNRs")

    # labels from the clean takes (array frame)
    src_dir = {s.name: clean_doa(s.x[1:5], fs, a, s.active) for s in speech}
    src_dir.update({n.name: clean_doa(n.x[1:5], fs, a) for n in noise})

    wd = work_dir(cfg)
    rng = np.random.default_rng(cfg["seed"])
    rows, scenes = [], []
    for sp in speech:
        save_wav(wd / f"{sp.name}__clean.wav", sp.x[1, sp.lead :], fs)
        if sp.ref is not None:
            save_wav(wd / f"{sp.name}__akane.wav", sp.ref[sp.lead :] / (np.abs(sp.ref).max() + 1e-9) * 0.5, fs)
        for nz in noise:
            for snr in an["snrs"]:
                sc = make_scene(sp, nz, snr, rng)
                f, S = stft(sc.s, fs, nfft, hop)
                _, Nn = stft(sc.n, fs, nfft, hop)
                X = S + Nn
                lead_f = sp.lead // hop
                act_f = np.resize(sp.active[::hop], X.shape[-1])
                W, diag = spatial.all_weights(X, S, Nn, lead_f, act_f, f, mics, an)
                T = sc.s.shape[1]
                ev = slice(sp.lead, T)
                act = sp.active[ev]
                s_ref = sc.s[0, ev]
                sep = circ_diff(src_dir[sp.name][0], src_dir[nz.name][0])
                base = {"sid": sc.sid, "utt": sp.name, "talker_angle": sp.angle, "noise": nz.kind,
                        "speaker": nz.channel, "snr_in_nominal": snr,
                        "talker_doa": src_dir[sp.name][0], "noise_doa": src_dir[nz.name][0], "separation": sep,
                        "doa_est": diag["doa_deg"], "doa_err": circ_diff(diag["doa_deg"], src_dir[sp.name][0])}
                snr_in = snr_db(s_ref, sc.n[0, ev], act)
                for name, Wm in W.items():
                    so = istft(spatial.apply(Wm, S), fs, nfft, hop, T)[ev]
                    no = istft(spatial.apply(Wm, Nn), fs, nfft, hop, T)[ev]
                    yo = so + no
                    rows.append({**base, "method": name, "snr_in": snr_in,
                                 "snr_out": snr_db(so, no, act), "si_sdr": si_sdr(yo, s_ref),
                                 "speech_si_sdr": si_sdr(so, s_ref)})
                    if snr in an["write_snrs"]:
                        save_wav(wd / f"{sc.sid}__{name}.wav", yo, fs)
                if snr in an["write_snrs"]:
                    scenes.append({"sid": sc.sid, "utt": sp.name, "text_id": sp.text_id})
        print(f"  {sp.name}: done ({len(rows)} rows)")
    for r in rows:
        r["snr_gain"] = r["snr_out"] - r["snr_in"]
    write(out / "spatial.json", {"rows": rows, "scenes": scenes, "src_doa": src_dir,
                                 "methods": list(W.keys()), "config": an})


if __name__ == "__main__":
    main()
