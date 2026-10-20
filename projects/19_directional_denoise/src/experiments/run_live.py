"""Stage 3: the live takes (talker reading while the interferer plays).

1. Superposition check: the talker live take replayed the *same* interferer
   samples as its noise-only take, so in the talker-silent stretch before the
   beep ``live ~= g * noise_take``. The residual says how repeatable the
   acoustic path is, i.e. how far the mixed scenes are from reality. The
   residual over the utterance is then the live take's speech component, which
   gives an exact SNR on a real live recording too.
2. Front ends on the live audio, including the chip's own processed ch0 (what
   home2 uses today), with and without DFN -> Whisper WER.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np

from ..analysis import asr, spatial
from ..analysis.geometry import respeaker_v2_positions
from ..analysis.metrics import snr_db, wer
from ..analysis.stft import istft, stft
from ..data.align import align
from ..data.noises import beep, chirp
from ..data.scenes import _load
from ..data.texts import TEXTS
from ..data.wavio import load_wav, save_wav
from .common import ROOT, setup, work_dir, write


def segments(t, fs, lead_s):
    tl = t["meta"]["timeline"]
    al = align(t["x"][1:5], fs, tl)
    fp, c1 = tl["fs"], tl["chirp1"]
    noise_start = c1 + len(chirp(fp)) + int(0.6 * fp)  # interferer starts after the gap
    pre_end = al.to_rec(tl["beep"] - c1) - 0.05 * fs
    utt0 = al.to_rec(tl["beep"] + len(beep(fp)) - c1) + 0.1 * fs
    utt1 = al.to_rec(tl["content_end"] - c1)
    return al, int(round(pre_end)), int(round(utt0)), int(round(utt1)), noise_start


def main():
    cfg, out = setup(__doc__)
    a, an, d = cfg["audio"], cfg["analysis"], cfg["dfn"]
    fs, nfft, hop = a["fs"], a["nfft"], a["hop"]
    mics = respeaker_v2_positions(a["mic_radius"])
    lead = int(an["lead_s"] * fs)
    wd = work_dir(cfg) / "live"
    sess = ROOT / cfg["session"]["dir"]
    res = {"takes": []}
    jobs = []
    for meta_p in sorted(sess.glob("live_*.json")):
        meta = json.loads(meta_p.read_text())
        t = _load(meta_p.with_suffix(".wav"), an["highpass_hz"])
        sp = meta["spec"]
        al, pre_end, u0, u1, _ = segments(t, fs, an["lead_s"])
        x6 = np.concatenate([t["x"][:, pre_end - lead : pre_end], t["x"][:, u0:u1]], axis=1)
        x = x6[1:5]
        info = {"name": meta["name"], "noise": sp["noise"], "channel": sp["channel"], "angle": sp["talker_angle"],
                "text_id": sp["text_id"]}

        # 1. superposition check against the noise-only take
        nz_p = sess / f"noise_{sp['noise']}_{sp['channel']}.wav"
        tn = _load(nz_p, an["highpass_hz"])
        tln = tn["meta"]["timeline"]
        aln = align(tn["x"][1:5], fs, tln)
        played_live = t["play"][meta["timeline"]["chirp1"] :]
        played_nz = tn["play"][tln["chirp1"] :]
        # playback offset of the interferer start in each take (relative to chirp1)
        k_live = meta["timeline"]["beep"] - int(cfg["playback"]["pre_silence_s"] * tln["fs"]) - meta["timeline"]["chirp1"]
        k_nz = tln["content"] - tln["chirp1"]
        same = float(np.corrcoef(played_live[k_live : k_live + 48000 * 5], played_nz[k_nz : k_nz + 48000 * 5])[0, 1])
        info["same_interferer_played"] = same
        if same > 0.99:
            # recording index in the noise take that corresponds to each live sample
            def nz_index(i_live):
                n_play = (i_live - al.offset) / (al.ratio * fs / al.fs_play) - k_live + k_nz
                return aln.to_rec(n_play)

            idx = np.concatenate([np.arange(pre_end - lead, pre_end), np.arange(u0, u1)])
            ni = np.round(nz_index(idx)).astype(int)
            ok = (ni >= 0) & (ni < tn["x"].shape[1])
            n4 = np.zeros_like(x)
            n4[:, ok] = tn["x"][1:5, ni[ok]]
            pre = slice(0, lead)
            g = float(np.sum(x[:, pre] * n4[:, pre]) / np.sum(n4[:, pre] ** 2))
            resid = x[:, pre] - g * n4[:, pre]
            info["superposition"] = {
                "gain": g,
                "residual_db": float(10 * np.log10(np.sum(resid**2) / np.sum(x[:, pre] ** 2))),
                "room_floor_db": None,
            }
            s4 = x - g * n4  # speech component of the live take (+ unmatched residual)
            S_live, N_live = s4, g * n4
        else:
            S_live = N_live = None

        # 2. front ends on the live mixture (noise model = the lead)
        f, X = stft(x, fs, nfft, hop)
        Rn = spatial.covariance(X, slice(0, lead // hop))
        h = spatial.gevd_rtf(spatial.covariance(X, slice(lead // hop, None)), Rn)
        az, _ = spatial.doa_from_mixture(X, Rn, lead // hop, f, mics)
        W = {"ref": spatial.ref_mic(4, len(f)), "mean": spatial.mic_mean(4, len(f)),
             "mute": spatial.mute_loudest(Rn), "mvdr_doa": spatial.mvdr(Rn, spatial.farfield_h(mics, az, f), an["loading_ff"]),
             "mvdr_rtf": spatial.mvdr(Rn, h, an["loading"]), "gsc": spatial.gsc(X, h, lead // hop, an["gsc_mu"], an["gsc_gate_db"])}
        T = x.shape[1]
        outs = {m: istft(spatial.apply(Wm, X), fs, nfft, hop, T)[lead:] for m, Wm in W.items()}
        outs["ch0_chip"] = x6[0, lead:]
        exact = {}
        if S_live is not None:
            _, S = stft(S_live, fs, nfft, hop)
            _, Nn = stft(N_live, fs, nfft, hop)
            for m, Wm in W.items():
                so = istft(spatial.apply(Wm, S), fs, nfft, hop, T)[lead:]
                no = istft(spatial.apply(Wm, Nn), fs, nfft, hop, T)[lead:]
                exact[m] = snr_db(so, no)
            exact = {m: v - exact["ref"] for m, v in exact.items()}
        info["snr_gain_exact"] = exact
        for m, y in outs.items():
            p = wd / f"{meta['name']}__{m}.wav"
            save_wav(p, y / (np.abs(y).max() + 1e-9) * 0.5, fs)
            jobs.append({"in": str(p), "out": str(wd / "dfn" / f"{meta['name']}__{m}"),
                         "modes": ["cont", "home2"], "attens": [12, None]})
        res["takes"].append(info)

    spec = wd / "dfn_jobs.json"
    spec.write_text(json.dumps({"jobs": jobs, "modes": ["cont"], "attens": [12]}))
    subprocess.run([str((ROOT / d["python"]).resolve()), str(ROOT / "src" / "dfn_runner.py"), str(spec)], check=True, cwd=ROOT)
    rows = []
    for info in res["takes"]:
        text = TEXTS[info["text_id"]]
        for m in ["ref", "mean", "mute", "mvdr_doa", "mvdr_rtf", "gsc", "ch0_chip"]:
            conds = [("off", None, wd / f"{info['name']}__{m}.wav")]
            conds += [(mode, att, wd / "dfn" / f"{info['name']}__{m}__{mode}__att{att}.wav")
                      for mode in ("cont", "home2") for att in ("12", "inf")]
            for mode, att, p in conds:
                y = load_wav(p)[0][0]
                hyp = asr.transcribe(y, cfg["asr"]["model"], cfg["asr"]["language"])
                rows.append({"take": info["name"], "method": m, "dfn": mode, "atten": att,
                             "wer": wer(text, hyp), "hyp": hyp})
    res["rows"] = rows
    write(out / "live.json", res)


if __name__ == "__main__":
    main()
