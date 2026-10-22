"""Figures + LaTeX macros for the directional denoising project."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.plotting import COL_W, TEXT_W, apply_style  # noqa: E402

# reference categorical palette (fixed order) + neutral for baselines; every
# series also has its own marker/line style so identity is never colour-only
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GRAY, LIGHT = "#6b6b6b", "#a8a8a8"
STYLE = {
    "ref": dict(color=GRAY, marker="o", ls="-", label="raw mic (ch1)"),
    "mean": dict(color=LIGHT, marker="v", ls=":", label="mean of 4 mics"),
    "mute": dict(color=S[1], marker="s", ls="--", label="mute noisiest mic"),
    "mute_soft": dict(color=S[1], marker="D", ls=":", label="soft mic weighting"),
    "ds": dict(color=LIGHT, marker="^", ls="-.", label="delay-and-sum (DOA)"),
    "mvdr_doa": dict(color=S[4], marker="P", ls="-.", label="MVDR, far-field DOA"),
    "mvdr_rtf": dict(color=S[0], marker="o", ls="-", label="MVDR, estimated RTF"),
    "gsc": dict(color=S[2], marker="X", ls="--", label="GSC (NLMS)"),
    "oracle": dict(color=S[3], marker="*", ls=":", label="oracle MVDR"),
    "ch0_chip": dict(color="#3d3d3d", marker="h", ls="-", label="chip ch0 (home2 today)"),
}
SHORT = {"ch0_chip": "chip\nch0", "ref": "raw\nmic", "mute": "mute\nmic", "mvdr_doa": "MVDR\nDOA",
         "gsc": "GSC", "mvdr_rtf": "MVDR\nRTF"}
ORDER = ["ref", "mean", "mute", "mute_soft", "ds", "mvdr_doa", "gsc", "mvdr_rtf", "oracle"]
NOISE_LABEL = {"pink": "pink", "fan": "fan", "cafeteria": "cafeteria", "talker": "2nd talker", "music": "music"}
DIGIT = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
         "7": "Seven", "8": "Eight", "9": "Nine"}


def _name(s: str) -> str:
    s = re.sub(r"(\d+)", lambda m: "".join(DIGIT[c] for c in m.group(1)), s)
    return re.sub(r"[^A-Za-z]", "", s)


def _load(R: Path, name: str):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def _iqr(x):
    return np.percentile(x, 25), np.percentile(x, 75)


def fig_spatial(sp, out, mac):
    df = pd.DataFrame(sp["rows"])
    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W * 0.36), gridspec_kw={"width_ratios": [1, 1.25]})
    methods = [m for m in ORDER if m in set(df.method)]
    for i, m in enumerate(methods):
        g = df[df.method == m].snr_gain
        lo, hi = _iqr(g)
        st = STYLE[m]
        a1.plot([lo, hi], [i, i], color=st["color"], lw=2.2, solid_capstyle="round")
        a1.plot(np.median(g), i, marker=st["marker"], color=st["color"], ms=6, mec="white", mew=0.8, ls="none")
        a1.text(hi + 0.4, i, f"{np.median(g):+.1f}", va="center", fontsize=6.5, color="#333")
        mac[f"Gain{_name(m)}"] = f"{np.median(g):.1f}"
    a1.set_yticks(range(len(methods)), [STYLE[m]["label"] for m in methods])
    a1.axvline(0, color="#999", lw=0.6)
    a1.set_xlabel("SNR gain over raw mic (dB)")
    a1.set_title("(a) all 320 scenes")
    bins = [0, 45, 90, 135, 180]
    df["sepb"] = pd.cut(df.separation, bins)
    centers = [(a + b) / 2 for a, b in zip(bins[:-1], bins[1:])]
    for m in ["mute", "mvdr_doa", "gsc", "mvdr_rtf", "oracle"]:
        st = STYLE[m]
        y = df[df.method == m].groupby("sepb", observed=False).snr_gain.median().values
        a2.plot(centers, y, color=st["color"], marker=st["marker"], ls=st["ls"], ms=4, label=st["label"])
    a2.set_xticks(centers, [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])])
    a2.set_xlabel("talker-interferer separation (deg)")
    a2.set_ylabel("median SNR gain (dB)")
    a2.set_title("(b) gain vs separation")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "spatial.pdf")
    plt.close(fig)


ATT_ORDER = ["off", "6", "12", "18", "24", "inf"]


def _att_key(r):
    return "off" if r["dfn"] == "off" else str(r["atten"]).replace(".0", "")


def fig_atten(da, out, mac):
    df = pd.DataFrame([r for r in da["rows"] if r["method"] not in ("clean", "akane")])
    ref = pd.DataFrame([r for r in da["rows"] if r["method"] in ("clean", "akane")])
    df["att"] = [_att_key(r) for r in df.to_dict("records")]
    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W * 0.36))
    x = np.arange(len(ATT_ORDER))
    for m in ["ref", "mute", "mvdr_doa", "gsc", "mvdr_rtf", "oracle"]:
        st = STYLE[m]
        c = df[(df.method == m) & (df.dfn != "home2")]
        w = c.groupby("att").wer.mean().reindex(ATT_ORDER) * 100
        e = c.groupby("att").estoi_akane.mean().reindex(ATT_ORDER)
        a1.plot(x, w.values, color=st["color"], marker=st["marker"], ls=st["ls"], ms=4, label=st["label"])
        a2.plot(x, e.values, color=st["color"], marker=st["marker"], ls=st["ls"], ms=4)
        h = df[(df.method == m) & (df.dfn == "home2")]
        if len(h):
            a1.plot(x[2], h.wer.mean() * 100, marker=st["marker"], mfc="white", mec=st["color"], ms=6, ls="none")
            a2.plot(x[2], h.estoi_akane.mean(), marker=st["marker"], mfc="white", mec=st["color"], ms=6, ls="none")
        for att in ATT_ORDER:
            v = c[c.att == att].wer.mean()
            mac[f"Wer{_name(m)}{_name('att' + att)}"] = f"{100 * v:.0f}"
        if len(h):
            mac[f"Wer{_name(m)}HomeTwo"] = f"{100 * h.wer.mean():.0f}"
    floor = ref[ref.method == "clean"].wer.mean() * 100
    a1.axhline(floor, color="#999", lw=0.8, ls=":")
    a1.text(len(x) - 1, floor + 1, "clean speech", ha="right", fontsize=6.5, color="#666")
    mac["WerClean"] = f"{floor:.0f}"
    mac["WerAkane"] = f"{ref[ref.method == 'akane'].wer.mean() * 100:.0f}"
    labels = ["no DFN", "6", "12\n(home2)", "18", "24", "no limit"]
    for a in (a1, a2):
        a.set_xticks(x, labels)
        a.set_xlabel("DeepFilterNet3 atten_lim_db")
    a1.set_ylabel("Whisper WER (%)")
    a2.set_ylabel("ESTOI vs close-talk mic")
    a1.set_title("(a) recognition (0 dB input SNR)")
    a2.set_title("(b) intelligibility")
    h, lab = a1.get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", mfc="white", mec=GRAY, ls="none", ms=6))
    lab.append("home2 64 ms chunks")
    fig.legend(h, lab, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=6.3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "atten.pdf")
    plt.close(fig)


def fig_noise(da, out, mac):
    df = pd.DataFrame([r for r in da["rows"] if r["method"] not in ("clean", "akane")])
    df["att"] = [_att_key(r) for r in df.to_dict("records")]
    df["noise"] = df.sid.str.split("__").str[1].str.rsplit("_", n=1).str[0]
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))
    noises = ["pink", "fan", "music", "cafeteria", "talker"]
    conds = [("ref", "off"), ("ref", "12"), ("mvdr_rtf", "off"), ("mvdr_rtf", "12")]
    width = 0.2
    for k, (m, att) in enumerate(conds):
        st = STYLE[m]
        c = df[(df.method == m) & (df.att == att) & (df.dfn != "home2")]
        y = c.groupby("noise").wer.mean().reindex(noises) * 100
        xs = np.arange(len(noises)) + (k - 1.5) * width
        ax.bar(xs, y.values, width * 0.9, color=st["color"] if att == "12" else "white",
               edgecolor=st["color"], lw=1.0, hatch=None if att == "12" else "////",
               label=f"{st['label']}{' + DFN 12' if att == '12' else ''}")
        for n, v in zip(noises, y.values):
            mac[f"Wer{_name(m)}{_name('att' + att)}{_name(n)}"] = f"{v:.0f}"
    ax.set_xticks(range(len(noises)), [NOISE_LABEL[n].replace(" ", "\n") for n in noises])
    ax.set_ylabel("Whisper WER (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)
    ax.set_ylim(0, 112)
    fig.tight_layout()
    fig.savefig(out / "noise.pdf")
    plt.close(fig)


def fig_live(lv, out, mac):
    df = pd.DataFrame(lv["rows"])
    apply_style()
    fig, axes = plt.subplots(1, len(lv["takes"]), figsize=(TEXT_W, TEXT_W * 0.3), sharey=True)
    methods = ["ch0_chip", "ref", "mute", "mvdr_doa", "gsc", "mvdr_rtf"]
    for ax, t in zip(np.atleast_1d(axes), lv["takes"]):
        sub = df[df["take"] == t["name"]]
        for i, m in enumerate(methods):
            st = STYLE[m]
            off = sub[(sub.method == m) & (sub.dfn == "off")].wer.mean() * 100
            on = sub[(sub.method == m) & (sub.dfn == "cont") & (sub.atten == "12")].wer.mean() * 100
            ax.bar(i - 0.2, off, 0.38, color="white", edgecolor=st["color"], hatch="////", lw=1)
            ax.bar(i + 0.2, on, 0.38, color=st["color"], edgecolor=st["color"], lw=1)
            key = _name(t["noise"])
            mac[f"Live{key}{_name(m)}Off"] = f"{off:.0f}"
            mac[f"Live{key}{_name(m)}Dfn"] = f"{on:.0f}"
        ax.set_xticks(range(len(methods)), [SHORT[m] for m in methods], fontsize=6)
        ax.set_title(f"live: {NOISE_LABEL.get(t['noise'], t['noise'])} from the left speaker")
        ax.axhline(100, color="#bbb", lw=0.6, ls=":")
    np.atleast_1d(axes)[0].set_ylabel("Whisper WER (%)")
    from matplotlib.patches import Patch

    fig.legend([Patch(facecolor="white", edgecolor=GRAY, hatch="////"), Patch(facecolor=GRAY)],
               ["front end only", "+ DFN3 (atten 12)"], loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "live.pdf")
    plt.close(fig)
    for t in lv["takes"]:
        if t.get("superposition"):
            mac["SuperResidual"] = f"{t['superposition']['residual_db']:.1f}"
        for m, v in t.get("snr_gain_exact", {}).items():
            mac[f"LiveGain{_name(m)}"] = f"{v:.1f}"


def fig_doa(sp, out, mac):
    df = pd.DataFrame([r for r in sp["rows"] if r["method"] == "ref"])
    apply_style()
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.62))
    snrs = sorted(df.snr_in_nominal.unique())
    for k, n in enumerate(["pink", "fan", "music", "cafeteria", "talker"]):
        y = [df[(df.noise == n) & (df.snr_in_nominal == s)].doa_err.median() for s in snrs]
        ax.plot(snrs, y, color=S[k], marker="osDXP"[k], ls=["-", "--", ":", "-.", "-"][k], ms=4, label=NOISE_LABEL[n])
    for s in snrs:
        mac[f"DoaErrSnr{'M' if s < 0 else ''}{_name(str(abs(int(s))))}"] =f"{df[df.snr_in_nominal == s].doa_err.median():.0f}"
    ax.set_xlabel("input SNR (dB)")
    ax.set_ylabel("median DOA error (deg)")
    ax.set_xticks(snrs)
    ax.legend(frameon=False, ncol=2, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "doa.pdf")
    plt.close(fig)


def stats_macros(da, mac):
    """Paired (per scene) WER differences + Wilcoxon signed-rank p-values."""
    from scipy.stats import wilcoxon

    df = pd.DataFrame([r for r in da["rows"] if r["method"] not in ("clean", "akane")])
    df["att"] = [("off" if r["dfn"] == "off" else f"{r['dfn']}{r['atten']}") for r in df.to_dict("records")]
    pairs = {
        "RtfVsRef": (("mvdr_rtf", "off"), ("ref", "off")),
        "MuteVsRef": (("mute", "off"), ("ref", "off")),
        "GscVsRtf": (("gsc", "off"), ("mvdr_rtf", "off")),
        "RefDfnTwelve": (("ref", "cont12"), ("ref", "off")),
        "RtfDfnTwelve": (("mvdr_rtf", "cont12"), ("mvdr_rtf", "off")),
        "RtfDfnInf": (("mvdr_rtf", "continf"), ("mvdr_rtf", "off")),
        "RefDfnInf": (("ref", "continf"), ("ref", "off")),
        "RtfHomeTwo": (("mvdr_rtf", "home212"), ("mvdr_rtf", "cont12")),
        "RefHomeTwo": (("ref", "home212"), ("ref", "cont12")),
        "RtfDfnVsRefDfn": (("mvdr_rtf", "cont12"), ("ref", "cont12")),
    }
    for k, (a, b) in pairs.items():
        x = df[(df.method == a[0]) & (df.att == a[1])].set_index("sid").wer
        y = df[(df.method == b[0]) & (df.att == b[1])].set_index("sid").wer
        j = x.index.intersection(y.index)
        d = (x[j] - y[j]) * 100
        p = wilcoxon(x[j], y[j]).pvalue if (d != 0).any() else 1.0
        mac[f"Diff{k}"] = f"{d.mean():+.1f}"
        if p < 1e-3:
            m, e = f"{p:.0e}".split("e")
            mac[f"P{k}"] = f"${m}{{\\times}}10^{{{int(e)}}}$"
        else:
            mac[f"P{k}"] = f"{p:.2f}"
    mac["NDfnScenes"] = str(df.sid.nunique())
    mac["NTranscripts"] = f"{len(da['rows']):,}"


def p15_macros(p15, mac):
    o = p15["orientation"]
    mac["PFifteenOffset"] = f"{o['ours']['azimuth_offset_deg'] if 'ours' in o else o.get('azimuth_offset_deg', float('nan')):.0f}"
    mac["PFifteenLooErr"] = f"{p15['doa_loo_median_err_deg']:.0f}"
    mac["PFifteenSpearmanMax"] = f"{max(abs(v) for v in p15['spearman_log_distance'].values()):.2f}"
    mac["PFifteenMaeBest"] = f"{min(v for k, v in p15['loo_mae_m'].items() if k != 'predict_mean'):.2f}"
    mac["PFifteenMaeMean"] = f"{p15['loo_mae_m']['predict_mean']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    R, out = HERE / args.results, HERE / args.out
    out.mkdir(parents=True, exist_ok=True)
    mac: dict[str, str] = {}
    sp, da, lv = _load(R, "spatial.json"), _load(R, "dfn_asr.json"), _load(R, "live.json")
    if sp:
        fig_spatial(sp, out, mac)
        fig_doa(sp, out, mac)
        df = pd.DataFrame(sp["rows"])
        mac["NScenes"] = str(df.sid.nunique())
        mac["NUtts"] = str(df.utt.nunique())
    if da:
        fig_atten(da, out, mac)
        fig_noise(da, out, mac)
    if da:
        stats_macros(da, mac)
    if lv:
        fig_live(lv, out, mac)
    p15 = _load(HERE.parent / "15_spatial_audio_attention" / "results", "hw_calibration.json")
    if p15:
        p15_macros(p15, mac)
    lines = ["% auto-generated by src/figures/plot_denoise.py"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (HERE / args.macros).parent.mkdir(parents=True, exist_ok=True)
    (HERE / args.macros).write_text("\n".join(lines) + "\n")
    print(f"{len(mac)} macros, figures in {out}")


if __name__ == "__main__":
    main()
