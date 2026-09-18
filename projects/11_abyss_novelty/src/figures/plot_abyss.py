"""Figures + LaTeX macros for the abyss-novelty paper.  Reads results/*.json,
writes six PDFs to paper/figures and every quoted number to paper/macros.tex."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt
import yaml

from utils.plotting import apply_style

SC_NAME = {"msp": "max-softmax", "energy": "energy", "knn": "kNN", "maha": "Mahalanobis", "ae": "autoencoder",
           "pca_msp": "PCA+logistic"}
SC_COL = {"msp": "#ef6c00", "energy": "#f9a825", "knn": "#1565c0", "maha": "#2e7d32", "ae": "#6a1b9a", "pca_msp": "#9e9e9e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex"); ap.add_argument("--config", default="configs/local_small.yaml")
    a = ap.parse_args()
    apply_style()
    R = Path(a.results)
    bud = json.loads((R / "budget.json").read_text()); nov = json.loads((R / "novelty.json").read_text())
    real = json.loads((R / "real.json").read_text()); clu = json.loads((R / "cluster.json").read_text())
    sur = json.loads((R / "survey.json").read_text())
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    macros = []; m = macros.append

    # ------------------------------------------------------------------ #
    # Fig 1: what reaches the abyss
    # ------------------------------------------------------------------ #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})
    L = {r["type"]: r for r in bud["light"]}
    rad = {int(r["f_hz"]): r for r in bud["radio"]}
    ac = {(r["source"], r["spreading"]): r for r in bud["acoustics"] if r["sea_state"] == 3}
    items = [
        ("satellite ocean colour (1/Kd, type I)", 1.0 / L["I"]["kd"], "#fbc02d"),
        ("ideal 2-way optics, type I", L["I"]["reach_two_way_m"], "#f9a825"),
        ("ideal 2-way optics, type III", L["III"]["reach_two_way_m"], "#ff8f00"),
        ("bioluminescence camera (in situ)", bud["biolum_camera_range_m"], "#00838f"),
        ("radar 10 GHz", rad[10000000000]["reach_m"], "#c62828"),
        ("radio 1 MHz", rad[1000000]["reach_m"], "#e53935"),
        ("ELF radio 3 Hz", rad[3]["reach_m"], "#ef5350"),
        ("eDNA plume (k=0.05/h)", next(r for r in bud["edna"] if r["k_per_h"] == 0.05)["footprint_m"], "#6a1b9a"),
        ("sound: fish chorus 130 dB", ac[("fish_chorus", "hybrid")]["range_m"], "#64b5f6"),
        ("sound: click 210 dB @20 kHz", ac[("odontocete_click", "hybrid")]["range_m"], "#1e88e5"),
        ("sound: baleen call 180 dB @50 Hz", ac[("baleen_call", "hybrid")]["range_m"], "#0d47a1"),
    ]
    ys = np.arange(len(items))[::-1]
    for y_, (lab, v, col) in zip(ys, items):
        a1.barh(y_, max(v, 0.005), color=col, height=0.7)
    a1.set_yticks(ys); a1.set_yticklabels([i[0] for i in items], fontsize=6.5)
    a1.set_xscale("log"); a1.set_xlim(0.005, 3e5)
    for d, lab in ((200, "euphotic"), (1000, "1 km"), (4000, "abyssal"), (11000, "hadal")):
        a1.axvline(d, color="grey", lw=0.6, ls=":"); a1.text(d, len(items) - 0.4, lab, fontsize=5.5, rotation=90, va="top", ha="right", color="grey")
    a1.set_xlabel("reach (m): depth for remote modalities, range for in-water ones")
    a1.set_title("What can reach the abyss?")
    fr = np.array(bud["freqs_hz"]); rv = np.array(bud["range_vs_f_180db_m"]) / 1000
    a2.loglog(fr, np.maximum(rv, 1e-2), "o-", color="#0d47a1", label="180 dB source, sea state 3")
    a2.axhline(11, color="grey", lw=0.6, ls=":"); a2.text(fr[0], 12, "hadal depth", fontsize=6, color="grey")
    a2.set_xlabel("frequency (Hz)"); a2.set_ylabel("passive detection range (km)")
    a2.set_title("Sound: range vs frequency"); a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "budget.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: example log-mel clips (synthetic families + real calls)
    # ------------------------------------------------------------------ #
    from src.sim.species import FAMILIES, make_species, synth_call
    from src.sim.propagate import received
    from src.novelty.features import log_mel
    from src.data.watkins import load_watkins
    audio, sim = cfg["audio"], cfg["sim"]
    rng = np.random.default_rng(3)
    panels = []
    for fam in FAMILIES:
        sp = make_species(0, fam, rng, tuple(sim["sl_db"]))
        c = synth_call(sp, audio["fs_hz"], audio["clip_s"], rng)
        fc = sp.params.get("f0", sp.params.get("fc", 1000))
        x, snr = received(c, sp.sl_db, fc, 5000.0, audio["fs_hz"], sim["sea_state"], sim["shipping"], rng)
        panels.append((f"synthetic: {fam}", log_mel(x, audio["fs_hz"], audio["n_mels"], audio["n_frames"], audio["n_fft"], audio["fmin_hz"], audio["fmax_hz"])))
    wk = load_watkins(HERE, audio)
    if wk is not None:
        X, lab = wk
        for name in ("Sperm_Whale", "Humpback_Whale"):
            i = np.flatnonzero(lab == name)
            if len(i):
                panels.append((f"Watkins: {name.replace('_', ' ')}", X[i[0]]))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.75, 1.7))
    for ax, (t, M) in zip(axes, panels):
        ax.imshow(M, origin="lower", aspect="auto", cmap="magma", vmin=-2.5, vmax=3)
        ax.set_title(t, fontsize=6.5); ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel("mel band", fontsize=6)
    fig.tight_layout(); fig.savefig(out / "clips.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: novelty AUROC sweeps (synthetic)
    # ------------------------------------------------------------------ #
    rows = {r["tag"]: r for r in nov["rows"]}
    scores = nov["scores"]
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.3))
    sweeps = [("range_max", "max receiver range (km)", [r for r in nov["rows"] if r["tag"].startswith("range_max")], lambda r: r["range_km"][1]),
              ("n_known", "known species in training", [r for r in nov["rows"] if r["tag"].startswith("n_known")] + [rows["base"]], lambda r: r["n_known"]),
              ("clips", "training clips per species", [r for r in nov["rows"] if r["tag"].startswith("clips")], lambda r: r["clips_per_species"])]
    for ax, (key, xl, rr, fx) in zip(axes, sweeps):
        rr = sorted(rr, key=fx)
        for s in scores:
            ax.plot([fx(r) for r in rr], [r[s]["auroc"] for r in rr], "o-", color=SC_COL[s], label=SC_NAME[s], ms=3)
        ax.plot([fx(r) for r in rr], [r["known_acc_cnn"] for r in rr], "k--", lw=0.8, label="known-class acc.")
        ax.set_xlabel(xl); ax.set_ylim(0.3, 1.02); ax.axhline(0.5, color="grey", lw=0.5, ls=":")
        if key == "range_max":
            ax.set_xscale("log")
    axes[0].set_ylabel("AUROC unknown vs known"); axes[0].set_title("Farther = quieter")
    axes[1].set_title("More known species"); axes[2].set_title("More training data")
    axes[2].legend(fontsize=5, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(out / "novelty_sweeps.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: scores, synthetic vs real
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    xx = np.arange(len(scores))
    ax.bar(xx - 0.2, [rows["base"][s]["auroc"] for s in scores], 0.4, color="#1565c0", label="synthetic (12 known + 5 unknown)")
    if real.get("available"):
        ax.bar(xx + 0.2, [real["summary"][s]["auroc_mean"] for s in scores], 0.4, color="#ef6c00",
               yerr=[real["summary"][s]["auroc_std"] for s in scores], capsize=2, label=f"Watkins ({real['n_species']} species, 6 held out)")
    ax.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax.set_xticks(xx); ax.set_xticklabels([SC_NAME[s] for s in scores], fontsize=6, rotation=20)
    ax.set_ylabel("AUROC"); ax.set_ylim(0.3, 1.0); ax.set_title("Is it a species we know?"); ax.legend(fontsize=5.5, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "novelty_real.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: from flagged clips to candidate species
    # ------------------------------------------------------------------ #
    cr = clu["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.3))
    U = [r["n_unknown"] for r in cr]
    a1.plot(U, [r["k_hat"] for r in cr], "o-", color="#6a1b9a", label="estimated new species $\\hat k$")
    a1.plot(U, [r["k_true"] for r in cr], "k--", lw=0.8, label="true (among flagged)")
    a1.set_xlabel("unknown species present"); a1.set_ylabel("species"); a1.set_title("How many new species?"); a1.legend(fontsize=6)
    a2.plot(U, [r["ari"] for r in cr], "o-", color="#6a1b9a", label="ARI")
    a2.plot(U, [r["recall_unknown"] for r in cr], "s--", color="#1565c0", label="recall of unknown clips")
    a2.plot(U, [r["precision_unknown"] for r in cr], "^:", color="#2e7d32", label="precision of flags")
    a2.set_xlabel("unknown species present"); a2.set_ylim(0, 1.05); a2.set_title("Flagging and grouping quality"); a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "cluster.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 6: how many remain, where to look
    # ------------------------------------------------------------------ #
    acc = sur["accumulation"]; al = sur["allocation"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.4))
    h = [x["hours"] for x in acc]
    a1.semilogx(h, [x["s_obs"] for x in acc], "o-", color="#1565c0", label="species observed")
    a1.errorbar(h, [x["chao1"] for x in acc], yerr=[x["chao1_se"] for x in acc], fmt="s--", color="#2e7d32", capsize=2, label="Chao1 estimate")
    a1.axhline(sur["n_species"], color="grey", lw=0.8, ls=":"); a1.text(h[0], sur["n_species"] * 0.93, "true richness", fontsize=6, color="grey")
    a1.set_xlabel("sensor-hours"); a1.set_ylabel("species"); a1.set_title("How many remain?"); a1.legend(fontsize=6, loc="upper left")
    ax2 = a1.twinx(); ax2.semilogx(h, [x["discovery_prob"] for x in acc], "^:", color="#ef6c00"); ax2.set_ylabel("P(next detection is new)", color="#ef6c00", fontsize=7)
    ax2.set_yscale("log"); ax2.tick_params(axis="y", labelsize=6, colors="#ef6c00")
    q = [x["prior_quality"] for x in al]
    a2.plot(q, [x["species_greedy"] for x in al], "o-", color="#2e7d32", label="greedy on satellite prior")
    a2.plot(q, [x["species_uniform"] for x in al], "s--", color="#9e9e9e", label="uniform")
    a2.set_xlabel("P(satellite habitat prior is right)"); a2.set_ylabel(f"species found in {sur['budget_hours']} h"); a2.set_title("Where to look?"); a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "survey.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    m(("EuphoticI", f"{L['I']['euphotic_m']:.0f}")); m(("EuphoticIII", f"{L['III']['euphotic_m']:.0f}"))
    m(("OpticsReachI", f"{L['I']['reach_two_way_m']:.0f}")); m(("OpticsReachIII", f"{L['III']['reach_two_way_m']:.0f}"))
    m(("OceanColourDepth", f"{1/L['I']['kd']:.0f}")); m(("BiolumRange", f"{bud['biolum_camera_range_m']:.0f}"))
    m(("SkinRadar", f"{1000*rad[10000000000]['skin_depth_m']:.1f}")); m(("SkinMHz", f"{100*rad[1000000]['skin_depth_m']:.0f}"))
    m(("SkinELF", f"{rad[3]['skin_depth_m']:.0f}")); m(("ReachELF", f"{rad[3]['reach_m']/1000:.1f}"))
    m(("RangeBaleenSph", f"{ac[('baleen_call','spherical')]['range_m']/1000:.0f}")); m(("RangeBaleenHyb", f"{ac[('baleen_call','hybrid')]['range_m']/1000:.0f}"))
    m(("RangeClick", f"{ac[('odontocete_click','hybrid')]['range_m']/1000:.0f}")); m(("RangeFish", f"{ac[('fish_chorus','hybrid')]['range_m']:.0f}"))
    m(("RangeSnap", f"{ac[('invertebrate_snap','hybrid')]['range_m']/1000:.1f}")); m(("RangeQuiet", f"{ac[('soft_body_quiet','hybrid')]['range_m']:.0f}"))
    e05 = next(r for r in bud["edna"] if r["k_per_h"] == 0.05)
    m(("EdnaFootprint", f"{e05['footprint_m']:.0f}")); m(("EdnaMemory", f"{e05['memory_h']:.0f}"))
    m(("ThorpFifty", f"{bud['thorp_alpha_db_km']['50']:.3f}")); m(("ThorpTwentyK", f"{bud['thorp_alpha_db_km']['20000']:.1f}"))
    b = rows["base"]
    m(("NovNKnown", str(b["n_known"]))); m(("NovNUnknown", str(b["n_unknown"]))); m(("NovClips", str(b["clips_per_species"])))
    m(("NovSnrBase", f"{b['snr_mean_db']:.0f}")); m(("NovAccBase", f"{b['known_acc_cnn']:.2f}"))
    for s in scores:
        m((f"NovAuroc{s.replace('_','').title()}", f"{b[s]['auroc']:.2f}"))
    m(("NovBestScore", SC_NAME[b["best_score"]])); m(("NovBestAuroc", f"{b[b['best_score']]['auroc']:.2f}"))
    m(("NovBestCiLo", f"{b['best_auroc_ci'][0]:.2f}")); m(("NovBestCiHi", f"{b['best_auroc_ci'][1]:.2f}"))
    m(("NovBestFpr", f"{100*b[b['best_score']]['fpr95']:.0f}"))
    r300 = rows["range_max_300"]; r10 = rows["range_max_10"]
    m(("NovSnrFar", f"{r300['snr_mean_db']:.0f}")); m(("NovSnrNear", f"{r10['snr_mean_db']:.0f}"))
    m(("NovMahaFar", f"{r300['maha']['auroc']:.2f}")); m(("NovMahaNear", f"{r10['maha']['auroc']:.2f}"))
    m(("NovKnnFar", f"{r300['knn']['auroc']:.2f}")); m(("NovMspFar", f"{r300['msp']['auroc']:.2f}")); m(("NovMspNear", f"{r10['msp']['auroc']:.2f}"))
    m(("NovMahaFour", f"{rows['n_known_4']['maha']['auroc']:.2f}")); m(("NovMspFour", f"{rows['n_known_4']['msp']['auroc']:.2f}"))
    m(("NovMahaClipsFifteen", f"{rows['clips_15']['maha']['auroc']:.2f}")); m(("NovMspClipsFifteen", f"{rows['clips_15']['msp']['auroc']:.2f}"))
    m(("NovAeMax", f"{max(r['ae']['auroc'] for r in nov['rows']):.2f}")); m(("NovPcaMax", f"{max(r['pca_msp']['auroc'] for r in nov['rows']):.2f}"))
    if real.get("available"):
        m(("RealAvailable", "yes")); m(("RealClips", str(real["n_clips"]))); m(("RealSpecies", str(real["n_species"])))
        m(("RealAcc", f"{real['known_acc_mean']:.2f}"))
        for s in scores:
            m((f"RealAuroc{s.replace('_','').title()}", f"{real['summary'][s]['auroc_mean']:.2f}"))
            m((f"RealStd{s.replace('_','').title()}", f"{real['summary'][s]['auroc_std']:.2f}"))
        best_r = max(scores, key=lambda s: real["summary"][s]["auroc_mean"])
        m(("RealBestScore", SC_NAME[best_r])); m(("RealBestAuroc", f"{real['summary'][best_r]['auroc_mean']:.2f}"))
        aris = [r["cluster"]["ari"] for r in real["rows"] if "cluster" in r]
        m(("RealAriMean", f"{np.mean(aris):.2f}" if aris else "--"))
    else:
        m(("RealAvailable", "no")); m(("RealAcc", "--")); m(("RealBestScore", "--")); m(("RealBestAuroc", "--")); m(("RealClips", "0")); m(("RealSpecies", "0")); m(("RealAriMean", "--"))
    for r in cr:
        m((f"CluAriU{r['n_unknown']}", f"{r['ari']:.2f}")); m((f"CluKhatU{r['n_unknown']}", str(r["k_hat"]))); m((f"CluKtrueU{r['n_unknown']}", str(r["k_true"])))
        m((f"CluRecallU{r['n_unknown']}", f"{r['recall_unknown']:.2f}")); m((f"CluPrecU{r['n_unknown']}", f"{r['precision_unknown']:.2f}"))
    m(("CluPrecMin", f"{min(r['precision_unknown'] for r in cr):.2f}")); m(("CluFlagQ", f"{100*clu['flag_quantile']:.0f}"))
    m(("SurS", str(sur["n_species"])))
    for x in acc:
        if x["hours"] in (10, 100, 1000, 3000):
            m((f"SurSeen{x['hours']}", f"{x['s_obs']:.0f}")); m((f"SurChao{x['hours']}", f"{x['chao1']:.0f}")); m((f"SurChaoSe{x['hours']}", f"{x['chao1_se']:.0f}"))
            m((f"SurDisc{x['hours']}", f"{100*x['discovery_prob']:.1f}"))
    for x in al:
        tag = str(x["prior_quality"]).replace(".", "")
        m((f"SurGain{tag}", f"{x['gain_pct']:.0f}")); m((f"SurGreedy{tag}", f"{x['species_greedy']:.0f}")); m((f"SurUniform{tag}", f"{x['species_uniform']:.0f}"))
    m(("SurBudget", str(sur["budget_hours"]))); m(("SurRichSea", str(sur["habitats"]["seamount"]["richness"]))); m(("SurRichPlain", str(sur["habitats"]["plain"]["richness"])))
    # LaTeX command names cannot contain digits.
    WORDS = {"2": "Two", "3": "Three", "5": "Five", "8": "Eight", "10": "Ten", "100": "Hundred", "1000": "Thousand",
             "3000": "ThreeK", "05": "Half", "07": "Seven", "09": "Nine"}
    import re
    def _fix(k):
        return re.sub(r"(\d+)", lambda mm: WORDS.get(mm.group(1), mm.group(1)), k)
    macros = [(_fix(k), v) for k, v in macros]
    assert all(not re.search(r"\d", k) for k, _ in macros), [k for k, _ in macros if re.search(r"\d", k)]
    Path(a.macros).write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros) + "\n")
    print(f"[figures] wrote {len(macros)} macros and 6 figures")


if __name__ == "__main__":
    main()
