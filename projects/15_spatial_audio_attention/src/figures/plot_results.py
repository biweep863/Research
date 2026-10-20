"""Figures + LaTeX macros for the spatial audio attention project."""
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

from utils.plotting import TEXT_W, apply_style, col_fig  # noqa: E402

DIGIT_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
              "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
VARIANT_LABEL = {"srp": "SRP-PHAT (full)", "srp_harmonic": "+ F0 harmonic weights",
                 "srp_no_nsub": "no noise subtraction", "srp_no_snrw": "no SNR weights",
                 "srp_plain": "plain SRP-PHAT", "gcc": "GCC-PHAT + LS"}
FEAT_LABEL = {"cdr_db": "CDR (dB)", "level_db": "level (dBFS)", "snr_db": "SNR (dB)", "tilt_db": "tilt (dB)"}


def _name(s: str) -> str:
    s = re.sub(r"(\d+)", lambda m: "".join(DIGIT_WORD[c] for c in m.group(1)), s)
    return re.sub(r"[^A-Za-z]", "", s)


def _load(R, name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_doa(doa, out, m):
    sw = sorted({r["snr_db"] for r in doa["summary"]})
    noises = sorted({r["noise"] for r in doa["summary"]})
    rts = sorted({r["rt60_s"] for r in doa["summary"]})
    variants = list(dict.fromkeys(r["variant"] for r in doa["summary"]))
    apply_style()
    fig, axes = plt.subplots(1, len(noises), figsize=(TEXT_W, TEXT_W * 0.33), sharey=True)
    for ax, noise in zip(np.atleast_1d(axes), noises):
        for v in variants:
            y = []
            for snr in sw:
                sel = [r["mae_pooled_deg"] for r in doa["summary"]
                       if r["noise"] == noise and r["variant"] == v and r["snr_db"] == snr and r["mae_pooled_deg"] is not None]
                y.append(np.mean(sel) if sel else np.nan)
            ax.plot(sw, y, marker="o", ms=3, label=VARIANT_LABEL.get(v, v),
                    lw=2 if v == "srp" else 1, ls="-" if v == "srp" else "--")
        ax.set_yscale("log")
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"{noise} noise (mean over RT60 {', '.join(map(str, rts))} s)")
    np.atleast_1d(axes)[0].set_ylabel("DOA MAE (deg)")
    np.atleast_1d(axes)[-1].legend(loc="upper right", frameon=False)
    fig.savefig(out / "doa_error.pdf")
    plt.close(fig)
    for r in doa["summary"]:
        if r["mae_pooled_deg"] is None:
            continue
        key = _name(f"doa {r['noise']} {r['variant']} snr{r['snr_db']} rt{int(round(r['rt60_s'] * 10))}")
        m(key, f"{r['mae_pooled_deg']:.1f}")
        if r.get("median_framewise_deg") is not None:
            m(key + "Med", f"{r['median_framewise_deg']:.1f}")
        m(key + "N", str(r["n"]))


def fig_distance(dist, out, m):
    rows = dist["rows"]
    rts = sorted({r["rt60_s"] for r in rows})
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_W, TEXT_W * 0.3))
    for ax, k in zip(axes, FEAT_LABEL):  # rt60_s is a covariate, not plotted
        for j, rt in enumerate(rts):
            sel = [r for r in rows if r["rt60_s"] == rt]
            dx = (j - (len(rts) - 1) / 2) * 0.05  # small dodge so rooms do not overlap
            ax.scatter([r["dist_m"] + dx for r in sel], [r["features"][k] for r in sel], s=5, label=f"RT60 {rt}s")
        ax.set_xlabel("distance (m)")
        rho = dist["spearman_log_distance"][k]
        ax.set_title(FEAT_LABEL[k] + "\n" + r"$\rho$" + f" = {rho:+.2f}")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "distance_features.pdf")
    plt.close(fig)

    cv = dist["cv_mae_leave_angle_out_m"]
    xr = dist["cross_room_mae_m"] or {}
    labels = ["predict mean", "inverse-sq. prior", *cv.keys(), *[f"cross-room: {k}" for k in xr]]
    vals = [dist["predict_mean_mae_m"], dist["prior_inverse_square_mae_m"], *cv.values(), *xr.values()]
    fig, ax = col_fig(0.9)
    colors = ["0.6", "0.6"] + ["tab:blue"] * len(cv) + ["tab:red"] * len(xr)
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("distance MAE (m)")
    fig.savefig(out / "distance_cv.pdf")
    plt.close(fig)
    within = [v["cdr_db"] for v in dist["spearman_by_rt60"].values()]
    m("rhoCdrRoomMin", f"{min(within):+.2f}")  # most negative = strongest
    m("rhoCdrRoomMax", f"{max(within):+.2f}")
    for k, v in dist["spearman_log_distance"].items():
        m(_name(f"rho {k}"), f"{v:+.2f}")
    for k, v in cv.items():
        m(_name(f"distcv {k}"), f"{v:.2f}")
    for k, v in xr.items():
        m(_name(f"distxroom {k}"), f"{v:.2f}")
    m("distPredictMean", f"{dist['predict_mean_mae_m']:.2f}")
    m("distPrior", f"{dist['prior_inverse_square_mae_m']:.2f}")
    m("distN", str(len(rows)))


def fig_beam(beam, out, m):
    rows = beam["rows"]
    rts = sorted({r["rt60_s"] for r in rows})
    fig, ax = col_fig(0.7)
    for rt in rts:
        sel = sorted([r for r in rows if r["rt60_s"] == rt], key=lambda r: r["interferer_deg"])
        x = [r["interferer_deg"] for r in sel]
        ax.plot(x, [r["gain_db"]["mvdr"] for r in sel], marker="o", label=f"MVDR, RT60 {rt}s")
        ax.plot(x, [r["gain_db"]["delay_and_sum"] for r in sel], marker="s", ls="--", label=f"DS, RT60 {rt}s")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("interferer azimuth (target at 0 deg)")
    ax.set_ylabel("output SINR gain vs mic 1 (dB)")
    ax.legend(frameon=False)
    fig.savefig(out / "beamform_gain.pdf")
    plt.close(fig)
    g_m = [r["gain_db"]["mvdr"] for r in rows]
    g_d = [r["gain_db"]["delay_and_sum"] for r in rows]
    m("beamMvdrMean", f"{np.mean(g_m):.1f}")
    m("beamMvdrMin", f"{np.min(g_m):.1f}")
    m("beamMvdrMax", f"{np.max(g_m):.1f}")
    m("beamDsMean", f"{np.mean(g_d):.1f}")
    m("beamDoaErrMax", f"{np.max([r['doa_err_deg'] for r in rows]):.1f}")


def pd_ffill(x):
    """Forward/back-fill NaNs of a 1-D array."""
    x = np.array(x, float)
    idx = np.where(np.isfinite(x), np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = x[idx]
    first = np.argmax(np.isfinite(x))
    y[:first] = x[first]
    return y


def fig_radar(demo, out, m):
    tr = demo["track"]
    t = np.array([r["t"] for r in tr])
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_W * 0.5, TEXT_W * 0.45), sharex=True)
    apply_style()
    true_az = np.array([np.nan if r["true_az"] is None else r["true_az"] for r in tr], float)
    true_r = np.array([np.nan if r["true_range"] is None else r["true_range"] for r in tr], float)
    # draw estimates on the branch closest to the truth so 359 vs 1 deg is not a jump
    ref = pd_ffill(true_az)
    unwrap = lambda a: ref + (((np.asarray(a, float) - ref) + 180) % 360 - 180)  # noqa: E731
    axes[0].plot(t, true_az, color="k", lw=2, label="truth")
    axes[0].plot(t, unwrap([r["track_az"] for r in tr]), color="tab:red", lw=1, label="track")
    axes[0].scatter(t, unwrap([r["az_meas"] for r in tr]), s=2, color="tab:orange", label="raw DOA")
    axes[0].set_ylabel("azimuth (deg)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].plot(t, true_r, color="k", lw=2)
    axes[1].plot(t, [r["track_range"] for r in tr], color="tab:red", lw=1)
    axes[1].set_ylabel("distance (m)")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "radar_demo_track.pdf")
    plt.close(fig)
    s = demo["summary"]
    m("demoAzMedErr", f"{s['median_track_az_err_deg']:.1f}")
    m("demoAzPNinetyErr", f"{s['p90_track_az_err_deg']:.1f}")
    m("demoRangeMedErr", f"{s['median_track_range_err_m']:.2f}")
    rb = s.get("relative_bands")
    if rb and rb["per_waypoint"]:
        m("demoBandAcc", f"{100 * rb['frame_accuracy']:.0f}")
        m("demoBandWpCorrect", str(sum(p["pred"] == p["true"] for p in rb["per_waypoint"])))
        m("demoBandWpN", str(len(rb["per_waypoint"])))


def tab_vad_highpass(vh, out, m):
    """Macros only (table typeset in 04_results.tex): DOA MAE / max per cut-off."""
    for r in vh["rows"]:
        k = _name(f"vadhp {r['noise']} rt{int(round(r['rt60_s'] * 10))} hp{r['highpass_hz'] or 'none'}")
        m(k + "Mae", f"{r['mae_deg']:.1f}")
        m(k + "Max", f"{r['max_deg']:.0f}")
        m(k + "Det", str(r["n_detected"]))


def fig_hw(hw, out, m):
    """Hardware session: room-noise spectrum on the raw mics, and the two range cues per utterance."""
    n = hw["noise"]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W * 0.3))
    f, p = np.array(n["psd_hz"]), np.array(n["psd_db"])
    axes[0].plot(f, p, color="0.3", lw=1)
    for fc, ls in ((150, ":"), (200, "--")):
        axes[0].axvline(fc, color="tab:red", ls=ls, lw=1, label=f"VAD high-pass {fc} Hz")
    axes[0].axvspan(85, 255, color="tab:blue", alpha=0.1, label="F0 search band")
    axes[0].set_xlim(0, 1000)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("noise PSD (dB)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("room noise, raw mics (silence)")
    lr = hw.get("live_relative")
    if lr:
        us = [u for u in lr["utterances"] if u["n"] >= 5 and (u["snr_db_med"] or -99) > 3]
        col = {"near": "tab:red", "mid": "0.5", "far": "tab:blue"}
        for b in ("near", "mid", "far"):
            sel = [u for u in us if u["band_speech_db"] == b]
            axes[1].scatter([u["speech_db_med"] for u in sel], [u["cdr_db_med"] for u in sel], s=8,
                            color=col[b], alpha=0.7, label=f"{b} (speech-level band)")
        axes[1].set_xlabel("noise-subtracted speech level (dBFS)")
        axes[1].set_ylabel("CDR (dB)")
        axes[1].legend(frameon=False, fontsize=7)
        axes[1].set_title(f"live session, {len(us)} utterances")
    fig.tight_layout()
    fig.savefig(out / "hw_session.pdf")
    plt.close(fig)

    bf = n["band_power_frac"]
    m("hwNoiseDb", f"{n['broadband_dbfs']:.1f}")
    m("hwNoiseLowPct", f"{100 * (bf['0-60'] + bf['60-100']):.0f}")
    m("hwNoiseHpTwoHundredDb", f"{n['highpass_200_dbfs']:.1f}")
    for r in hw["vad_ablation"]:
        k = "None" if r["highpass_hz"] is None else _name(str(r["highpass_hz"]))
        m(f"hwVadTalk{k}", str(r["speech_in_talk"]))
        m(f"hwVadFalse{k}", str(r["speech_in_silence"]))
    r0 = hw["vad_ablation"][0]
    m("hwVadTalkN", str(r0["talk_frames"]))
    m("hwVadSilN", str(r0["silence_frames"]))
    c = hw["clip"]
    m("hwClipModelRange", f"{c['model_range_m']:.1f}")
    m("hwClipCdr", f"{c['features']['cdr_db']:.1f}")
    m("hwClipSnr", f"{c['features']['snr_db']:.1f}")
    m("hwFwRtSixty", f"{hw['device']['firmware_rt60_s']:.1f}")
    if "live_no_highpass" in hw:
        l0 = hw["live_no_highpass"]
        m("hwLiveZeroSpeech", str(l0["speech_frames"]))
        m("hwLiveZeroFrames", str(l0["frames"]))
        m("hwLiveZeroSec", f"{l0['seconds']:.0f}")
    if lr:
        a = lr["cue_agreement"]
        m("hwLiveSpeechPct", f"{100 * lr['speech_frames'] / lr['frames']:.0f}")
        m("hwLiveMin", f"{lr['seconds'] / 60:.0f}")
        m("hwLiveUtt", str(a["n_utterances_ge5_frames"]))
        m("hwRhoSpeechCdr", f"{a['spearman_speech_vs_cdr']:.2f}")
        m("hwBandAgree", f"{100 * a['band_agreement_speech_vs_cdr']:.0f}")
        m("hwSwitchSpeech", f"{100 * a['switch_rate_speech_db']:.0f}")
        m("hwSwitchCdr", f"{100 * a['switch_rate_cdr_db']:.0f}")
        for cue in ("speech_db", "cdr_db"):
            for b, v in a[f"bands_{cue}"].items():
                m(f"hwBand{_name(cue)}{b.capitalize()}", f"{100 * v:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    macros: list[str] = []

    def m(k, v):
        macros.append(f"\\newcommand{{\\{k}}}{{{v}}}")

    for name, fn in [("doa_sim.json", fig_doa), ("distance_sim.json", fig_distance),
                     ("beamform_sim.json", fig_beam), ("radar_demo.json", fig_radar),
                     ("vad_highpass_sim.json", tab_vad_highpass), ("hw_session.json", fig_hw)]:
        d = _load(R, name)
        if d is None:
            print(f"missing {name}, skipped")
            continue
        fn(d, out, m)
        print(f"figures from {name}")
    hw = _load(R, "hw_calibration.json")
    m("hwAvailable", "yes" if hw else "no")
    if hw:  # labelled grid of session 2 (2026-10-18, recorded with project 19)
        m("hwCalN", str(hw["n_recordings"]))
        m("hwCalOffset", f"{hw['orientation']['azimuth_offset_deg']:.0f}")
        m("hwCalLoo", f"{hw['doa_loo_median_err_deg']:.0f}")
        m("hwCalSpearMax", f"{max(abs(v) for v in hw['spearman_log_distance'].values()):.2f}")
        mae = hw["loo_mae_m"]
        m("hwCalMaeBest", f"{min(v for k, v in mae.items() if k not in ('predict_mean', 'sim_prior')):.2f}")
        m("hwCalMaeMean", f"{mae['predict_mean']:.2f}")
        m("hwCalMaeSim", f"{mae['sim_prior']:.1f}")
    Path(args.macros).write_text("\n".join(macros) + "\n")
    print(f"wrote {args.macros} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
