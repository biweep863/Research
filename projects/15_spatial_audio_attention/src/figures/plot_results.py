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
                     ("beamform_sim.json", fig_beam), ("radar_demo.json", fig_radar)]:
        d = _load(R, name)
        if d is None:
            print(f"missing {name}, skipped")
            continue
        fn(d, out, m)
        print(f"figures from {name}")
    hw = _load(R, "hw_calibration.json")
    m("hwAvailable", "yes" if hw else "no")
    Path(args.macros).write_text("\n".join(macros) + "\n")
    print(f"wrote {args.macros} ({len(macros)} macros)")


if __name__ == "__main__":
    main()
