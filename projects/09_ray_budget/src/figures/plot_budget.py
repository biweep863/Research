"""Figures + LaTeX macros for the ray-budget paper.

Reads results/{accel,lod,sampling,pareto}.json and re-renders the canonical
scene for the beauty/cost-map figure; writes PDFs to paper/figures and every
quoted number to paper/macros.tex, so recomputing the figures rewrites the prose.
"""
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

from utils.plotting import col_fig, text_fig, apply_style
from src.render.scene import build_scene, bake_field
from src.render.bvh import BVH
from src.render.tracer import render, cost_map

CLASS_COLOR = {
    "lossless": "#2e7d32", "sampling": "#1565c0", "denoise": "#00838f",
    "lod": "#ef6c00", "stack": "#c62828",
}


def _grp(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()

    res = Path(args.results)
    accel = json.loads((res / "accel.json").read_text())
    lod = json.loads((res / "lod.json").read_text())
    sampling = json.loads((res / "sampling.json").read_text())
    pareto = json.loads((res / "pareto.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    macros: list[tuple[str, str]] = []
    m = macros.append

    # ------------------------------------------------------------------ #
    # Fig 1: beauty render + per-pixel intersection-cost heatmap
    # ------------------------------------------------------------------ #
    w, h = pareto["width"], pareto["height"]
    rng = np.random.default_rng(0)
    scene = build_scene(n_blades=pareto["blades"], n_spheres=3,
                        width=w, height=h, rng=rng)
    bvh = BVH(bake_field(scene, scene.modes), 4)
    img, _ = render(scene, bvh, spp=12, seed=1)
    cmap = cost_map(scene, bvh)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    a1.imshow(np.clip(img, 0, 1)); a1.set_title("The scene (path traced)")
    a1.axis("off")
    im = a2.imshow(cmap, cmap="inferno")
    a2.set_title("Intersection tests / pixel"); a2.axis("off")
    fig.colorbar(im, ax=a2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "scene_cost.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: acceleration scaling (tests + memory vs scene size)
    # ------------------------------------------------------------------ #
    rows = accel["rows"]
    blades = [r["blades"] for r in rows]
    brute = [r["brute"]["tests"] for r in rows]
    bvhc = [r["bvh"]["tests"] for r in rows]
    inst = [r["instanced"]["tests"] for r in rows]
    mem_dup = [r["mem_duplicated_floats"] for r in rows]
    mem_ins = [r["mem_instanced_floats"] for r in rows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.6))
    a1.loglog(blades, brute, "o-", color="#b71c1c", label="brute force")
    a1.loglog(blades, bvhc, "s-", color="#2e7d32", label="BVH")
    a1.loglog(blades, inst, "^-", color="#1565c0", label="BVH + instancing")
    a1.set_xlabel("grass blades"); a1.set_ylabel("intersection tests / frame")
    a1.set_title("Cost of one frame vs scene size"); a1.legend()
    a2.loglog(blades, mem_dup, "o-", color="#b71c1c", label="duplicated")
    a2.loglog(blades, mem_ins, "^-", color="#1565c0", label="instanced")
    a2.set_xlabel("grass blades"); a2.set_ylabel("scene floats")
    a2.set_title("Geometry memory"); a2.legend()
    fig.tight_layout()
    fig.savefig(out / "accel_scaling.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: LOD -- quality and cost vs distance threshold
    # ------------------------------------------------------------------ #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.6))
    for mode, mk, cl in [("simplified", "o-", "#ef6c00"),
                         ("billboard", "s-", "#6a1b9a")]:
        r = lod["modes"][mode]
        thr = [x["threshold"] for x in r]
        a1.plot(thr, [x["ssim"] for x in r], mk, color=cl, label=mode)
        a2.plot(thr, [x["cost_frac"] for x in r], mk, color=cl, label=mode)
    a1.set_xlabel("full-detail fraction (threshold)"); a1.set_ylabel("SSIM vs full")
    a1.set_title("LOD quality price"); a1.legend(); a1.invert_xaxis()
    a2.set_xlabel("full-detail fraction (threshold)"); a2.set_ylabel("cost / full")
    a2.set_title("LOD cost saving"); a2.legend(); a2.invert_xaxis()
    fig.tight_layout()
    fig.savefig(out / "lod_curve.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: sampling convergence + denoiser
    # ------------------------------------------------------------------ #
    sr = sampling["rows"]
    spps = [x["spp"] for x in sr]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.6))
    a1.loglog(spps, [x["rmse"] for x in sr], "o-", color="#1565c0", label="RMSE")
    ref_line = sr[0]["rmse"] * (np.array(spps) / spps[0]) ** -0.5
    a1.loglog(spps, ref_line, "--", color="grey", label="slope $-1/2$")
    a1.set_xlabel("samples / pixel"); a1.set_ylabel("RMSE vs reference")
    a1.set_title(f"Monte-Carlo noise (fit {sampling['mc_slope']:.2f})"); a1.legend()
    a2.semilogx(spps, [x["ssim"] for x in sr], "o-", color="#1565c0", label="raw")
    a2.semilogx(spps, [x["ssim_denoised"] for x in sr], "s-", color="#00838f",
                label="denoised")
    a2.set_xlabel("samples / pixel"); a2.set_ylabel("SSIM vs reference")
    a2.set_title("Denoiser buys quality"); a2.legend()
    fig.tight_layout()
    fig.savefig(out / "sampling.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: the cost-quality Pareto frontier
    # ------------------------------------------------------------------ #
    pts = pareto["points"]
    fig, ax = text_fig(height_ratio=0.5)
    seen = set()
    for p in pts:
        cl = p["class"]
        ax.scatter(p["rel_cost"], p["ssim"], color=CLASS_COLOR[cl], s=42,
                   zorder=3, label=cl if cl not in seen else None,
                   edgecolors="white", linewidths=0.6)
        seen.add(cl)
    stack = next(p for p in pts if p["class"] == "stack")
    ax.annotate("full stack", (stack["rel_cost"], stack["ssim"]),
                textcoords="offset points", xytext=(8, -10), fontsize=7,
                color="#c62828")
    ax.set_xscale("log")
    ax.set_xlabel("relative cost (brute force @ full quality = 1)")
    ax.set_ylabel("SSIM vs reference")
    ax.set_title("Cost-quality frontier of ray-tracing shortcuts")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "pareto.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    big = accel["rows"][-1]
    acc_speed = big["brute"]["tests"] / big["bvh"]["tests"]
    acc_speed_inst = big["brute"]["tests"] / big["instanced"]["tests"]
    mem_ratio = big["mem_duplicated_floats"] / big["mem_instanced_floats"]
    samp = {x["spp"]: x for x in sr}
    s1 = samp[min(samp)]
    dn_gain = 100 * (s1["ssim_denoised"] - s1["ssim"])
    lod_simp = lod["modes"]["simplified"]
    lod_bill = lod["modes"]["billboard"]
    lod_worst = min(lod_bill, key=lambda x: x["ssim"])
    stack = next(p for p in pts if p["class"] == "stack")
    bvh_pt = next(p for p in pts if p["label"] == "BVH")

    m(("SceneBlades", _grp(pareto["blades"])))
    m(("SceneTris", _grp(accel["rows"][2]["triangles"])))  # 200-blade row
    m(("BigBlades", _grp(big["blades"])))
    m(("BigTris", _grp(big["triangles"])))
    m(("BruteTestsBig", _grp(big["brute"]["tests"])))
    m(("BvhTestsBig", _grp(big["bvh"]["tests"])))
    m(("AccelSpeedup", f"{acc_speed:.0f}"))
    m(("AccelSpeedupInst", f"{acc_speed_inst:.0f}"))
    m(("MemRatio", f"{mem_ratio:.0f}"))
    m(("BvhSpeedupRender", f"{pareto['lossless_speedup_bvh']:.0f}"))
    m(("InstSpeedupRender", f"{pareto['lossless_speedup_instanced']:.0f}"))
    m(("McSlope", f"{sampling['mc_slope']:.2f}"))
    m(("RefSpp", f"{pareto['reference_spp']}"))
    m(("SsimOneSpp", f"{s1['ssim']:.3f}"))
    m(("SsimOneSppDn", f"{s1['ssim_denoised']:.3f}"))
    m(("DenoiseGainOne", f"{dn_gain:.1f}"))
    m(("LodSimpFloor", f"{min(x['ssim'] for x in lod_simp):.3f}"))
    m(("LodBillFloor", f"{lod_worst['ssim']:.3f}"))
    m(("LodCostFloor", f"{100*min(x['cost_frac'] for x in lod_simp):.0f}"))
    m(("StackSsim", f"{stack['ssim']:.3f}"))
    m(("StackSpeedup", f"{1.0/stack['rel_cost']:.0f}"))
    m(("BvhRelCostPct", f"{100*bvh_pt['rel_cost']:.1f}"))

    Path(args.macros).write_text(
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros) + "\n")
    print(f"[figures] wrote {len(macros)} macros and 5 figures")


if __name__ == "__main__":
    main()
