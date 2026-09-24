"""Experiment 5 -- Chatbot Arena human votes.

Model-level comparison graph (pairs with >= min_votes votes).  (a) Hodge split
of the real log-odds flow, with a bootstrap over votes for confidence
intervals and a Bradley-Terry-simulated null (same graph, same counts, a
perfectly transitive truth) to separate genuine cycles from sampling noise.
(b) Vote-level train/held-out split per edge; items have learned embeddings
(one-hot input to the networks); train scalar, antisymmetric and GPM models
with the Bradley-Terry loss, log per-component fit of the training flow, and
held-out log-loss / accuracy (ties excluded from accuracy).  (c) The same
with a fraction of training votes flipped.  Writes results/arena.json.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import dump, load_cfg, project_root
from .shared import log_steps, summarize_fit
from src.data.arena import edge_flow, load_arena
from src.hodge.decompose import COMPONENTS, dims, energy_fractions, hodgerank, projectors
from src.hodge.graph import from_edges
from src.kernels.empirical import empirical_edge_ntk
from src.kernels.edge_ntk import hodge_rates
from src.models.nets import build
from src.train import fit


def main():
    cfg, out = load_cfg()
    ac, seed = cfg["arena"], cfg.get("seed", 0)
    A = load_arena(project_root())
    if A is None:
        print("[arena] data not available -> skipped")
        dump(out, "arena.json", {"available": False}); return
    names, vi, vj, vo = A["names"], A["i"], A["j"], A["outcome"]
    n = len(names)
    edges, y, tot, wins = edge_flow(vi, vj, vo, n, ac["min_votes"])
    g = from_edges(n, edges.tolist())
    # Keep only items that appear in the kept graph.
    used = np.unique(edges)
    remap = -np.ones(n, int); remap[used] = np.arange(len(used))
    edges = remap[edges]; n2 = len(used)
    g = from_edges(n2, edges.tolist())
    order = {tuple(e): k for k, e in enumerate(g.edges.tolist())}
    perm = np.array([order[tuple(e)] for e in edges.tolist()])
    y_g = np.zeros(g.n_edges); y_g[perm] = y
    tot_g = np.zeros(g.n_edges); tot_g[perm] = tot
    P = projectors(g); dm = dims(g, P)
    fr = energy_fractions(y_g, P)
    print(f"[arena] items={n2} edges={g.n_edges} votes={int(tot_g.sum())} dims={dm}  fractions={ {k: round(v, 3) for k, v in fr.items()} }")

    # Bootstrap over votes, and a transitive null with the same counts.
    rng = np.random.default_rng(seed)
    s_hr = hodgerank(g, y_g, w=tot_g)
    p_null = 1 / (1 + np.exp(-(s_hr[g.edges[:, 1]] - s_hr[g.edges[:, 0]])))
    boots, nulls = [], []
    for b in range(ac["bootstrap"]):
        wb = rng.binomial(tot_g.astype(int), np.clip(np.exp(y_g) / (1 + np.exp(y_g)), 1e-6, 1 - 1e-6))
        yb = np.log((wb + 0.5) / (tot_g - wb + 0.5)); boots.append(energy_fractions(yb, P))
        wn = rng.binomial(tot_g.astype(int), p_null)
        yn = np.log((wn + 0.5) / (tot_g - wn + 0.5)); nulls.append(energy_fractions(yn, P))
    # Note: parametric resampling adds a second layer of sampling noise, so the
    # bootstrap distribution is shifted toward cycles; it is reported for
    # spread only.  The test of genuine intransitivity is the transitive null.
    ci = {c: [float(np.quantile([x[c] for x in boots], q)) for q in (0.025, 0.975)] for c in COMPONENTS}
    p_null_cyc = float(np.mean([1 - x["grad"] >= 1 - fr["grad"] for x in nulls]))
    null = {c: [float(np.quantile([x[c] for x in nulls], q)) for q in (0.025, 0.5, 0.975)] for c in COMPONENTS}
    print(f"[arena] cyclic fraction {1 - fr['grad']:.3f}  bootstrap CI grad {ci['grad']}  null (transitive) grad median {null['grad'][1]:.3f}")

    # Vote-level split for training / held-out evaluation.
    kept_pairs = {tuple(used[e]) for e in g.edges.tolist()}
    keep = np.array([(a, b) in kept_pairs for a, b in zip(vi.tolist(), vj.tolist())])
    vi2, vj2, vo2 = remap[vi[keep]], remap[vj[keep]], vo[keep]
    split = np.random.default_rng(seed + 1).random(len(vi2)) < ac["heldout_frac"]
    E = g.n_edges; eidx = {tuple(e): k for k, e in enumerate(g.edges.tolist())}
    ev = np.array([eidx[(a, b)] for a, b in zip(vi2.tolist(), vj2.tolist())])
    tr_w = np.bincount(ev[~split], weights=vo2[~split], minlength=E); tr_n = np.bincount(ev[~split], minlength=E).astype(float)
    y_tr = np.log((tr_w + 0.5) / (tr_n - tr_w + 0.5))
    ho_mask = split & (vo2 != 0.5)
    X = np.eye(n2)
    i, j = g.edges[:, 0], g.edges[:, 1]
    steps = log_steps(ac["steps"], ac["n_log"])
    runs = []
    for s in range(ac["seeds"]):
        for flip in ac["flip_grid"]:
            rng = np.random.default_rng(seed + s + int(100 * flip))
            w_f = tr_w.copy()
            if flip > 0:
                w_int = np.round(tr_w).astype(int); n_int = tr_n.astype(int)
                fw = rng.binomial(w_int, flip); fl = rng.binomial(n_int - w_int, flip)
                w_f = (w_int - fw + fl).astype(float)
            for kind in ac["models"]:
                m = build(kind, n2, ac["width"], 1, seed=seed + s, **({"k": 4} if kind == "gpm" else {}))
                K = empirical_edge_ntk(m, torch.tensor(X[i], dtype=torch.float32), torch.tensor(X[j], dtype=torch.float32))
                rates = hodge_rates(K, P)
                eta = cfg["dynamics"]["lr_frac"] * E / float(np.linalg.eigvalsh(K).max())
                log = fit(m, X[i], X[j], P, y_tr, loss="bt", wins=w_f, tot=tr_n, lr=4 * eta,
                          steps=ac["steps"], log_steps=steps,
                          heldout=(X[vi2[ho_mask]], X[vj2[ho_mask]], vo2[ho_mask] - 0.5))
                with torch.no_grad():
                    gh = m(torch.tensor(X[vi2[ho_mask]], dtype=torch.float32), torch.tensor(X[vj2[ho_mask]], dtype=torch.float32)).double().numpy()
                p = 1 / (1 + np.exp(-gh)); o = vo2[ho_mask]
                ll = float(-np.mean(o * np.log(p + 1e-12) + (1 - o) * np.log(1 - p + 1e-12)))
                acc = float(np.mean((p > 0.5) == (o > 0.5)))
                acc_curve = np.array(log["ho_acc"]); kb = int(np.argmax(acc_curve))
                runs.append({"seed": s, "flip": flip, "model": kind, "rates": rates,
                             "acc_best": float(acc_curve[kb]), "step_best": int(log["step"][kb]), "acc_curve": log["ho_acc"],
                             "ratio": rates["grad"] / max(np.nanmean([rates["curl"], rates["harm"]] if dm["harm"] else [rates["curl"]]), 1e-12),
                             "crossings": summarize_fit(log), "heldout_logloss": ll, "heldout_acc": acc,
                             "fit_log": {k: log[k] for k in ("step", "fit_grad", "fit_curl", "fit_harm")}})
                cr = runs[-1]["crossings"]
                print(f"[arena] seed={s} flip={flip:.1f} {kind:7s} ratio={runs[-1]['ratio']:.1f} t90 grad={cr['grad']['0.9']} "
                      f"curl={cr['curl']['0.9']}  held-out logloss={ll:.4f} acc={acc:.3f}")
    # Baseline: HodgeRank scores on the training flow (weighted), scored as BT.
    s_tr = hodgerank(g, y_tr, w=tr_n)
    gh = s_tr[vj2[ho_mask]] - s_tr[vi2[ho_mask]]; p = 1 / (1 + np.exp(-gh)); o = vo2[ho_mask]
    hr_ll = float(-np.mean(o * np.log(p + 1e-12) + (1 - o) * np.log(1 - p + 1e-12))); hr_acc = float(np.mean((p > 0.5) == (o > 0.5)))
    top = [names[used[k]] for k in np.argsort(-s_hr)[:5]]
    dump(out, "arena.json", {"available": True, "items": int(n2), "edges": int(E), "votes": int(tot_g.sum()),
                             "dims": dm, "fractions": fr, "bootstrap_ci": ci, "null": null, "p_null_cyclic_ge_observed": p_null_cyc, "top5": top,
                             "n_heldout": int(ho_mask.sum()), "hodgerank_heldout": {"logloss": hr_ll, "acc": hr_acc},
                             "runs": runs, "steps": steps.tolist(), "min_votes": ac["min_votes"]})


if __name__ == "__main__":
    main()
