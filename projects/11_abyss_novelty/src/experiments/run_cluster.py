"""Experiment 4 -- from "novel" to "how many new species?" on synthetic data.

Flag test clips whose novelty score exceeds the 95th percentile of known
scores, cluster the flagged embeddings, and compare with the true unknown
species: ARI and estimated vs true number of new species, over several U.
Writes results/cluster.json."""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg, resolve_device
from .run_novelty import evaluate
from src.novelty.cluster import ari, cluster_novel
from src.sim.dataset import make_clips, make_species_set


def main():
    cfg, out_dir = load_cfg()
    ncfg, sim, audio, cc = cfg["novelty"], cfg["sim"], cfg["audio"], cfg["cluster"]
    seed = cfg.get("seed", 0)
    device = resolve_device(cfg)
    rows = []
    for n_unknown in (2, 3, 5, 8):
        for rep in range(1):
            rng = np.random.default_rng(seed + 10 * n_unknown + rep)
            sp = make_species_set(sim["n_known"], n_unknown, rng, sim["families"], tuple(sim["sl_db"]))
            cs = make_clips(sp, sim["n_known"], sim["clips_per_species"], audio, sim, rng)
            res, sc, te_unk, z, y_te = evaluate(cs, sim["n_known"], ncfg, seed + rep, ncfg["test_frac"], device)
            s = sc[cc["score"]]
            thr = np.quantile(s[~te_unk], cc["flag_quantile"])
            flag = s > thr
            lab_pred, k_hat, sil = cluster_novel(z[flag], tuple(cc["k_range"]), cc["link"])
            truth = y_te[flag]
            rows.append({"n_unknown": n_unknown, "rep": rep, "n_flagged": int(flag.sum()),
                         "recall_unknown": float(flag[te_unk].mean()), "precision_unknown": float(te_unk[flag].mean()),
                         "k_hat": int(k_hat), "k_true": int(len(np.unique(truth[te_unk[flag]]))), "ari": ari(truth, lab_pred),
                         "ari_unknown_only": ari(truth[te_unk[flag]], lab_pred[te_unk[flag]]) if te_unk[flag].sum() > 1 else None,
                         "silhouette": float(sil), "auroc_score": res[cc["score"]]["auroc"]})
            print(f"[cluster] U={n_unknown} rep={rep}: flagged {flag.sum()} (recall {rows[-1]['recall_unknown']:.2f}, "
                  f"precision {rows[-1]['precision_unknown']:.2f})  k_hat={k_hat} k_true={rows[-1]['k_true']}  ARI={rows[-1]['ari']:.2f}")
    dump(out_dir, "cluster.json", {"score": cc["score"], "flag_quantile": cc["flag_quantile"], "rows": rows})


if __name__ == "__main__":
    main()
