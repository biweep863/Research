"""Experiment 3 -- the same open-set pipeline on real marine-mammal calls
(Watkins / BEANS).  Hold out U species as unknown, repeat for several random
holdouts.  Writes results/real.json (with available=false if no data)."""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg, project_root
from src.data.watkins import load_watkins
from src.novelty import scores as S
from src.novelty.cluster import ari, cluster_novel
from src.novelty.models import fit_pca_logistic, logits_and_embed, recon_error, train_autoencoder, train_classifier


def main():
    cfg, out_dir = load_cfg()
    rc, ncfg, audio, cc = cfg["real"], cfg["novelty"], cfg["audio"], cfg["cluster"]
    seed = cfg.get("seed", 0)
    data = load_watkins(project_root(), audio, rc["cache_max_per_species"])
    if data is None:
        print("[real] Watkins data NOT available -> skipping")
        dump(out_dir, "real.json", {"available": False})
        return
    X, labels = data
    names, counts = np.unique(labels, return_counts=True)
    keep = names[counts >= rc["min_clips_per_species"]]
    m = np.isin(labels, keep)
    X, labels = X[m], labels[m]
    names = np.unique(labels)
    print(f"[real] {len(X)} clips, {len(names)} species with >= {rc['min_clips_per_species']} clips")

    rows = []
    for d in range(rc["holdout_draws"]):
        rng = np.random.default_rng(seed + 100 + d)
        unk_names = rng.choice(names, rc["n_unknown"], replace=False)
        known_names = np.array([n for n in names if n not in unk_names])
        is_unknown = np.isin(labels, unk_names)
        y = np.array([np.searchsorted(known_names, l) if l in known_names else -1 for l in labels])
        is_test = rng.random(len(y)) < ncfg["test_frac"]
        tr = ~is_test & ~is_unknown
        te = is_test | is_unknown
        te_unk = is_unknown[te]
        c, a = rc.get("cnn", ncfg["cnn"]), ncfg["ae"]
        model = train_classifier(X[tr], y[tr], len(known_names), c["embed"], c["epochs"], c["lr"], c["batch"], seed + d)
        lg, z = logits_and_embed(model, X[te]); _, ztr = logits_and_embed(model, X[tr])
        ae = train_autoencoder(X[tr], a["latent"], a["epochs"], a["lr"], a["batch"], seed + d)
        pipe = fit_pca_logistic(X[tr], y[tr], ncfg["pca_dim"], seed + d)
        p_pca = pipe.predict_proba(X[te].reshape(te.sum(), -1))
        sc = {"msp": S.msp(lg), "energy": S.energy(lg), "knn": S.knn(ztr, z, ncfg["knn_k"]),
              "maha": S.mahalanobis(ztr, y[tr], z), "ae": recon_error(ae, X[te]), "pca_msp": 1 - p_pca.max(1)}
        res = {k: S.metrics(v, te_unk) for k, v in sc.items()}
        res["known_acc_cnn"] = float(np.mean(lg.argmax(1)[~te_unk] == y[te][~te_unk]))
        res["known_acc_pca"] = float(np.mean(pipe.classes_[p_pca.argmax(1)][~te_unk] == y[te][~te_unk]))
        # Clustering of flagged-novel clips (same rule as run_cluster).
        s = sc[cc["score"]]
        thr = np.quantile(s[~te_unk], cc["flag_quantile"])
        flag = s > thr
        lab_true = labels[te][flag]
        if flag.sum() >= 4:
            lab_pred, k_hat, sil = cluster_novel(z[flag], tuple(cc["k_range"]), cc["link"])
            res["cluster"] = {"n_flagged": int(flag.sum()), "frac_flagged_unknown": float(te_unk[flag].mean()),
                              "k_hat": int(k_hat), "k_true": int(len(np.unique(lab_true[te_unk[flag]]))),
                              "ari": ari(lab_true, lab_pred), "silhouette": float(sil)}
        res.update({"draw": d, "unknown_species": unk_names.tolist(), "n_train": int(tr.sum()),
                    "n_test_known": int((~te_unk).sum()), "n_test_unknown": int(te_unk.sum())})
        rows.append(res)
        print(f"[real] draw {d}: acc={res['known_acc_cnn']:.2f}  " + "  ".join(f"{k}={res[k]['auroc']:.2f}" for k in ncfg["scores"])
              + (f"  cluster ARI={res['cluster']['ari']:.2f} k={res['cluster']['k_hat']}/{res['cluster']['k_true']}" if "cluster" in res else ""))
    summary = {k: {"auroc_mean": float(np.mean([r[k]["auroc"] for r in rows])),
                   "auroc_std": float(np.std([r[k]["auroc"] for r in rows]))} for k in ncfg["scores"]}
    dump(out_dir, "real.json", {"available": True, "n_clips": int(len(X)), "n_species": int(len(names)),
                                "species": names.tolist(), "rows": rows, "summary": summary,
                                "known_acc_mean": float(np.mean([r["known_acc_cnn"] for r in rows]))})


if __name__ == "__main__":
    main()
