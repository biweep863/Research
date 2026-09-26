"""Experiment 3 -- the same open-set pipeline on real marine-mammal calls
(Watkins / BEANS).  Hold out U species as unknown, repeat for several random
holdouts.  Writes results/real.json (with available=false if no data), and
results/real_pretrained.json alongside it when real.backbone is configured
(both the CNN and the backbone paths run on the SAME clips/splits per draw,
so the AUROCs are directly comparable -- see doc/05_decisions.md)."""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg, project_root, resolve_device
from src.data.watkins import load_watkins
from src.novelty import scores as S
from src.novelty.cluster import ari, cluster_novel
from src.novelty.models import (fit_pca_logistic, fit_probe, logits_and_embed, recon_error,
                                train_autoencoder, train_classifier)


def _cnn_scores(X, y, tr, te, te_unk, ncfg, rc, seed, d, device):
    c, a = rc.get("cnn", ncfg["cnn"]), ncfg["ae"]
    model = train_classifier(X[tr], y[tr], len(np.unique(y[tr])), c["embed"], c["epochs"], c["lr"],
                             c["batch"], seed + d, device)
    lg, z = logits_and_embed(model, X[te], device); _, ztr = logits_and_embed(model, X[tr], device)
    ae = train_autoencoder(X[tr], a["latent"], a["epochs"], a["lr"], a["batch"], seed + d, device)
    pipe = fit_pca_logistic(X[tr], y[tr], ncfg["pca_dim"], seed + d)
    p_pca = pipe.predict_proba(X[te].reshape(te.sum(), -1))
    sc = {"msp": S.msp(lg), "energy": S.energy(lg), "knn": S.knn(ztr, z, ncfg["knn_k"]),
          "maha": S.mahalanobis(ztr, y[tr], z), "ae": recon_error(ae, X[te], device), "pca_msp": 1 - p_pca.max(1)}
    res = {k: S.metrics(v, te_unk) for k, v in sc.items()}
    res["known_acc_cnn"] = float(np.mean(lg.argmax(1)[~te_unk] == y[te][~te_unk]))
    res["known_acc_pca"] = float(np.mean(pipe.classes_[p_pca.argmax(1)][~te_unk] == y[te][~te_unk]))
    return res, sc, z


def _backbone_scores(E, y, tr, te, te_unk, ncfg, rc, seed, d, cnn_sc):
    pca_dim = rc.get("probe", {}).get("pca_dim", 128)
    C = rc.get("probe", {}).get("C", 1.0)
    pipe = fit_probe(E[tr], y[tr], pca_dim, C, seed + d)
    lg = pipe.decision_function(E[te])
    if lg.ndim == 1:  # binary edge case
        lg = np.stack([-lg, lg], axis=1)
    z = pipe[:-1].transform(E[te])   # PCA-reduced embedding (post-scaler)
    ztr = pipe[:-1].transform(E[tr])
    sc = {"msp": S.msp(lg), "energy": S.energy(lg), "knn": S.knn(ztr, z, ncfg["knn_k"]),
          "maha": S.mahalanobis(ztr, y[tr], z),
          # ae/pca_msp don't depend on the backbone -- reuse the mel-CNN-path values (doc/05_decisions.md)
          "ae": cnn_sc["ae"], "pca_msp": cnn_sc["pca_msp"]}
    res = {k: S.metrics(v, te_unk) for k, v in sc.items()}
    res["known_acc_cnn"] = float(np.mean(lg.argmax(1)[~te_unk] == y[te][~te_unk]))
    res["known_acc_pca"] = res["known_acc_cnn"]
    return res, sc, z


def _cluster_block(sc, cc, labels_te, te_unk, z):
    s = sc[cc["score"]]
    thr = np.quantile(s[~te_unk], cc["flag_quantile"])
    flag = s > thr
    lab_true = labels_te[flag]
    if flag.sum() < 4:
        return None
    lab_pred, k_hat, sil = cluster_novel(z[flag], tuple(cc["k_range"]), cc["link"])
    return {"n_flagged": int(flag.sum()), "frac_flagged_unknown": float(te_unk[flag].mean()),
            "k_hat": int(k_hat), "k_true": int(len(np.unique(lab_true[te_unk[flag]]))),
            "ari": ari(lab_true, lab_pred), "silhouette": float(sil)}


def main():
    cfg, out_dir = load_cfg()
    rc, ncfg, audio, cc = cfg["real"], cfg["novelty"], cfg["audio"], cfg["cluster"]
    seed = cfg.get("seed", 0)
    device = resolve_device(cfg)
    limit_draws = cfg.get("_limit")

    data = load_watkins(project_root(), audio, rc["cache_max_per_species"], rc.get("cache", "watkins_mel.npz"))
    if data is None:
        print("[real] Watkins data NOT available -> skipping")
        dump(out_dir, "real.json", {"available": False})
        return
    X, labels, keys = data
    names, counts = np.unique(labels, return_counts=True)
    keep = names[counts >= rc["min_clips_per_species"]]
    m = np.isin(labels, keep)
    X, labels = X[m], labels[m]
    keys = keys[m] if keys is not None else None
    names = np.unique(labels)
    print(f"[real] {len(X)} clips, {len(names)} species with >= {rc['min_clips_per_species']} clips")

    backbone_id = rc.get("backbone", "none")
    E, E_labels, E_keys = None, None, None
    if backbone_id not in ("none", None):
        from src.novelty.backbone import load_or_build_embeddings
        loaded = load_or_build_embeddings(project_root(), rc, device)
        if loaded is None:
            print(f"[real] backbone={backbone_id} requested but raw waveform data is unavailable -> CNN-only")
            backbone_id = "none"
        else:
            E_full, E_labels_full, E_keys_full = loaded
            key_to_idx = {k: i for i, k in enumerate(E_keys_full)}
            order = [key_to_idx[k] for k in keys if k in key_to_idx]
            mask = np.array([k in key_to_idx for k in keys])
            if mask.sum() < len(keys):
                print(f"[real] {len(keys) - mask.sum()} mel clips have no matching embedding; dropping from both paths")
                X, labels, keys = X[mask], labels[mask], keys[mask]
            E = E_full[order]
            assert len(E) == len(X)

    n_draws = rc["holdout_draws"] if limit_draws is None else min(limit_draws, rc["holdout_draws"])
    rows, rows_pt = [], []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + 100 + d)
        unk_names = rng.choice(names, rc["n_unknown"], replace=False)
        known_names = np.array([n for n in names if n not in unk_names])
        is_unknown = np.isin(labels, unk_names)
        y = np.array([np.searchsorted(known_names, l) if l in known_names else -1 for l in labels])
        is_test = rng.random(len(y)) < ncfg["test_frac"]
        tr = ~is_test & ~is_unknown
        te = is_test | is_unknown
        te_unk = is_unknown[te]

        res, sc, z = _cnn_scores(X, y, tr, te, te_unk, ncfg, rc, seed, d, device)
        if flag_block := _cluster_block(sc, cc, labels[te], te_unk, z):
            res["cluster"] = flag_block
        res.update({"draw": d, "unknown_species": unk_names.tolist(), "n_train": int(tr.sum()),
                    "n_test_known": int((~te_unk).sum()), "n_test_unknown": int(te_unk.sum()),
                    "backbone": "none"})
        rows.append(res)
        line = f"[real] draw {d}: acc={res['known_acc_cnn']:.2f}  " + "  ".join(f"{k}={res[k]['auroc']:.2f}" for k in ncfg["scores"])
        if "cluster" in res:
            line += f"  cluster ARI={res['cluster']['ari']:.2f} k={res['cluster']['k_hat']}/{res['cluster']['k_true']}"
        print(line)

        if E is not None:
            res_pt, sc_pt, z_pt = _backbone_scores(E, y, tr, te, te_unk, ncfg, rc, seed, d, sc)
            if flag_block_pt := _cluster_block(sc_pt, cc, labels[te], te_unk, z_pt):
                res_pt["cluster"] = flag_block_pt
            best = max(ncfg["scores"], key=lambda k: res[k]["auroc"])
            res_pt.update({"draw": d, "unknown_species": unk_names.tolist(), "n_train": int(tr.sum()),
                          "n_test_known": int((~te_unk).sum()), "n_test_unknown": int(te_unk.sum()),
                          "backbone": backbone_id,
                          "paired_delta": float(res_pt[best]["auroc"] - res[best]["auroc"])})
            rows_pt.append(res_pt)
            print(f"[real-pretrained] draw {d}: acc={res_pt['known_acc_cnn']:.2f}  "
                  + "  ".join(f"{k}={res_pt[k]['auroc']:.2f}" for k in ncfg["scores"])
                  + f"  paired_delta={res_pt['paired_delta']:+.3f}")

    summary = {k: {"auroc_mean": float(np.mean([r[k]["auroc"] for r in rows])),
                   "auroc_std": float(np.std([r[k]["auroc"] for r in rows]))} for k in ncfg["scores"]}
    dump(out_dir, "real.json", {"available": True, "n_clips": int(len(X)), "n_species": int(len(names)),
                                "n_unknown": rc["n_unknown"], "species": names.tolist(), "rows": rows,
                                "summary": summary, "backbone": "none",
                                "known_acc_mean": float(np.mean([r["known_acc_cnn"] for r in rows]))})

    if rows_pt:
        summary_pt = {k: {"auroc_mean": float(np.mean([r[k]["auroc"] for r in rows_pt])),
                          "auroc_std": float(np.std([r[k]["auroc"] for r in rows_pt]))} for k in ncfg["scores"]}
        deltas = [r["paired_delta"] for r in rows_pt]
        dump(out_dir, "real_pretrained.json", {
            "available": True, "n_clips": int(len(X)), "n_species": int(len(names)),
            "n_unknown": rc["n_unknown"], "species": names.tolist(), "rows": rows_pt,
            "summary": summary_pt, "backbone": backbone_id,
            "known_acc_mean": float(np.mean([r["known_acc_cnn"] for r in rows_pt])),
            "paired_delta_mean": float(np.mean(deltas)), "paired_delta_std": float(np.std(deltas))})


if __name__ == "__main__":
    main()
