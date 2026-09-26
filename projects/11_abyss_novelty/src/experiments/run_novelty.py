"""Experiment 2 -- open-set novelty on synthetic deep-sea bioacoustics.

Train on K known species, score held-out known + U unknown species with six
novelty scores; sweep receiver range (SNR), number of known species and
training clips per species.  Writes results/novelty.json."""
from __future__ import annotations

import time
import zlib

import numpy as np

from .common import dump, load_cfg, resolve_device
from src.novelty import scores as S
from src.novelty.models import (fit_pca_logistic, logits_and_embed, recon_error, train_autoencoder,
                                train_classifier)
from src.sim.dataset import make_clips, make_species_set
from utils.stats import bootstrap_ci


def evaluate(cs, n_known, ncfg, seed, test_frac, device="cpu"):
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(cs.y)) < test_frac
    tr = ~is_test & cs.known
    te = is_test | ~cs.known
    te_unk = ~cs.known[te]
    c = ncfg["cnn"]
    model = train_classifier(cs.X[tr], cs.y[tr], n_known, c["embed"], c["epochs"], c["lr"], c["batch"], seed, device)
    lg, z = logits_and_embed(model, cs.X[te], device); _, ztr = logits_and_embed(model, cs.X[tr], device)
    a = ncfg["ae"]
    ae = train_autoencoder(cs.X[tr], a["latent"], a["epochs"], a["lr"], a["batch"], seed, device)
    pipe = fit_pca_logistic(cs.X[tr], cs.y[tr], ncfg["pca_dim"], seed)
    p_pca = pipe.predict_proba(cs.X[te].reshape(te.sum(), -1))
    sc = {"msp": S.msp(lg), "energy": S.energy(lg), "knn": S.knn(ztr, z, ncfg["knn_k"]),
          "maha": S.mahalanobis(ztr, cs.y[tr], z), "ae": recon_error(ae, cs.X[te], device), "pca_msp": 1 - p_pca.max(1)}
    res = {k: S.metrics(v, te_unk) for k, v in sc.items()}
    res["known_acc_cnn"] = float(np.mean(lg.argmax(1)[~te_unk] == cs.y[te][~te_unk]))
    res["known_acc_pca"] = float(np.mean(pipe.classes_[p_pca.argmax(1)][~te_unk] == cs.y[te][~te_unk]))
    res["snr_mean_db"] = float(cs.snr.mean())
    res["n_train"] = int(tr.sum()); res["n_test_known"] = int((~te_unk).sum()); res["n_test_unknown"] = int(te_unk.sum())
    # bootstrap CI on the best score's AUROC
    best = max(ncfg["scores"], key=lambda k: res[k]["auroc"])
    boots = []
    brng = np.random.default_rng(seed + 1)
    for _ in range(ncfg["n_boot"]):
        idx = brng.integers(0, te.sum(), te.sum())
        if te_unk[idx].min() == te_unk[idx].max():
            continue
        boots.append(S.metrics(sc[best][idx], te_unk[idx])["auroc"])
    res["best_score"] = best
    res["best_auroc_ci"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else [None, None]
    return res, {k: v for k, v in sc.items()}, te_unk, z, cs.y[te]


def main():
    cfg, out_dir = load_cfg()
    ncfg, sim, audio = cfg["novelty"], cfg["sim"], cfg["audio"]
    seed = cfg.get("seed", 0)
    device = resolve_device(cfg)
    rows = []

    def run(tag, n_known, n_unknown, clips, range_km):
        # crc32 instead of hash(): Python randomizes str hash per process, which
        # made synthetic species draws non-reproducible across runs (doc/03).
        rng = np.random.default_rng(seed + zlib.crc32(tag.encode()) % 1000)
        sp = make_species_set(n_known, n_unknown, rng, sim["families"], tuple(sim["sl_db"]))
        cs = make_clips(sp, n_known, clips, audio, sim, rng, range_km=range_km)
        t0 = time.time()
        res, _, _, _, _ = evaluate(cs, n_known, ncfg, seed, ncfg["test_frac"], device)
        res.update({"tag": tag, "n_known": n_known, "n_unknown": n_unknown, "clips_per_species": clips,
                    "range_km": list(range_km), "time_s": time.time() - t0})
        rows.append(res)
        print(f"[novelty] {tag:28s} snr={res['snr_mean_db']:5.1f} dB  acc={res['known_acc_cnn']:.2f}  "
              + "  ".join(f"{k}={res[k]['auroc']:.2f}" for k in ncfg["scores"]))

    run("base", sim["n_known"], sim["n_unknown"], sim["clips_per_species"], sim["range_km"])
    for rmax in ncfg["sweeps"]["range_km_max"]:
        run(f"range_max_{rmax:g}", sim["n_known"], sim["n_unknown"], sim["clips_per_species"], [sim["range_km"][0], rmax])
    for k in ncfg["sweeps"]["n_known"]:
        run(f"n_known_{k}", k, sim["n_unknown"], sim["clips_per_species"], sim["range_km"])
    for c in ncfg["sweeps"]["clips_per_species"]:
        run(f"clips_{c}", sim["n_known"], sim["n_unknown"], c, sim["range_km"])
    dump(out_dir, "novelty.json", {"scores": ncfg["scores"], "rows": rows})


if __name__ == "__main__":
    main()
