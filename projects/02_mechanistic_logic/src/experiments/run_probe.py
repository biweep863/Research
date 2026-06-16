"""Train per-layer linear probes for logical validity, on both English and conlang.

This is the cheap, language-independent test of where in the network the
"valid vs. invalid" distinction is linearly decodable. A high probe accuracy
that transfers across English and conlang surface forms is evidence for a
language-independent logical representation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import yaml
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from utils.seeds import set_seed
from utils.model_loader import load_hf
from utils.hooks import hidden_states
from src.data.synth_logic import generate


def collect(model, tok, device, items, max_length=128):
    all_hs = None
    y = []
    surface = []
    for it in tqdm(items, desc="forward"):
        hs, _ = hidden_states(model, tok, it.prompt, max_length=max_length, device=device)
        # take last-token activation in each layer
        last = np.stack([h[0, -1].numpy() for h in hs], axis=0)  # (L+1, d)
        if all_hs is None:
            all_hs = np.empty((len(items), last.shape[0], last.shape[1]), dtype=np.float32)
        all_hs[len(y)] = last
        y.append(1 if it.label == "valid" else 0)
        surface.append(it.surface)
    return all_hs, np.array(y), np.array(surface)


def probe_per_layer(X, y, cv=5, seed=0):
    accs = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    for l in range(X.shape[1]):
        Xl = X[:, l, :]
        fold_accs = []
        for tr, te in skf.split(Xl, y):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xl[tr], y[tr])
            fold_accs.append(clf.score(Xl[te], y[te]))
        accs.append(float(np.mean(fold_accs)))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    items = generate(cfg, seed=cfg.get("seed", 0))
    print(f"[probe] generated {len(items)} items")

    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)

    for m in cfg["models"]:
        lm = load_hf(m["name"], device=cfg.get("device", "auto"), dtype=cfg.get("dtype"))
        X, y, surf = collect(lm.model, lm.tokenizer, lm.device, items)
        results = {"layers": int(X.shape[1])}
        for s in sorted(set(surf.tolist())):
            mask = surf == s
            results[f"probe_{s}"] = probe_per_layer(X[mask], y[mask], cv=cfg["probe"]["cv_folds"], seed=cfg.get("seed", 0))
        # cross-surface generalisation: train English, test conlang and vice-versa
        en = surf == "english"; cl = surf == "conlang"
        cross_en2cl, cross_cl2en = [], []
        for l in range(X.shape[1]):
            clf = LogisticRegression(max_iter=2000).fit(X[en, l], y[en])
            cross_en2cl.append(float(clf.score(X[cl, l], y[cl])))
            clf = LogisticRegression(max_iter=2000).fit(X[cl, l], y[cl])
            cross_cl2en.append(float(clf.score(X[en, l], y[en])))
        results["cross_en2cl"] = cross_en2cl
        results["cross_cl2en"] = cross_cl2en
        out = out_root / f"probe_{m['short']}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"[probe] wrote {out}")


if __name__ == "__main__":
    main()
