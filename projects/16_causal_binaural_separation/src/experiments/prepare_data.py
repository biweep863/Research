"""Download corpora, build BRIR banks and the fixed evaluation sets (all cached).

python -m src.experiments.prepare_data --config configs/local_small.yaml

Splits:
  train / val / test : KEMAR HRTF, disjoint rooms, disjoint speakers,
                       training noise environments split in time.
  heldout            : CIPIC HRTF (another head), new rooms, test speakers,
                       noise environments never seen in training.
"""

from __future__ import annotations

import json
import os
import time

from .common import cache_tag, data_root, dump, load_cfg, resolve_device  # noqa: I001  (sets sys.path)

import numpy as np
import torch

from src.data.corpora import fetch_hrtf, load_noise, load_speech
from src.data.hrtf import load_hrtf
from src.data.mixtures import Mixer, fixed_set
from src.data.rooms import make_bank

BANK_SEED = {"train": 1, "val": 2, "test": 3, "heldout": 4}
SET_SEED = {"val": 11, "test": 12, "heldout": 13}


def get_bank(cfg: dict, split: str) -> dict:
    root = data_root(cfg)
    f = root / "cache" / f"bank_{cache_tag(cfg)}_{split}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        meta = json.loads(f.with_suffix(".json").read_text())
        return {"target_full": z["target_full"], "target_early": z["target_early"], "noise": z["noise"], **meta}
    which = "heldout" if split == "heldout" else "main"
    name = cfg["data"]["hrtf"][which]
    fetch_hrtf(name, root)
    hrtf = load_hrtf(name, cfg["fs"], root / "raw")
    n = cfg["rooms"][f"n_{split}"]
    t = time.time()
    bank = make_bank(n, cfg["rooms"], hrtf, cfg["fs"], BANK_SEED[split], workers=min(8, os.cpu_count() or 1))
    print(f"[prepare] bank {split}: {n} rooms with HRTF '{hrtf.name}' in {time.time() - t:.0f}s")
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, target_full=bank["target_full"], target_early=bank["target_early"], noise=bank["noise"])
    f.with_suffix(".json").write_text(json.dumps({"meta": bank["meta"], "hrtf": bank["hrtf"]}))
    return bank


def get_mixer(cfg: dict, split: str, device) -> Mixer:
    root = data_root(cfg)
    speech = load_speech("test" if split == "heldout" else split, cfg, root)
    noise = load_noise(split, cfg, root)
    return Mixer(speech, noise, get_bank(cfg, split), cfg, device)


def get_fixed_set(cfg: dict, split: str, device) -> dict:
    """Cached deterministic set for split in {val, test, heldout}."""
    f = data_root(cfg) / "cache" / f"set_{cache_tag(cfg)}_{split}.pt"
    if f.exists():
        return torch.load(f)
    mixer = get_mixer(cfg, split, device)
    seg = int(cfg["mix"]["test_seg_s"] * cfg["fs"])
    s = fixed_set(mixer, cfg["mix"][f"n_{split}"], seg, SET_SEED[split])
    s["synthetic"] = mixer.synthetic
    s["hrtf"] = get_bank(cfg, split)["hrtf"]
    torch.save(s, f)
    return s


def main():
    cfg, out_dir = load_cfg()
    device = resolve_device(cfg)
    root = data_root(cfg)
    summary = {"speech": {}, "noise": {}, "banks": {}, "sets": {}}
    for split in ("train", "val", "test"):
        p = load_speech(split, cfg, root)
        summary["speech"][split] = {"hours": p.hours, "files": len(p.lengths), "synthetic": p.synthetic}
    for split in ("train", "val", "test", "heldout"):
        p = load_noise(split, cfg, root)
        summary["noise"][split] = {"hours": p.hours, "synthetic": p.synthetic}
    for split in ("train", "val", "test", "heldout"):
        b = get_bank(cfg, split)
        rt = np.array([m["rt60"] for m in b["meta"]])
        summary["banks"][split] = {
            "rooms": len(b["meta"]), "hrtf": b["hrtf"], "rt60_mean": float(rt.mean()),
            "rt60_range": [float(rt.min()), float(rt.max())],
        }
    for split in ("val", "test", "heldout"):
        s = get_fixed_set(cfg, split, device)
        summary["sets"][split] = {
            "n": int(s["mix"].shape[0]), "seconds": s["mix"].shape[-1] / cfg["fs"],
            "snr_mean": float(s["snr_db"].mean()), "synthetic": bool(s["synthetic"]), "hrtf": s["hrtf"],
        }
    print(json.dumps(summary, indent=1))
    dump(out_dir, "data_summary.json", summary)


if __name__ == "__main__":
    main()
