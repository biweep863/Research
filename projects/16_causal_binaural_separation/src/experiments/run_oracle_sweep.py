"""Oracle masks + causal binaural Wiener across the latency grid (no training).

python -m src.experiments.run_oracle_sweep --config configs/local_small.yaml

For each latency L (hop = L/2) and window type:
  sym  : analysis window = synthesis window = L       (ordinary WOLA)
  asym : analysis window = max(32 ms, L), synthesis L (Mauler & Martin)
the ideal ratio mask (per ear / common gain) bounds what any mask can reach,
which isolates the frequency-resolution cost of the short window.
"""

from __future__ import annotations

from .common import dump, load_cfg, resolve_device  # noqa: I001  (sets sys.path)

import numpy as np
import torch
import torch.nn.functional as F

from src.models.classical import BinauralWiener, oracle_mask
from src.models.factory import hop_for

from .eval_common import enhance_set, score_set, summarize
from .prepare_data import get_fixed_set


def windows(lat_ms: float, cfg: dict) -> dict:
    fs = cfg["fs"]
    hop = hop_for(lat_ms, fs)
    ana = max(int(cfg["model"]["stft"]["analysis_ms"] * fs / 1000), 2 * hop)
    return {"sym": (2 * hop, hop), "asym": (-(-ana // hop) * hop, hop)}


@torch.no_grad()
def oracle_set(s: dict, n_fft: int, hop: int, kind: str, device, batch: int = 8) -> np.ndarray:
    T = s["mix"].shape[-1]
    pad = (-T) % hop
    outs = []
    for i in range(0, s["mix"].shape[0], batch):
        x = F.pad(s["mix"][i : i + batch], (0, pad)).to(device)
        y = F.pad(s["target"][i : i + batch], (0, pad)).to(device)
        outs.append(oracle_mask(x, y, n_fft, hop, kind)[..., :T].cpu())
    return torch.cat(outs).numpy()


def cue_floor(s: dict, snr_db: float, seed: int = 0) -> np.ndarray:
    """Clean target + independent white noise per ear at ``snr_db``: how much the
    cue estimators move when nothing spatial has changed (measurement floor)."""
    rng = np.random.default_rng(seed)
    t = s["target"].numpy()
    p = (t**2).mean(-1, keepdims=True)
    return t + rng.standard_normal(t.shape) * np.sqrt(p / 10 ** (snr_db / 10))


def main():
    import sys

    cfg, out_dir = load_cfg()
    device = resolve_device(cfg)
    fs = cfg["fs"]
    sets = {s: get_fixed_set(cfg, s, device) for s in ("test", "heldout")}
    entries = []

    def add(name, kind, window, lat, split, est):
        s = sets[split]
        per = score_set(est, s["mix"].numpy(), s["target"].numpy(), fs)
        summ = summarize(per)
        entries.append({"name": name, "kind": kind, "window": window, "latency_ms": lat, "set": split,
                        "summary": summ, "per_utt": {"si_sdr_i": per["si_sdr_i"], "estoi": per["estoi"]}})
        print(f"  {split:8s} {name:28s} SI-SDRi {summ['si_sdr_i'][0]:6.2f}  ESTOI {summ['estoi'][0]:.3f}  "
              f"dILD {summ['d_ild_db'][0]:.2f} dB  dITD {summ['d_itd_us'][0]:5.1f} us", flush=True)

    for split, s in sets.items():
        if "--floor-only" not in sys.argv:
            add("unprocessed", "unprocessed", "-", 0.0, split, s["mix"].numpy())
    test = sets["test"]
    floor_only = "--floor-only" in sys.argv
    if floor_only:  # append the floor to an existing sweep without recomputing it
        import json

        prev = json.loads((out_dir / "oracle_sweep.json").read_text())["entries"]
        entries[:] = [e for e in prev if e["kind"] != "floor"]
    for snr in (20.0, 30.0):
        add(f"floor_{snr:g}dB", "floor", f"{snr:g}dB", 0.0, "test", cue_floor(test, snr))
    for lat in ([] if floor_only else cfg["oracle"]["latencies_ms"]):
        for wname, (n_fft, hop) in windows(lat, cfg).items():
            if wname == "sym" and n_fft == windows(lat, cfg)["asym"][0]:
                continue
            for kind in ("ear", "common"):
                est = oracle_set(test, n_fft, hop, kind, device)
                add(f"oracle_{kind}_{wname}_{lat:g}ms", f"oracle_{kind}", wname, lat, "test", est)
            wf = BinauralWiener(n_fft, hop, fs)
            for split, s in sets.items():
                est = enhance_set(wf, hop, s["mix"], device, batch=32)
                add(f"wiener_{wname}_{lat:g}ms", "wiener", wname, lat, split, est)
    dump(out_dir, "oracle_sweep.json", {"entries": entries, "synthetic_data": bool(test["synthetic"])})


if __name__ == "__main__":
    main()
