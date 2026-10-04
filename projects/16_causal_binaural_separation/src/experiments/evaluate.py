"""Score every trained run on the test and held-out sets.

python -m src.experiments.evaluate --config configs/local_small.yaml

Writes results/eval/<run>.json (per-utterance metrics + bootstrap CIs) and
results/eval_summary.json (all runs, plus paired Wilcoxon tests between
adjacent latencies of each causal family).
"""

from __future__ import annotations

import json

from .common import dump, load_cfg, resolve_device  # noqa: I001  (sets sys.path)

import numpy as np
import torch
from scipy.stats import wilcoxon

from src.models.factory import build_model, run_name

from .eval_common import enhance_set, score_set, summarize
from .prepare_data import get_fixed_set
from .train_sweep import sweep_specs


def paired_tests(evals: dict, cfg: dict, split: str = "test", metric: str = "si_sdr_i") -> list[dict]:
    """Wilcoxon signed-rank on per-utterance ``metric`` between adjacent latencies."""
    out = []
    for fam in cfg["sweep"]["families"]:
        runs = sorted(
            (e for e in evals.values() if e["spec"]["family"] == fam and e["spec"].get("causal", True)
             and not e["spec"].get("lookahead", 0) and e["spec"].get("head") == cfg["sweep"]["head"]
             and not e["spec"].get("steps")),
            key=lambda e: e["spec"]["latency_ms"],
        )
        for a, b in zip(runs, runs[1:]):
            x = np.asarray(a[split]["per_utt"][metric])
            y = np.asarray(b[split]["per_utt"][metric])
            p = float(wilcoxon(y, x).pvalue) if len(x) > 5 else float("nan")
            out.append({"family": fam, "from_ms": a["spec"]["latency_ms"], "to_ms": b["spec"]["latency_ms"],
                        "mean_diff": float(np.mean(y - x)), "p": p, "n": len(x)})
    return out


def write_examples(audio_dir, name: str, est: np.ndarray, s: dict, fs: int, n: int = 3) -> None:
    """Before/after WAVs (stereo = left/right ear) for listening checks."""
    import soundfile as sf

    audio_dir.mkdir(exist_ok=True)
    for i in range(min(n, est.shape[0])):
        peak = float(np.abs(s["mix"][i].numpy()).max()) + 1e-9
        for tag, x in (("mix", s["mix"][i].numpy()), ("target", s["target"][i].numpy()), (name, est[i])):
            f = audio_dir / f"utt{i}_{tag}.wav"
            if tag == name or not f.exists():
                sf.write(f, (0.9 * x / peak).T, fs)


def main():
    cfg, out_dir = load_cfg()
    device = resolve_device(cfg)
    fs = cfg["fs"]
    sets = {s: get_fixed_set(cfg, s, device) for s in ("test", "heldout")}
    (out_dir / "eval").mkdir(exist_ok=True)
    evals = {}
    for spec in sweep_specs(cfg):
        name = run_name(spec)
        f = out_dir / "eval" / f"{name}.json"
        ck = out_dir / "ckpt" / f"{name}.pt"
        if f.exists():
            evals[name] = json.loads(f.read_text())
            continue
        if not ck.exists():
            print(f"[evaluate] {name}: no checkpoint, skipping")
            continue
        model = build_model(spec, cfg).to(device).eval()
        model.load_state_dict(torch.load(ck)["state"])
        rec = {"name": name, "spec": spec, "latency_ms": model.latency_samples / fs * 1000,
               "params": sum(p.numel() for p in model.parameters())}
        for split, s in sets.items():
            est = enhance_set(model, model.hop, s["mix"], device)
            per = score_set(est, s["mix"].numpy(), s["target"].numpy(), fs)
            rec[split] = {"summary": summarize(per), "per_utt": per}
            sm = rec[split]["summary"]
            print(f"  {split:8s} {name:30s} SI-SDRi {sm['si_sdr_i'][0]:6.2f}  ESTOI {sm['estoi'][0]:.3f}  "
                  f"dILD {sm['d_ild_db'][0]:.2f} dB  dITD {sm['d_itd_us'][0]:5.1f} us", flush=True)
            if split == "test":
                write_examples(out_dir / "audio", name, est, s, fs)
        dump(out_dir / "eval", f"{name}.json", rec)
        evals[name] = rec
    summary = {
        "runs": {n: {"spec": e["spec"], "latency_ms": e["latency_ms"], "params": e["params"],
                     "test": e["test"]["summary"], "heldout": e["heldout"]["summary"]} for n, e in evals.items()},
        "wilcoxon_adjacent_latency": paired_tests(evals, cfg),
        "synthetic_data": bool(sets["test"]["synthetic"]),
    }
    dump(out_dir, "eval_summary.json", summary)


if __name__ == "__main__":
    main()
