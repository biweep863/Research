"""Per-submodule layer sensitivity: quantize only attention OR only MLP
layers and measure accuracy + logit MSE relative to FP32."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import yaml
from utils.seeds import set_seed
from utils.model_loader import load_hf
from src.data.load_math import load_all
from src.experiments.quantize_models import clone_and_quantize
from src.experiments.run_math_eval import accuracy, logit_distance


MODULE_HINTS = {
    "attn": ["attn", "attention", "self_attn"],
    "mlp":  ["mlp", "ffn", "feed_forward", "fc"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    items = load_all(cfg, seed=cfg.get("seed", 0))
    ref_items = items[: cfg["mse_eval"]["n_samples"] * 2]
    bits = cfg["layerwise_sensitivity"]["bit_width"]

    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)

    for m in cfg["models"]:
        lm = load_hf(m["name"], device=cfg.get("device", "auto"), dtype=cfg.get("dtype"))
        baseline = accuracy(lm.model, lm.tokenizer, ref_items, lm.device)
        rows = []
        for label in cfg["layerwise_sensitivity"]["modules"]:
            hints = MODULE_HINTS[label]
            qm = clone_and_quantize(lm.model, bits=bits, only_modules=hints)
            acc = accuracy(qm, lm.tokenizer, ref_items, lm.device)
            mse, kl = logit_distance(lm.model, qm, lm.tokenizer, ref_items, lm.device,
                                     n_samples=cfg["mse_eval"]["n_samples"],
                                     max_length=cfg["mse_eval"]["max_length"])
            rows.append({"module": label, "bits": bits,
                         "acc": acc, "delta_acc": acc - baseline,
                         "logit_mse": mse, "logit_kl": kl})
            del qm
        out = out_root / f"sensitivity_{m['short']}.json"
        out.write_text(json.dumps({"baseline_acc": baseline, "bits": bits, "rows": rows}, indent=2))
        print(f"[sensitivity] wrote {out}")


if __name__ == "__main__":
    main()
