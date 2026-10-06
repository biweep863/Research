"""Train every (family x latency x head x causality x lookahead) cell of the sweep.

python -m src.experiments.train_sweep --config configs/local_small.yaml [--only name1,name2]

Finished runs (checkpoint + log present) are skipped, so the sweep can be
interrupted and resumed.
"""

from __future__ import annotations

import math
import time

from .common import dump, load_cfg, resolve_device  # noqa: I001  (sets sys.path)

import numpy as np
import torch

from src.models.factory import build_model, hop_for, run_name
from src.models.losses import total_loss

from .eval_common import run_padded
from .prepare_data import get_fixed_set, get_mixer


def sweep_specs(cfg: dict) -> list[dict]:
    s = cfg["sweep"]
    fams = s["families"]
    ana = cfg["model"]["stft"]["analysis_ms"]
    specs = []
    for fam in fams:
        for lat in s["latencies_ms"]:
            if fam == "stft_sym" and lat >= ana:
                continue  # identical to stft_asym once the synthesis window reaches the analysis window
            specs.append({"family": fam, "latency_ms": lat, "head": s["head"], "causal": True})
    other = "common" if s["head"] == "ear" else "ear"
    for lat in s.get("head_ablation_ms", []):
        for fam in ("stft_asym", "tasnet"):
            if fam in fams:
                specs.append({"family": fam, "latency_ms": lat, "head": other, "causal": True})
    for lat in s.get("noncausal_ms", []):
        for fam in ("stft_asym", "tasnet"):
            if fam in fams:
                specs.append({"family": fam, "latency_ms": lat, "head": s["head"], "causal": False})
    for D in s.get("lookahead_frames", []):
        try:
            hop_for(s["lookahead_budget_ms"], cfg["fs"], D)
        except ValueError:
            continue
        specs.append({"family": "stft_asym", "latency_ms": s["lookahead_budget_ms"], "head": s["head"],
                      "causal": True, "lookahead": D})
    lc = cfg.get("long_check") or {}
    for cell in lc.get("cells", []):  # same cells, longer training: do the gaps survive better models?
        specs.append({"head": s["head"], "causal": True, **cell, "steps": lc["steps"]})
    return specs


@torch.no_grad()
def si_sdr_torch(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-utterance SI-SDR averaged over ears, (B, 2, T) -> (B,)."""
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + 1e-12)
    t = a * ref
    return (10 * torch.log10(t.pow(2).sum(-1) / ((est - t).pow(2).sum(-1) + 1e-12) + 1e-12)).mean(-1)


@torch.no_grad()
def validate(model, val: dict, device, batch: int = 16) -> float:
    model.eval()
    vals = []
    for i in range(0, val["mix"].shape[0], batch):
        x = val["mix"][i : i + batch].to(device)
        y = val["target"][i : i + batch].to(device)
        e = run_padded(model, x, model.hop)
        vals.append(si_sdr_torch(e, y) - si_sdr_torch(x, y))
    model.train()
    return float(torch.cat(vals).mean())


def train_one(spec: dict, cfg: dict, mixer, val: dict, device, out_dir) -> dict:
    name = run_name(spec)
    t_cfg = cfg["train"]
    torch.manual_seed(cfg["seed"])
    model = build_model(spec, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Learned TasNet basis: 10x lower lr keeps the pass-through init from being wrecked early (doc/05)
    basis = [p for n, p in model.named_parameters() if n in ("enc.weight", "dec.weight")]
    rest = [p for n, p in model.named_parameters() if n not in ("enc.weight", "dec.weight")]
    groups = [{"params": rest}] + ([{"params": basis, "lr": t_cfg["lr"] * t_cfg["basis_lr_scale"]}] if basis else [])
    opt = torch.optim.AdamW(groups, lr=t_cfg["lr"], weight_decay=1e-4)
    steps, warm = spec.get("steps") or t_cfg["steps"], t_cfg["warmup_steps"]

    def lr_lambda(s):
        cos = 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
        return min(1.0, (s + 1) / warm) * max(t_cfg["lr_min"] / t_cfg["lr"], cos)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    seg = int(cfg["mix"]["train_seg_s"] * cfg["fs"])
    seg -= seg % model.hop
    rng = np.random.default_rng([cfg["seed"], 7])
    log = {"step": [], "loss": [], "snr": [], "ild": [], "ipd": []}
    val_log = {"step": [], "si_sdr_i": []}
    best, best_state = -1e9, None
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        b = mixer.sample(t_cfg["batch"], seg, rng)
        est = model(b["mix"])
        loss, parts = total_loss(est, b["target"], cfg["fs"], cue_weight=t_cfg["cue_weight"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])
        opt.step()
        sched.step()
        if step % 25 == 0 or step == 1:
            log["step"].append(step)
            log["loss"].append(loss.item())
            for k in ("snr", "ild", "ipd"):
                log[k].append(parts.get(k, float("nan")))
        if step % t_cfg["val_every"] == 0 or step == steps:
            v = validate(model, val, device)
            val_log["step"].append(step)
            val_log["si_sdr_i"].append(v)
            print(f"  [{name}] step {step:5d} loss {loss.item():7.3f} val SI-SDRi {v:6.2f} dB "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if v > best:
                best = v
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"spec": spec, "state": best_state}, ckpt_dir / f"{name}.pt")
    rec = {
        "name": name, "spec": spec, "params": n_params, "hop": model.hop,
        "latency_samples": model.latency_samples, "latency_ms": model.latency_samples / cfg["fs"] * 1000,
        "best_val_si_sdr_i": best, "train_s": time.time() - t0, "log": log, "val": val_log,
        "synthetic_data": bool(mixer.synthetic),
    }
    dump(out_dir / "runs", f"{name}.json", rec)
    return rec


def main():
    cfg, out_dir = load_cfg()
    device = resolve_device(cfg)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    (out_dir / "runs").mkdir(exist_ok=True)
    specs = sweep_specs(cfg)
    if cfg["_only"]:
        specs = [s for s in specs if run_name(s) in cfg["_only"]]
    print(f"[train_sweep] {len(specs)} runs: {[run_name(s) for s in specs]}")
    mixer = get_mixer(cfg, "train", device)
    if mixer.synthetic:
        print("[train_sweep] WARNING: training on synthetic fallback data; results will not transfer to speech")
    val = get_fixed_set(cfg, "val", device)
    for spec in specs:
        name = run_name(spec)
        if (out_dir / "runs" / f"{name}.json").exists() and (out_dir / "ckpt" / f"{name}.pt").exists():
            print(f"[train_sweep] {name}: done, skipping")
            continue
        print(f"[train_sweep] {name}")
        train_one(spec, cfg, mixer, val, device, out_dir)


if __name__ == "__main__":
    main()
