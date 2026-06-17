"""Train a tiny NODE-Transformer and a discrete baseline, then sweep solvers /
discrete depths and record (accuracy, latency, NFE)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import torch
import torch.nn as nn
import yaml

from utils.seeds import set_seed
from utils.model_loader import best_device
from src.data.synth_seq import make_batch
from src.model import NODETransformer


def total_correct(logits, target, ignore_index=0):
    pred = logits.argmax(-1)
    mask = (target != ignore_index)
    return ((pred == target) & mask).sum().item(), mask.sum().item()


def step(x, model, fwd_kwargs, training: bool = True):
    # next-token loss on the *target* half of the sequence
    inp = x[:, :-1]
    tgt = x[:, 1:]
    if "depth" in fwd_kwargs:
        logits, nfe = model.forward_discrete(inp, fwd_kwargs["depth"])
    else:
        logits, nfe = model.forward_continuous(inp, **fwd_kwargs)
    loss = nn.functional.cross_entropy(logits.transpose(1, 2), tgt)
    return loss, logits, tgt, nfe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))
    device = best_device(cfg.get("device", "auto"))

    mcfg = cfg["model"]
    model = NODETransformer(mcfg["vocab_size"], mcfg["d_model"], mcfg["n_heads"],
                            mcfg.get("ffn_mult", 4),
                            max_len=cfg["task"]["seq_len"] * 2 + 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])

    print(f"[node] device={device} params={sum(p.numel() for p in model.parameters())}")
    # training uses a fixed cheap RK4 schedule for stability
    train_fwd = dict(solver="rk4", step_size=0.25)
    losses = []
    for s in range(cfg["train"]["steps"]):
        x = make_batch(cfg["task"]["type"], cfg["train"]["batch_size"],
                       cfg["task"]["seq_len"], cfg["model"]["vocab_size"],
                       device=device, seed=s)
        loss, *_ = step(x, model, train_fwd)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            print(f"[node]  step {s} loss {loss.item():.3f}")
            losses.append({"step": s, "loss": loss.item()})

    # evaluation
    model.eval()
    eval_x = make_batch(cfg["task"]["type"], cfg["eval"]["n_examples"],
                        cfg["task"]["seq_len"], cfg["model"]["vocab_size"],
                        device=device, seed=99999)
    rows = []
    with torch.no_grad():
        # ODE configs
        for o in cfg["ode_configs"]:
            kw = {k: v for k, v in o.items() if k != "name"}
            for fk in ("atol", "rtol", "step_size"):
                if fk in kw:
                    kw[fk] = float(kw[fk])
            t0 = time.perf_counter()
            loss, logits, tgt, nfe = step(eval_x, model, kw, training=False)
            dt = time.perf_counter() - t0
            c, n = total_correct(logits, tgt)
            rows.append({"name": o["name"], "kind": "ode",
                         "acc": c / max(1, n), "loss": loss.item(),
                         "latency_s": dt, "nfe": nfe})
        # discrete depths
        for d in cfg["discrete_depths"]:
            t0 = time.perf_counter()
            loss, logits, tgt, nfe = step(eval_x, model, {"depth": d}, training=False)
            dt = time.perf_counter() - t0
            c, n = total_correct(logits, tgt)
            rows.append({"name": f"discrete_{d}", "kind": "discrete",
                         "acc": c / max(1, n), "loss": loss.item(),
                         "latency_s": dt, "nfe": nfe})

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep.json").write_text(json.dumps({"train_losses": losses, "rows": rows}, indent=2))
    print(f"[node] wrote {out_dir/'sweep.json'}")


if __name__ == "__main__":
    main()
