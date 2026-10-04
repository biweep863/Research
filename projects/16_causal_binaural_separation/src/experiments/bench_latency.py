"""Measured real-time cost of streaming on CPU (batch 1, one thread).

python -m src.experiments.bench_latency --config configs/local_small.yaml

For each latency and causal family: feed ``n_chunks`` hops through
``model.stream`` and time every call. Real-time factor RTF = per-chunk compute
/ chunk duration; RTF < 1 means the model keeps up, and the total delay heard
is the algorithmic latency plus at most one chunk of compute.
Weights do not change the cost, so trained checkpoints are used when present
and freshly initialised models otherwise.
"""

from __future__ import annotations

import platform
import time

from .common import dump, load_cfg  # noqa: I001  (sets sys.path)

import numpy as np
import torch

from src.analysis.compute import macs_per_second
from src.models.factory import build_model, run_name


@torch.no_grad()
def time_stream(model, n_chunks: int, warmup: int = 50) -> np.ndarray:
    st = model.init_state(1, "cpu")
    x = torch.randn(n_chunks + warmup, 1, 2, model.hop) * 0.05
    times = np.empty(n_chunks)
    for i in range(warmup):
        model.stream(x[i], st)
    for i in range(n_chunks):
        t = time.perf_counter()
        model.stream(x[warmup + i], st)
        times[i] = time.perf_counter() - t
    return times


def main():
    cfg, out_dir = load_cfg()
    fs = cfg["fs"]
    b = cfg["bench"]
    torch.set_num_threads(b["threads"])
    rows = []
    for lat in b["latencies_ms"]:
        for fam in cfg["sweep"]["families"]:
            if fam == "stft_sym" and lat >= cfg["model"]["stft"]["analysis_ms"]:
                continue
            spec = {"family": fam, "latency_ms": lat, "head": cfg["sweep"]["head"], "causal": True}
            model = build_model(spec, cfg).eval()
            ck = out_dir / "ckpt" / f"{run_name(spec)}.pt"
            if ck.exists():
                model.load_state_dict(torch.load(ck)["state"])
            t = time_stream(model, b["n_chunks"])
            chunk_s = model.hop / fs
            row = {
                "family": fam, "latency_ms": lat, "hop": model.hop, "chunk_ms": chunk_s * 1000,
                "median_ms": float(np.median(t) * 1000), "p99_ms": float(np.quantile(t, 0.99) * 1000),
                "rtf_median": float(np.median(t) / chunk_s), "rtf_p99": float(np.quantile(t, 0.99) / chunk_s),
                "params": sum(p.numel() for p in model.parameters()),
                "mmacs_per_s": macs_per_second(model, fs) / 1e6,
            }
            rows.append(row)
            print(f"  {fam:10s} {lat:5.1f} ms  chunk {row['chunk_ms']:.2f} ms  compute med {row['median_ms']:.3f} "
                  f"p99 {row['p99_ms']:.3f} ms  RTF {row['rtf_median']:.2f}/{row['rtf_p99']:.2f}  "
                  f"{row['mmacs_per_s']:.0f} MMAC/s", flush=True)
    dump(out_dir, "bench_latency.json", {
        "rows": rows, "threads": b["threads"], "cpu": platform.processor(), "torch": torch.__version__,
    })


if __name__ == "__main__":
    main()
