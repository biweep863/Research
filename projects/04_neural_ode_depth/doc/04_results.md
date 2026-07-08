# 04 · Results

Headline (56k-param model, synthetic copy task):
- Next-token loss range across 7 ODE configs:  **0.053 nats**
- Next-token loss range across 3 discrete depths: **22.9 nats**

Interpretation: a block trained with a cheap continuous schedule is
robust to any inference-time solver budget within two orders of
magnitude of accuracy; the same block unrolled at a discrete depth
different from its training depth corrupts the residual stream
catastrophically. This is the "anytime inference" property the paper
claims.

On raw wall-clock the strongest discrete configuration is slightly
faster than the cheapest equivalent-accuracy ODE, so the story is
robustness/latency trade-off, not a raw latency win.

See `paper/main.pdf` for figures (Pareto front, acc-vs-NFE,
train-loss curve).
