# 04 · Results (local pilot, `local_small.yaml`)

Formal write-up: `paper/main.pdf`. All numbers below come from `results/`
and are mirrored in `paper/macros.tex`. Test set: 300 × 3 s; held-out: 150
mixtures with a different head, unseen noise and unseen rooms.

## Headlines
| Question | Answer |
|---|---|
| What does the window cost? (oracle IRM, asym.) | 6.88 dB at 1 ms vs 7.88 dB at 32 ms, i.e. about 1 dB. Symmetric window at 1 ms: 6.15 dB; asymmetric analysis recovers 0.73 dB (ESTOI 0.851 vs 0.781) |
| What does the future buy? | ≈0. AsymDF 4 ms: non-causal 2.55 vs causal 2.54 dB (2k steps), 3.60 vs 3.57 (12k). Matched-hop lookahead: +0.03 dB (+4 ms) and −0.03 dB (+6 ms) |
| What does learning cost? | 4.69 dB (2k) and 3.64 dB (12k) between the oracle and the non-causal model at 4 ms: the dominant term |
| 4 ms within 1 dB of 32 ms? | "Yes" (4 ms is 1.59 dB *better* at 12k), but uninformative: short hops get 16× more supervised frames per step, and 32 ms is still under-trained |
| Do the cues survive? | ILD yes: all models < 1 dB JND (0.36 dB best, 1.21 dB unprocessed). ITD no: best 47.8 µs vs 20 µs JND; even the oracle is 21.7 µs, estimator floor 3.8 µs |
| Generalisation (CIPIC head + unseen noise) | Large drop for all models. The common-gain head transfers best (1.58 dB held-out vs 0.44 dB per-ear DF at 4 ms) |
| Real time on 1 CPU thread? | AsymDF: 0.42 ms per chunk at every hop → RTF 0.85 at 1 ms, 0.21 at 4 ms. Conv-TasNet: 2.46 ms per chunk → real-time only from 8 ms (RTF 0.61) |
| Classical baseline | Causal common-gain Wiener: +0.6…1.2 dB, no ESTOI gain |

## Verdict on the hypothesis (doc/00)
- *Loss is dominated by resolution, not by the future*: **half supported.**
  The future is worth ≈0, but resolution (0.64 dB at 4 ms) is also small
  compared with learning (3.6–4.7 dB).
- *4 ms within 1 dB of 32 ms*: met, but not counted as support (training
  confound).
- *Target cues within JND*: **falsified for ITD**, met for ILD.
- *Lookahead beats window at a fixed budget*: not observed once the hop is
  matched, so it doesn't falsify the hypothesis.

## Statistics
Paired Wilcoxon tests between adjacent latencies (AsymDF) show quality
**falling** with latency at equal steps: 2→4 ms −0.22 dB (p = 5e-25),
4→8 ms −0.39 dB, 8→32 ms −1.06 dB (p = 2e-48). Every family shows the same
sign. This is the training-density confound, not acoustics.

## What the GPU run must answer
Does the future start to matter once the learning gap closes (1–2M
parameters, 40k steps, train-clean-100)? Does the 32 ms model overtake the
4 ms one at a frame-matched budget? `make all CONFIG=configs/gpu_full.yaml`
runs the same cells at 6 latencies.
