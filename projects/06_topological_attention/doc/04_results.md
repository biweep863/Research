# 04 · Results

## Headline (GPT-2 small, 10-pair pilot)
- Mean $H_0$ total persistence:  truthful **3.406** vs hallucinated **3.338** (Δ ≈ 0.07).
- Std ≈ 2.1 on $H_0$ total persistence (unpaired).
- Min per-layer $p$-value (unpaired Welch): **0.852 at layer 7**.
- Layers reaching $p < 0.05$: **0 / 12**.

## Power analysis — corrected

The pilot's own write-up originally prescribed $N \geq \sim 200$ pairs from
$\Delta \approx 0.07$ against an **unpaired** std $\approx 2.1$
($\alpha=0.05$, $\beta=0.2$). That arithmetic is wrong: Cohen's
$d = 0.07 / 2.1 \approx 0.033$, and a two-sample Welch test needs
$n \approx 2(z_{\alpha/2}+z_\beta)^2/d^2 \approx 14{,}000$ pairs *per group*
to detect an effect that small at 80% power — not 200. Confirmed
empirically below: the unpaired test stays null even at $N \approx 500$.

The correct fix is not more $N$ on the unpaired test, but a **paired**
design: each pair shares a prompt, so a paired (or Wilcoxon) test on the
per-pair difference exploits that correlation and needs far less data.
Retrospectively re-analysing the 10-pair pilot itself with a paired test
gives per-layer $d_z$ (mean difference / std of the difference) of
**0.13–0.33** — an order of magnitude larger than the unpaired $d=0.033$
computed from pooled variance. At $d_z \approx 0.2$, $\alpha=0.05$,
80% power needs $N \approx 200$; with a Holm correction across ~12–36
layers, $N \approx 300$–$400$. This is the analysis this project
pre-registered as primary before running `gpu_full.yaml` (see
`doc/05_decisions.md`).

## Headline (GPU run: three models, TruthfulQA, exact length match)

Meta-Llama-3-8B was the originally intended scaled model, but its HF
gated-access request was still pending when the run needed to start;
**Qwen/Qwen3-8B** (Apache-2.0, ungated, comparable size) was substituted
per the plan's documented fallback order (see `doc/05_decisions.md`). A
`gpt2` control run on the *same* TruthfulQA-sourced pairs (N=509,
matched with GPT-2's own tokenizer) was run alongside it to separate
"bigger model" from "more/different data" effects. HF access to
Llama-3-8B was approved mid-session and a same-design run (N=495) was
added as a further replication rather than a replacement.

| | gpt2 (control, N=509) | Qwen3-8B (N=490) | Llama-3-8B (N=495) |
|---|---|---|---|
| Layers | 12 | 36 | 32 |
| Unpaired Welch, min $p$ | 0.579 | 0.613 | 0.481 |
| Unpaired Welch, $N$ sig. ($p<0.05$) | 0/12 | 0/36 | 0/32 |
| **Paired $t$-test, min $p$** | 1.64e-06 (layer 5) | 7.91e-07 (layer 21) | **9.14e-11 (layer 18)** |
| **Paired, Holm-sig. layers** | 3/12 | 11/36 | **13/32** |
| Wilcoxon signed-rank, min $p$ | 2.09e-05 | 3.72e-06 | 4.80e-10 |
| Length-normalized (H0/n_tokens), min $p$ | 3.33e-06 | 2.63e-07 | 7.58e-11 |
| No-BOS-sink variant, min $p$ | 1.25e-05 | 1.23e-08 | 7.25e-05 |
| Mean paired effect size $d_z$ | -0.022 | +0.034 | **+0.093** |
| Max $\lvert d_z\rvert$ | 0.215 | 0.226 | 0.298 |

**Reading this honestly.** The *unpaired* Welch test — the pilot's
original method — stays null in all three models even at N≈500,
consistent with the corrected power analysis above (it would need ~14k
pairs to detect an effect this small). The *paired* test — the
pre-registered primary analysis, appropriate because pairs share a
prompt — finds statistically significant per-layer differences in all
three models after Holm correction, confirmed by the non-parametric
Wilcoxon test and a length-normalized control. The three models trace a
consistent trend: GPT-2's mean effect direction is slightly *negative*
($d_z=-0.022$, opposite the pilot's hypothesis) despite 3 layers
clearing significance; Qwen3-8B is positive but small ($d_z=+0.034$);
Llama-3-8B is positive and roughly 3× larger ($d_z=+0.093$, max
$0.298$ — near the top of the pilot's own retrospective 0.13–0.33
range), with 13 of 32 layers (41%) surviving Holm correction. Two
independent 7–8B-parameter models now show a correctly-signed,
statistically robust effect that the small pilot and GPT-2-scale
control do not — compatible with the topological signal being real but
weak at small scale, and growing with model capacity. This is not proof
of a causal or general trend (two data points), and the no-BOS-sink
variant flips direction between the two big models (weaker than the
with-sink result for Llama-3-8B, stronger for Qwen3-8B) — an open
question flagged rather than explained away.

$H_1$ remains uninformative at these prompt lengths in all three models
(mean total persistence ≈0.000–0.023, no layers show a resolvable
signal); $H_2$ is uniformly zero, as expected for such short filtered
complexes.

See `paper/main.pdf` for figures ($H_0$/$H_1$ per layer, paired $d_z$
per layer with bootstrap CI, $-\log_{10} p$ per layer, Betti curves).
