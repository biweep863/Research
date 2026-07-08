# 04 · Results

Headline (GPT-2 small, 10-pair pilot):
- Mean $H_0$ total persistence:  truthful **3.406** vs hallucinated **3.338** (Δ ≈ 0.07).
- Std ≈ 2.1 on $H_0$ total persistence.
- Min per-layer $p$-value: **0.852 at layer 7**.
- Layers reaching $p < 0.05$: **0 / 12**.

Interpretation: the pilot is by design underpowered. The observed
effect size (Δ ≈ 0.07 against a std ≈ 2.1) combined with $\alpha=0.05$,
$\beta=0.2$ prescribes $N \geq \sim 200$ pairs per condition for a
properly powered Welch $t$-test. The `gpu_full.yaml` configuration is
sized accordingly and targets TruthfulQA. $H_1$ is not informative at
this prompt length — very few persistent 1-cycles appear in the
distance matrices.

The pilot's value is validating the end-to-end pipeline
(attention → distance → Vietoris–Rips → statistics) and producing
an effect-size estimate to plan the scaled run.

See `paper/main.pdf` for figures ($H_0$/$H_1$ per layer,
$-\log_{10} p$ per layer, Betti curves).
