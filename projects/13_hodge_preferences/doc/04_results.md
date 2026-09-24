# 04 · Results

Headline numbers from the local config. All of them are emitted as macros in `paper/macros.tex`.

## Spectrum (96 graphs, n ∈ {20, 40}, density 0.15–1, d ∈ {2, 8, 32}, depth 1–2)
- **Scalar model.** The cycle rate is ≤ 3e-16, confirming Proposition 1.
- **Antisymmetric net.** κ_grad/κ_cyc ranges from **4.0 to 123**.
  - It averages 13.8× at density 0.15 and **63×** on complete graphs, growing with mean degree.
  - Low feature dimension sharpens it (48× at d = 2 vs 26× at d = 32), and so does a shallow net (35× at depth 1 vs 17× at depth 2).
- **Top modes.** The top-10 eigenvectors carry **99 %** of their mass in gradient space.
- **Curl vs harmonic.** Their rate ratio is 0.80–1.71 (mean 1.02), so the two are **not ordered**.
- **Finite width.** The empirical ratio at init matches infinite width already at width 128 (14.5 vs 15.0). GPM is less ordered (4–6×).

## Dynamics (40 items, d = 8, 59/30/11 % grad/curl/harm)
- **Order.** It held in **12 of 12** runs.
- **How much later cycles are fitted (curl / harmonic).**
  - Antisymmetric net: **25× / 29×** (squared loss) and **19× / 22×** (BT loss).
  - GPM: 12× / 19× (squared loss).
- **Linear theory.** The linearised NTK prediction is within **0.02** of training at every step.
- **Held-out sign accuracy.** Scalar 0.681, antisymmetric **0.882**, GPM 0.840.

## Width and lazy/rich (widths 32–1024, α ∈ {0.1, 1, 10})
- **Order.** It held in **24 of 24** runs.
- **Gap.** t90(curl)/t90(grad) is 66× in the lazy regime and **17×** in the rich one.
- **Kernel drift.** 0.03 (lazy) vs 6.6 (rich).
- **Held-out accuracy.** Rich 0.892 vs lazy 0.868.

## Noise
- **Lemma 2.** The measured fraction is within **0.0048** of (|V|−1)/|E| in 9 graphs (0.025 on an 80-item complete graph).
- **Gaussian noise.** 0.20 of it lands in gradient space (theory 0.21).
- **Vote flips.** **0.60** of the perturbation lands in gradient space, because flips shrink the transitive signal.
- **t\*.** Accuracy at t* is 0.750, against 0.872 at the best step: too early.
- **Early-stopping gain at 40 % flips.**
  - Antisymmetric: +4.3 pts (0.752 → 0.795).
  - GPM: **+12.0 pts** (0.656 → 0.776).
  - Scalar: +4.1 pts.

## Chatbot Arena (63 models, 640 pairs, 52 200 votes)
- **Hodge split.** 84.7 % gradient / 15.3 % curl / 0.0 % harmonic.
- **Transitive null with the same counts.** 17.5 % cyclic (95 % interval 15.2–19.7). The observed cyclic fraction sits below it (p = 0.96), so **the cycles are sampling noise**.
- **Order.** The kernel ratio is 44× (antisymmetric) and 160× (GPM). Training fits the ranking by step ~39 and the cycles by ~2 000.
- **Held-out log-loss.** Scalar **0.6282** (= HodgeRank 0.6284) < GPM 0.6306 < antisymmetric 0.6317.
- **At 40 % flips.** Accuracy is 0.622 (scalar) vs 0.561 (antisymmetric) and 0.564 (GPM). The best early stop recovers 0.616 and 0.617.

## Interpretation
Training time is a knob on transitivity: rankings come first and cycles last.
Flexible cyclic models help only when real cycles exist; otherwise their late
phase fits noise, and early stopping turns them back into scalar models. See
`paper/main.pdf`.
