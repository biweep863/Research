# 05 · Decisions

## Latency = synthesis window (+ lookahead), not "chunk size"
Some papers quote the hop or the chunk as the latency. The overlap-add
output for a sample is only final after the *next* frame, so the honest
algorithmic latency is 2R. Every model is built from a latency budget
(`factory.hop_for`), and the streaming tests check the arithmetic. Compute
time is reported separately as an RTF, never folded into L.

## Asymmetric windows instead of only shrinking the STFT
A symmetric 2 ms window has 17 frequency bins at 16 kHz, which can't
resolve harmonics. The asymmetric analysis window keeps 32 ms of resolution
while the synthesis window, and with it the latency, stays at 2R. This is
the cheapest way to "spend the past", and the symmetric variant is trained
too so the effect can be measured, not assumed.
**Rejected:** a learned filterbank only (TasNet). It's included as a
family, but its frame length is tied to the latency, so it has no
long-window option.

## Target = direct path + early reflections (≤ 50 ms)
A dry target would ask the model to remove early reflections too, and with
them a large part of what defines the ITD/ILD in a room. A fully reverberant
target would make the cue metrics meaningless. The 50 ms boundary is the
usual one for perceptual integration of early reflections.
**Consequence:** the oracle IRM gains only about 6–8 dB, because late
reverberation overlaps the early target in TF, and SI-SDRi values are lower
than on dry-target benchmarks. They aren't comparable to those.

## Hybrid ISM + stochastic tail instead of a full-order ISM
An RT60 of 0.8 s needs image orders of about 50 (≈10⁵ images), each
rendered through an HRIR. Order 6 plus a level-matched binaural diffuse
tail gives the right decay (checked: T60 within 0.02 s) at 0.16 s per BRIR.

## Binaural SNR loss + cue loss, not per-ear SI-SDR
Per-ear SI-SDR is invariant to the gain of each ear, so a model can change
the ILD for free. The joint SNR depends on scale. The cue term (ILD + IPD)
then targets the TF bins where cues matter.

## Frame-wise LayerNorm in Conv-TasNet (not cLN / gLN)
gLN is non-causal. cLN accumulates statistics from t = 0, so streaming
state grows without bound and the result depends on when the device was
switched on. A per-frame channel LayerNorm is causal and has bounded state,
which is what lets `stream()` match `forward()` exactly.

## Conv-TasNet pass-through init, warm-up, and 0.1× basis learning rate
A randomly initialized TasNet starts 30–40 dB *below* the input. With
lr = 1e-3 on the encoder and decoder it diverged in the first steps, even
from an identity init (SNR −29 dB after 5 steps at 32 ms). What fixed it:
- an encoder of paired ±basis filters, which pass through the ReLU;
- a sqrt-Hann pseudo-inverse decoder;
- the mask bias set to +1, so the model starts as 0.73 × the input;
- a 100-step warm-up for all models;
- 0.1× learning rate on the basis.

Together these keep it stable (tested on 40 steps). The STFT model already
starts as identity, so both families start from the same point.
The filter count is raised to at least 4R (2 filters per frame sample) so
the pass-through is lossless at every latency. As a result, the 32 ms
TasNet has more filters than the 2 ms one.

## Sigmoid heads start at +1, not near-identity
The first sweep initialised the sigmoid mask/gain biases at +3 (STFT common
head) and +4 (TasNet) to start at the identity. At that point the sigmoid's
gradient is 0.018–0.045, and 6 of 8 sigmoid-headed runs stalled near
0 dB SI-SDRi after 2000 steps. That included TasNet at 4 and 32 ms and both
common heads, while TasNet at 2 and 8 ms happened to escape. With +1
(starting gain 0.73), 400-step probes improved training SNR by +6.7 dB
(TasNet) and +0.6 dB (common head), against +0.4 and +1.0 at the old
biases. All sigmoid-headed runs were deleted and retrained. The ear-head STFT
runs, which use deep-filter taps with no sigmoid, were kept.

## Equal steps ≠ equal training across latencies
At equal steps, a model with a 1 ms hop makes 16× more per-frame
predictions than one with a 16 ms hop. In the first sweep, validation
SI-SDRi *fell* with latency (2.8 → 1.1 dB from 2 to 32 ms), opposite to the
oracle trend. That's an artefact of under-training. The hypothesis test
therefore uses the `long_check` cells (12k steps), and the 2k-step sweep is
read only through paired comparisons at the *same* latency.

## Orthonormal TasNet basis
Even with the +1 bias, TasNet at 4 and 32 ms stayed at 0.2–0.4 dB while 2 ms
and 8 ms trained. At 4 and 32 ms the ± basis is *square* (64×64 and
512×512 per sign). A square Gaussian matrix is badly conditioned, so its
pseudo-inverse decoder had huge weights. Switching to QR-orthonormal columns
(condition number 1, so pinv = transpose) fixed every latency: 1.8–2.7 dB
after 2k steps. All TasNet runs were retrained with it.

## Model capacity is not held exactly constant across latencies
The STFT model's input and output layers scale with the number of bins
(17 bins for a 2 ms symmetric window, 257 for asymmetric), and TasNet's
filter count scales with the frame. The GRU/TCN core is fixed. The paper
reports parameter counts per cell. The alternative, matching parameters by
shrinking the core, would confound capacity with latency in the other
direction.

## PESQ is optional
`pesq` needs the MSVC build tools on Windows, which aren't installed. The
metric is wired in and reported as NaN when the package is missing. ESTOI
and SI-SDR carry the quality claims locally. On a Linux GPU box,
`pip install pesq` enables it with no code change.

## Faster ITD estimator
A 32× zero-padded IFFT (4 M points) made scoring take about 1.2 s per
utterance. It was replaced by an integer-lag peak followed by a direct
±1-sample fine search. The 1/32-sample grid is the same (≈2 µs), and the
unit tests (recovering −500…250 µs to within 3 µs) are unchanged. The
metric is about 4× faster.

## One seed per cell locally
The local run spends its budget on cells (19 runs), not on seeds. CIs are
over utterances. Seed-to-seed variance for small models is typically
0.2–0.5 dB, so local differences below that are not interpreted.
