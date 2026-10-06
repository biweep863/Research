# 02 · Methodology

## Pipeline
```
LibriSpeech ─┐                              ┌─ oracle IRM / Wiener ─┐
DEMAND ──────┼─► GPU mixer ◄── BRIR bank ◄──┤                       ├─► score_set ─► results/*.json
SOFA HRTF ───┘   (mix, early target)        │  ISM + HRIR + tail    │   (8 CPU workers)       │
                                            └─ train_sweep ─► ckpt ─┘                         ▼
                                               bench_latency (CPU, 1 thread) ─► plot_latency ─► figures + macros.tex ─► paper
```

## Latency definition (`src/models/framing.py`)
Frame t, with hop R and analysis length N, covers input samples
[(t+1)R − N, (t+1)R). It ends at the newest sample. Synthesis uses only the
last 2R samples of each frame, overlap-added at hop R, and the analysis ×
synthesis window equals a periodic Hann(2R) there, so reconstruction is
perfect (tested). Sample n is final once the frame ending at
(⌊n/R⌋ + 2)·R has been processed:

    algorithmic latency  L = 2R            (+ D·R with D lookahead frames)

Compute time comes on top of this and is reported separately as a
real-time factor. The asymmetric window (Mauler & Martin 2007; Wang et al.
2023) has a rising sqrt-Hann over N − R samples and a falling
sqrt-Hann(2R) half. The synthesis window is Hann(2R)/analysis on the last 2R
samples. With N = 2R it reduces to the ordinary sqrt-Hann pair.

For the learned models, streaming equality with the offline forward pass is
**tested** (`tests/test_models.py`): the outputs agree to 1e-4 after a delay
of L − R. Causality is tested too: changing future input leaves earlier
outputs unchanged, and the non-causal reference fails that same test.

## Models
| name | analysis | core | head | latency |
|---|---|---|---|---|
| `stft_asym` (AsymDF) | 32 ms asym. | LN → Linear → 2×GRU(128) | per-ear complex deep filter, 3 past frames | 2R |
| `stft_sym` | 2R sym. | same | same | 2R |
| `tasnet` | learned 2R basis (≥ 4R filters) | causal TCN 2×6 blocks, per-frame LN | per-ear real mask | 2R |
| `*_nc` | same | bidirectional GRU / centred convs | same | offline |
| `head=common` | — | — | one real gain for both ears | — |
| oracle IRM | asym / sym | √(\|S\|²/(\|S\|²+\|X−S\|²)) per ear or common | — | 2R |
| Wiener | asym / sym | min-tracking noise PSD, decision-directed SNR, common gain | — | 2R |

## Loss
L = −SNR_binaural(ŝ, s) + λ_cue·(|ΔILD|_w + 10·(1 − cos ΔIPD)_w), with λ_cue = 1.

- SNR is computed jointly over both ears and depends on scale, so neither
  ear can pick its own gain (per-ear SI-SDR would let it, which changes the
  ILD).
- The cue terms are weighted by target energy and restricted to bins within
  20 dB of the maximum. IPD only counts below 1.5 kHz.

## Metrics (`src/analysis/`)
- **SI-SDR(i)**, averaged over ears (Le Roux et al. 2019). The improvement is
  measured over the unprocessed mixture against the early target.
- **ESTOI** (Jensen & Taal 2016), via pystoi, averaged over ears.
- **PESQ-WB**: NaN unless the `pesq` package is installed (doc/05).
- **Target cue errors**, comparing the enhanced output with the clean early
  target:
  - |ΔITD| from band-limited (100–1500 Hz) GCC-PHAT with a 1/32-sample peak
    search;
  - |ΔILD| broadband;
  - per-bin energy-weighted |ΔILD| and |ΔIPD| (< 1.5 kHz).
- **Residual-noise cue shift**: the cues of ŝ − s against those of x − s.
- **JNDs used as pass lines**: ILD 1 dB, ITD 20 µs (Mills 1960; Klumpp &
  Eady 1956 report about 10 µs at best).

## Gap decomposition
At latency L, for the AsymDF family, all terms are in dB of SI-SDRi:
- total = C(32) − C(L)
- window / resolution = O(32) − O(L), where O is the oracle IRM
- future = NC(L) − C(L)
- learning = O(L) − NC(L)

These are not an exact additive split. They are reported side by side.

## Statistics
- Every summary is a mean with a 1000-sample bootstrap 95 % CI
  (`shared/utils/stats.bootstrap_ci`).
- Adjacent latencies of each causal family are compared with paired
  Wilcoxon signed-rank tests on per-utterance SI-SDRi (300 test utterances
  locally).
- Locally there is one seed per cell, so seed variance isn't estimated. That
  is a stated limitation, fixed in the GPU config by the larger test set,
  not by more seeds.

## Compute
The benchmark streams 2000 hops per model through `model.stream()` at batch
size 1 with `torch.set_num_threads(1)`, after a 50-hop warm-up. It reports:
- median and p99 wall time per chunk;
- RTF = time / chunk duration;
- analytic MACs/s, counted from layer shapes plus FFTs.
