# Project 10 — Who touched first? A boot-mounted IMU for foul detection

## Question
The connected match ball detects *when* it is struck (an IMU at its centre sees
the sudden change of acceleration). A foul turns on the other half of the
event: *what* the boot hit and *in which order*. Can a sensor in the boot —
same principle — tell ball from player from ground, time the contact, order a
ball touch and a player touch that happen almost together, and turn all of it
into an honest probability that the ball was played first? And where was it?

## The five stages
1. **Detection** — {none, ball, player, both} from a 0.6 s IMU window; the
   connected-ball threshold rule vs gradient boosting; sampling rate 100–1000 Hz;
   impact-rated (±200 g) vs phone-grade (±16 g) sensor; synthetic gait vs real
   ankle-IMU background (PAMAP2: walking, running, playing football).
2. **Timing and ordering** — onset precision vs rate; with the ball's own
   sensor, P(correct order) vs true separation.
3. **The anomalous overlap** — ball and opponent hit inside one pulse width; a
   two-template non-negative least-squares separator recovers order and gap.
4. **The probabilistic call** — knowledge-base base rates by situation + boot
   posterior + ball timing + separator order, fused (naive Bayes and stacked)
   into a calibrated P(ball first); cost-weighted decision with a review band.
5. **Where** — foot INS with zero-velocity updates and sparse UWB fixes;
   P(wrong side of the penalty-area line).

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/10_boot_contact
make test           # 12 tests: sensor chain, physics, detectors, onset, separator, fusion, INS (~5 s)
make run            # 5 experiments -> results/*.json (~2 min)
make figures        # 6 figures + paper/macros.tex
make paper          # compile paper/main.pdf
```
The real background comes from `data/pamap2_ankle.npz` (2.7 MB, checked in).
To rebuild it from the raw PAMAP2 archive set `PAMAP2_DIR=/path/to/PAMAP2_Dataset`.
Without either, the pipeline falls back to synthetic background and says so.

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
More windows, finer rate/gap sweeps, 20k decision cases, 200 localisation runs,
a third IMU grade — all config-gated, pure numpy/scipy/sklearn on CPU.

## Files
- `src/sim/{imu,gait,events,dataset}.py` — sensor model, running gait, ball/player/both events, window builder, clean templates
- `src/data/pamap2.py` — real ankle-IMU background loader + cache
- `src/detect/{features,threshold,ml,onset,separate}.py` — 16 pulse features, tuned threshold rule, boosting + isotonic calibration, onset (threshold, matched filter), two-template NNLS separator
- `src/decide/bayes.py` — naive-Bayes fusion, timing score, cost-weighted decision with review band, Brier/ECE
- `src/localize/ins.py` — 3-D foot trajectory, IMU synthesis, ZUPT + UWB Kalman INS, wrong-side probability
- `src/experiments/run_{detect,timing,overlap,decision,localize}.py` — write `results/*.json`
- `src/figures/plot_contact.py` — 6 figures + 109 LaTeX macros
- `paper/main.tex` — two-column paper (7 sections)
- `doc/` — overview, data, methodology, experiments log, results, decisions

## Headline
Boosting separates the four classes at macro-F1 ≈ 0.97 (0.91 on real
background) where the ball-style threshold rule gets 0.55–0.67. The ball onset
is timed to 0.5 ms at 1 kHz; with the ball's clock, ordering is ≥ 90 % correct
at 20 ms separation and chance below 3 ms. Inside the overlap the separator
recovers the order in 88 % of cases at 10 ms. Stacked fusion gives P(ball
first) with accuracy 0.944 and Brier 0.038 vs 0.667 / 0.218 for base rates
alone; a review band refers 5 % of cases and cuts field error to 3.4 %. ZUPT +
10 Hz UWB places the boot to 0.12 m — a penalty-area call is wrong 1.7 % of the
time at 25 cm from the line.
