# 03 · Experiments log

All runs use `configs/local_small.yaml` (seed 0) on a laptop CPU.

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-09-14 | local_small | `make test` | 12 passed, 4 s (after two onset fixes, see Observations) |
| 2026-09-14 | local_small | `run_detect` (35 s) | GB F1 0.936–0.969 sport, 0.879–0.961 consumer (66 % clipped), PAMAP2 0.908/0.874; threshold 0.547–0.693 |
| 2026-09-14 | local_small | `run_timing` (6 s) | ball onset 5.8 / 4.1 / 2.2 / 0.52 ms at 100/200/500/1000 Hz; ordering ≥ 0.9 at 20 ms (all rates), 0.88 at 10 ms @1 kHz |
| 2026-09-14 | local_small | `run_overlap` (7 s) | separator two-pulse ≥ 0.96; order acc 1.00/0.96/0.88/0.73/0.58 at ±20/15/10/6/3 ms; GB "both" 0.84–0.96 flat |
| 2026-09-14 | local_small | `run_decision` (34 s) | prior 0.667/0.218 → boot 0.809 → +ball 0.951/0.049 → naive all 0.927 → stacked 0.944/0.038, ECE 0.031 |
| 2026-09-14 | local_small | `run_localize` (25 s) | horizontal RMSE: free 3.3 m, ZUPT 0.77, UWB 1/5/10/20 Hz 0.27/0.15/0.12/0.09 m (sport) |
| 2026-09-14 | local_small | `make figures` (5 s) | 6 figures, 109 macros |
| 2026-09-14 | local_small | `make paper` | 7 pages, 1 overfull box, no undefined refs |

`make all` ≈ 2 min end to end.

## Observations
- **First onset estimator was wrong for player contacts.** Anchoring on the
  global |a| peak picked a *foot strike* (6–14 g) whenever the player pulse
  was faint (8 g), giving ~60 ms RMSE. Anchoring on the *horizontal* excess
  (strikes are vertical) cut gross errors from tens of percent to ~3 %; the
  remaining RMSE for the player is dominated by those few, so the paper
  reports the robust spread (2.1 ms at 1 kHz) alongside the RMSE.
- **A CUSUM onset was tried and dropped.** The gait background keeps a
  one-sided CUSUM positive for hundreds of ms, so it never resets before the
  contact; the matched filter on the rising edge is a cleaner second estimator
  and achieves sub-sample precision at 1 kHz (0.52 ms).
- **Template alignment bug.** `clean_templates` first returned pulses starting
  at 20 ms, biasing the matched filter by exactly 20 ms. Fixed by trimming to
  the onset; the unit test now pins the RMSE.
- **Ordering with the biggest pulse fails.** Using the global-peak onset as
  "boot contact time" made ordering collapse at large gaps (the ball pulse is
  always the biggest, so t_boot ≈ t_ball regardless of what came first). The
  boot's *first* onset from the separator fixes this; the empirical curve now
  follows the Φ(Δ/(σ√2)) theory once pulses are distinct.
- **Separator speed.** A Python double loop over (k₁,k₂) with `scipy.nnls`
  cost 0.12 s per window (220 s for the overlap sweep). The closed-form
  two-column NNLS vectorised with Gram matrices runs in < 1 ms per window.
- **Naive-Bayes fusion double counts.** Adding the separator to boot + ball
  *lowered* accuracy (0.951 → 0.927) because boot and separator read the same
  window. Stacked logistic fusion learned weights (1.68, 0.41, 1.29, 0.83) and
  gave the best Brier (0.038).
- **UWB phase artifact.** Scoring the position error at one contact instant
  made 2 Hz look better than 5 Hz (the last fix happened to be closer).
  Scoring over the second half of the run removed it; the curve is monotone.
- PAMAP2 has the soccer activity only for subjects 108 and 109 (optional
  protocol); subject 101's optional file has none, contrary to a first guess.
