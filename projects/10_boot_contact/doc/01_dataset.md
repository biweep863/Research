# 01 · Data — a physics simulator plus a real ankle-IMU background

There is no public recording of instrumented fouls (boot IMU + labelled
contacts). The data are therefore generated, with one real component.

## Synthetic boot signal (`src/sim/`)
Generated at `sim_rate` = 4 kHz in the boot frame (x forward, y lateral, z sole
normal), then passed through the sensor model and decimated.

- **Gait** (`gait.py`): running stride at 2.4–3.2 strides/s; smooth swing
  acceleration; vertical half-sine foot strike (6–14 g, 25–40 ms) with a
  braking x component; foot pitch rotates gravity into the sensor frame; gyro
  pitch rate with a slap at foot strike.
- **Ball** (`events.py`): half-sine deceleration along −x, peak 50–130 g,
  duration 7–12 ms (Nunome 2006), then a damped 150–260 Hz ringing at 18 % of
  the peak; gyro pitch arrested.
- **Player**: oblique half-sine (direction with lateral/vertical spread), peak
  8–45 g, duration 18–55 ms, followed by a 0.5–3 g push over 60–160 ms; lateral
  gyro twist.
- **Both**: ball + player with a signed gap `t_player − t_ball` (config range
  ±40 ms; negative = player first).
- **Ball sensor**: reports an impact time only when the ball is struck,
  quantised to a 500 Hz grid with a 1 ms clock offset (N(0,1 ms)).

## Sensor model (`imu.py`)
First-order low-pass at 45 % of Nyquist → bias (N(0, 0.02 g)) → white noise
scaled by √(fs/1000) → clipping at full scale → decimation. Two parts:
`sport` (±200 g, 0.08 g noise) and `consumer` (±16 g, 0.03 g noise).

## Real background: PAMAP2 (`src/data/pamap2.py`)
Reiss & Stricker 2012, UCI ML repository (CC-BY-4.0). Colibri IMU on the
dominant ankle at 100 Hz, ±16 g accelerometer + gyro. We use walking (4),
running (5) and — from the optional protocol of subjects 108 and 109 — playing
soccer (20). Subjects 101, 105, 106, 108, 109 → 10 (subject, activity)
segments capped at 120 s each → `data/pamap2_ankle.npz` (2.7 MB, checked in).
The raw 688 MB archive is not checked in; if present at `data/PAMAP2_Dataset`
or `$PAMAP2_DIR` the cache is rebuilt from it. If neither exists, experiments
fall back to synthetic background and say so in `results/detect.json`.

Real background is a random 0.6 s crop linearly upsampled to `sim_rate`, with
the synthetic event added on top. Since it contains nothing above 50 Hz it is
only used at its native 100 Hz (`detect.pamap2_fs`).

## Windows
0.6 s per window, event onset uniformly in the middle 30–60 %, six channels
[ax ay az (g), gx gy gz (deg/s)]. Local config: 300 windows/class/condition
(70/30 train/test), 400 timing events per condition, 120 windows per overlap
gap, 1500 decision cases (half for calibration), 20 localisation runs.
`configs/gpu_full.yaml` multiplies these by ~7–13× and adds finer sweeps.
