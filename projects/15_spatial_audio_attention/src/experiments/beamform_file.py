"""Selective spatial cancellation on a recording: steer at the tracked talker.

    python -m src.experiments.beamform_file data/recordings/<clip>.wav [--deepfilter]

Writes next to ``--out``:
  <name>_ch0.wav     firmware processed channel (what home2 uses today)
  <name>_mic1.wav    raw mic 1
  <name>_ds.wav      delay-and-sum toward the tracked talker
  <name>_mvdr.wav    MVDR toward the tracked talker; noise covariance learned
                     on VAD-silent frames (the silence model)
  <name>_mvdr_df.wav MVDR -> DeepFilterNet3 (only with --deepfilter and the
                     `deepfilternet` package installed), i.e. the proposed
                     home2 chain: spatial filter, then the existing DF node.
Listen and compare; there is no reference signal on real recordings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from ..analysis.beamform import apply_weights, delay_and_sum_weights, mvdr_weights
from ..analysis.localizer import analyze
from ..analysis.stft import ema_alpha, istft, stft
from ..data.respeaker_io import load_wav, save_wav, select_mics


def steer_track(frames, n_stft, hop, nfft, fs):
    """Tracked azimuth (rad, array frame) and speech flag per STFT frame."""
    t_loc = np.array([f.t for f in frames])  # end time of each localizer frame
    az = np.array([f.track_az_deg for f in frames])
    sp = np.array([f.is_speech for f in frames])
    valid = np.isfinite(az)
    if not valid.any():
        raise SystemExit("no talker was tracked in this file")
    # hold last valid track (and the first valid one backwards)
    idx = np.where(valid, np.arange(len(az)), 0)
    np.maximum.accumulate(idx, out=idx)
    az = az[idx]
    az[: np.argmax(valid)] = az[np.argmax(valid)]
    t_stft = np.arange(n_stft) * hop / fs + nfft / (2 * fs)  # centre-aligned frames
    k = np.clip(np.searchsorted(t_loc, t_stft), 0, len(frames) - 1)
    return np.deg2rad(az[k]), sp[k]


def run_mvdr(X, freqs, mics, az, speech, alpha, loading, c, update_every=8):
    M, F, N = X.shape
    Rn = np.tile(np.eye(M, dtype=complex) * 1e-8, (F, 1, 1))
    Y = np.zeros((F, N), dtype=complex)
    W = None
    for n in range(N):
        if not speech[n]:
            x = X[:, :, n]
            Rn = (1 - alpha) * Rn + alpha * np.einsum("if,jf->fij", x, x.conj())
        if W is None or n % update_every == 0:
            W = mvdr_weights(Rn, mics, az[n], freqs, c, loading)
        Y[:, n] = np.einsum("fm,mf->f", W.conj(), X[:, :, n])
    return Y


def deepfilter(y, fs):
    import scipy.signal
    import torch
    from df.enhance import enhance, init_df

    model, state, _ = init_df(log_level="none")
    peak = np.abs(y).max() + 1e-12
    y48 = scipy.signal.resample_poly(y / peak, 3, 1).astype(np.float32)
    with torch.no_grad():
        out = enhance(model, state, torch.from_numpy(y48)[None], atten_lim_db=12.0)
    return scipy.signal.resample_poly(out.cpu().numpy().squeeze(), 1, 3) * peak


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--config", default="configs/local_small.yaml")
    ap.add_argument("--out", default="data/beamformed")
    ap.add_argument("--deepfilter", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    a = cfg["audio"]
    x, fs = load_wav(args.wav)
    mics_x = select_mics(x, a["mic_channels"])
    loc, frames = analyze(mics_x, cfg)
    nfft, hop = loc.nfft, loc.hop
    freqs, X = stft(mics_x, fs, nfft, hop)
    az, speech = steer_track(frames, X.shape[-1], hop, nfft, fs)
    print(f"tracked azimuth (array frame): median {np.rad2deg(np.median(az)):.1f} deg, "
          f"{speech.mean() * 100:.0f}% speech frames")

    name = Path(args.wav).stem
    out = Path(args.out)
    T = mics_x.shape[1]
    if x.shape[0] >= 6:
        save_wav(out / f"{name}_ch0.wav", x[0], fs)
    save_wav(out / f"{name}_mic1.wav", mics_x[0], fs)
    # delay-and-sum with time-varying steering
    Y = np.zeros(X.shape[1:], dtype=complex)
    for n in range(X.shape[-1]):
        W = delay_and_sum_weights(loc.mics, az[n], freqs, loc.c)
        Y[:, n] = np.einsum("fm,mf->f", W.conj(), X[:, :, n])
    save_wav(out / f"{name}_ds.wav", istft(Y, fs, nfft, hop, T), fs)
    alpha = ema_alpha(a["noise_tau_s"], hop, fs)
    y_mvdr = istft(run_mvdr(X, freqs, loc.mics, az, speech, alpha, cfg["beamform"]["loading"], loc.c), fs, nfft, hop, T)
    save_wav(out / f"{name}_mvdr.wav", y_mvdr, fs)
    if args.deepfilter:
        try:
            save_wav(out / f"{name}_mvdr_df.wav", deepfilter(y_mvdr, fs), fs)
        except ImportError as e:
            print(f"DeepFilterNet not available ({e}); pip install deepfilternet")
    print(f"wrote {out}/{name}_*.wav")


if __name__ == "__main__":
    main()
