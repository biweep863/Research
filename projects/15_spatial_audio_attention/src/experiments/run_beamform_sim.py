"""Selective spatial cancellation: SI-SDR gain of beamformers steered at the target.

Scene: target talker at ``target_deg`` + an interfering talker (different F0)
at ``interferer_deg`` with SIR ``sir_db`` + diffuse noise. The interferer
starts first, so the target-silent lead gives the MVDR its noise+interference
covariance - the same "learn from silence" rule the localizer uses, but here
"silence" means "the tracked target is not talking".

Metric: segmental-free output SINR. The *same* weights are applied to the
target image and to the residual (interferer + noise) separately, so the gain
is pure selectivity and is not penalised by the beamformer's linear
filtering of the target (which SI-SDR vs. mic 1 would count as distortion).
SI-SDR vs. the beamformed target image is also reported. Steering uses an *estimated* DOA
(not the oracle): SRP-PHAT on R_after_onset - R_before_onset, i.e. attention
goes to the talker that newly appeared, even if it is not the loudest.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..analysis.beamform import apply_weights, delay_and_sum_weights, mvdr_weights
from ..analysis.doa import cross_spectra, peak_azimuth, srp_phat
from ..analysis.localizer import Localizer
from ..analysis.metrics import angular_error_deg, si_sdr
from ..analysis.stft import istft, stft
from ..data.simulate import simulate_scene, voiced_speech
from .common import Timer, parse_config, write_json


def beam_weights(mix, lead_samples, loc, cfg):
    """Estimate target DOA by onset subtraction, return {name: (F, M) weights}."""
    fs, nfft, hop = loc.fs, loc.nfft, loc.hop
    freqs, X = stft(mix, fs, nfft, hop)
    n_lead = max(1, lead_samples // hop - 2)
    Rn = cross_spectra(X[:, :, :n_lead])
    Rx = cross_spectra(X[:, :, n_lead + 4 :])
    P = srp_phat(Rx - Rn, freqs, loc.mics, loc.grid, loc.band, loc.c)
    az_rad, _ = peak_azimuth(P, loc.grid)
    M = loc.mics.shape[0]
    ref = np.zeros((len(freqs), M), dtype=complex)
    ref[:, 0] = 1.0  # "beamformer" that just picks mic 1
    W = {
        "mic1": ref,
        "delay_and_sum": delay_and_sum_weights(loc.mics, az_rad, freqs, loc.c),
        "mvdr": mvdr_weights(Rn, loc.mics, az_rad, freqs, loc.c, cfg["beamform"]["loading"]),
    }
    return W, float(np.rad2deg(az_rad))


def _beam(W, x, loc):
    _, X = stft(x, loc.fs, loc.nfft, loc.hop)
    return istft(apply_weights(W, X), loc.fs, loc.nfft, loc.hop, x.shape[1])


def _db(x):
    return float(10 * np.log10(np.mean(x**2) + 1e-20))


def main():
    _, cfg = parse_config(__doc__)
    sw, sim = cfg["beam_sweep"], cfg["sim"]
    fs = int(cfg["audio"]["fs"])
    loc = Localizer(cfg)  # only for geometry/grid/band
    rows = []
    with Timer() as tm:
        for iaz, rt60, seed in itertools.product(sw["interferer_deg"], sw["rt60_s"], range(sw["seeds"])):
            lead = int(1.0 * fs)
            tgt = voiced_speech(sim["utterance_s"] * 2, fs, f0=120, seed=10 + seed)
            itf = voiced_speech(sim["utterance_s"] * 2 + 1.0, fs, f0=210, seed=50 + seed)
            L1 = sim["source_level_dbfs_1m"]
            sources = [
                {"signal": np.concatenate([np.zeros(lead), tgt]), "az_deg": sw["target_deg"],
                 "dist_m": sw["distance_m"], "level_dbfs_1m": L1},
                {"signal": itf, "az_deg": iaz, "dist_m": sw["distance_m"], "level_dbfs_1m": L1 - sw["sir_db"]},
            ]
            sc = simulate_scene(cfg, sources, rt60=rt60, noise_dbfs=sw["noise_dbfs"], seed=seed)
            Ws, est = beam_weights(sc["mix"], lead, loc, cfg)
            tgt_img = sc["images"][0]
            resid = sc["mix"] - tgt_img
            sinr, sdr = {}, {}
            for k, W in Ws.items():
                yt = _beam(W, tgt_img, loc)[lead:]
                yr = _beam(W, resid, loc)[lead:]
                sinr[k] = _db(yt) - _db(yr)
                sdr[k] = si_sdr(yt + yr, yt)
            row = {"interferer_deg": iaz, "rt60_s": rt60, "seed": seed, "est_target_deg": est,
                   "doa_err_deg": float(angular_error_deg(est, sw["target_deg"])),
                   "sinr_db": sinr, "si_sdr_db": sdr,
                   "gain_db": {k: sinr[k] - sinr["mic1"] for k in sinr if k != "mic1"}}
            rows.append(row)
            print(f"itf={iaz:3d} rt60={rt60:.1f} doa_est={est:6.1f} " +
                  " ".join(f"{k}={v:+.1f}dB" for k, v in row["gain_db"].items()))
    write_json(cfg, "beamform_sim.json", {"rows": rows, "wall_s": tm.s})


if __name__ == "__main__":
    main()
