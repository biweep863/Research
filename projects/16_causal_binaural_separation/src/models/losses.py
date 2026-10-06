"""Training losses (torch).

Per-ear SI-SDR lets each ear pick its own scale, which silently changes the
ILD; the reconstruction term is therefore a scale-*dependent* binaural SNR,
plus an explicit interaural-cue term (in the spirit of Tokala et al. 2024).
"""

from __future__ import annotations

import torch


def binaural_snr_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """-SNR in dB over both ears jointly. (B, 2, T) -> scalar."""
    num = (ref**2).sum(dim=(1, 2))
    den = ((ref - est) ** 2).sum(dim=(1, 2))
    return -(10 * torch.log10((num + 1e-8) / (den + 1e-8))).mean()


def cue_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    fs: int,
    n_fft: int = 512,
    hop: int = 128,
    dyn_range_db: float = 20.0,
    ipd_max_hz: float = 1500.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy-weighted |dILD| (dB) and (1 - cos dIPD) on target-active bins."""
    win = torch.hann_window(n_fft, device=est.device)
    B = est.shape[0]

    def spec(x):
        return torch.stft(x.reshape(-1, x.shape[-1]), n_fft, hop, window=win, return_complex=True).view(
            B, 2, n_fft // 2 + 1, -1
        )

    E, S = spec(est), spec(ref)
    pw = S.abs().pow(2).sum(1)  # (B, F, N)
    thr = pw.amax(dim=(1, 2), keepdim=True) * 10 ** (-dyn_range_db / 10)
    w = pw * (pw > thr)
    eps = 1e-6
    ild_e = 20 * torch.log10((E[:, 0].abs() + eps) / (E[:, 1].abs() + eps))
    ild_s = 20 * torch.log10((S[:, 0].abs() + eps) / (S[:, 1].abs() + eps))
    l_ild = (w * (ild_e - ild_s).abs()).sum() / (w.sum() + eps)
    ipd_e = torch.angle(E[:, 0] * E[:, 1].conj())
    ipd_s = torch.angle(S[:, 0] * S[:, 1].conj())
    f = torch.fft.rfftfreq(n_fft, 1 / fs, device=est.device)
    wl = w * (f <= ipd_max_hz)[None, :, None]
    l_ipd = (wl * (1 - torch.cos(ipd_e - ipd_s))).sum() / (wl.sum() + eps)
    return l_ild, l_ipd


def total_loss(est, ref, fs, cue_weight: float = 1.0, ipd_weight: float = 10.0):
    l_snr = binaural_snr_loss(est, ref)
    if cue_weight == 0:
        return l_snr, {"snr": -l_snr.item()}
    l_ild, l_ipd = cue_loss(est, ref, fs)
    loss = l_snr + cue_weight * (l_ild + ipd_weight * l_ipd)
    return loss, {"snr": -l_snr.item(), "ild": l_ild.item(), "ipd": l_ipd.item()}
