"""Non-learned references on the same low-latency framing.

- ``oracle_mask``: ideal ratio mask computed from the true target. It bounds
  what *any* mask can do at a given window, so it isolates the
  frequency-resolution cost of a short window from learning error.
- ``BinauralWiener``: causal common-gain Wiener filter with decision-directed
  a-priori SNR (Ephraim & Malah 1984) and a minimum-tracking noise PSD. A
  common real gain keeps the interaural cues of everything it passes.
"""

from __future__ import annotations

import math

import torch

from .framing import asymmetric_windows, frame_signal, overlap_add


class _Wola:
    def __init__(self, n_fft: int, hop: int, device):
        ha, hs = asymmetric_windows(n_fft, hop)
        self.n_fft, self.hop = n_fft, hop
        self.ha = torch.tensor(ha, dtype=torch.float32, device=device)
        self.hs = torch.tensor(hs[-2 * hop :], dtype=torch.float32, device=device)

    def analysis(self, x):
        return torch.fft.rfft(frame_signal(x, self.n_fft, self.hop, extra_right=1) * self.ha, dim=-1)

    def synthesis(self, Y):
        seg = torch.fft.irfft(Y, n=self.n_fft, dim=-1)[..., -2 * self.hop :] * self.hs
        return overlap_add(seg, self.hop)


@torch.no_grad()
def oracle_mask(mix: torch.Tensor, target: torch.Tensor, n_fft: int, hop: int, kind: str = "ear"):
    """Ideal ratio mask sqrt(|S|^2 / (|S|^2 + |N|^2)), per ear or common to both ears."""
    w = _Wola(n_fft, hop, mix.device)
    X, S = w.analysis(mix), w.analysis(target)
    ps, pn = S.abs() ** 2, (X - S).abs() ** 2
    if kind == "common":
        ps, pn = ps.sum(1, keepdim=True), pn.sum(1, keepdim=True)
    m = torch.sqrt(ps / (ps + pn + 1e-12))
    return w.synthesis(m * X)


class BinauralWiener:
    """Frame-recursive, strictly causal; ``latency_samples`` = 2*hop."""

    def __init__(
        self,
        n_fft: int,
        hop: int,
        fs: int,
        tau_smooth_s: float = 0.02,
        tau_dd_s: float = 0.1,
        noise_rise_db_s: float = 3.0,
        gain_floor_db: float = -15.0,
    ):
        self.n_fft, self.hop, self.fs = n_fft, hop, fs
        self.a_s = math.exp(-hop / (fs * tau_smooth_s))
        self.a_dd = math.exp(-hop / (fs * tau_dd_s))
        self.rise = 10 ** (noise_rise_db_s * hop / fs / 10)
        self.g_min = 10 ** (gain_floor_db / 20)
        self.lookahead = 0

    @property
    def latency_samples(self) -> int:
        return 2 * self.hop

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = _Wola(self.n_fft, self.hop, x.device)
        X = w.analysis(x)  # (B, 2, Tf, F)
        P = (X.abs() ** 2).mean(1)  # (B, Tf, F) mean over ears
        B, Tf, Fq = P.shape
        G = torch.empty_like(P)
        p_s = P[:, 0].clone()
        lam = P[:, 0].clone() + 1e-10
        g_prev = torch.ones_like(p_s)
        gam_prev = torch.ones_like(p_s)
        for t in range(Tf):
            p_s = self.a_s * p_s + (1 - self.a_s) * P[:, t]
            lam = torch.minimum(p_s, lam * self.rise).clamp_min(1e-10)
            gam = P[:, t] / lam
            xi = self.a_dd * g_prev**2 * gam_prev + (1 - self.a_dd) * (gam - 1).clamp_min(0)
            g = (xi / (1 + xi)).clamp_min(self.g_min)
            G[:, t] = g
            g_prev, gam_prev = g, gam
        return w.synthesis(G.unsqueeze(1) * X)
