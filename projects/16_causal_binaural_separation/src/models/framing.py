"""Low-latency framing shared by every model in the project.

Frame ``t`` (hop ``R``, analysis length ``N``) covers input samples
``[(t+1)R - N, (t+1)R)``: it *ends* at the newest sample, so nothing in it
comes from the future. Synthesis only uses the last ``2R`` samples of each
frame, and overlap-adding those segments with hop ``R`` gives

    algorithmic latency = 2R samples  (+ D*R with D frames of lookahead).

With ``N > 2R`` the analysis window is asymmetric (Mauler & Martin 2007;
Wang, Wichern, Watanabe & Le Roux 2023): a long rising half gives good frequency
resolution, a short falling half keeps the latency at 2R. ``N == 2R`` is the
ordinary symmetric sqrt-Hann WOLA.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _hann_periodic(m: int) -> np.ndarray:
    n = np.arange(m)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / m)


def asymmetric_windows(n: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Analysis/synthesis windows with ``h_a * h_s`` = periodic Hann(2R) on the
    last 2R samples (COLA at hop R) and ``h_s`` = 0 elsewhere."""
    r = hop
    if n < 2 * r or n % r:
        raise ValueError(f"need n >= 2*hop and n % hop == 0, got n={n}, hop={hop}")
    h_long = _hann_periodic(2 * (n - r))
    h_short = _hann_periodic(2 * r)
    ha = np.zeros(n)
    ha[: n - r] = np.sqrt(h_long[: n - r])
    ha[n - r :] = np.sqrt(h_short[r:])
    hs = np.zeros(n)
    idx = np.arange(n - 2 * r, n - r)
    den = np.sqrt(h_long[idx])
    num = h_short[: r]
    hs[idx] = np.where(den > 1e-8, num / np.maximum(den, 1e-8), 0.0)
    hs[n - r :] = np.sqrt(h_short[r:])
    return ha, hs


def frame_signal(x: torch.Tensor, n: int, hop: int, extra_right: int = 0) -> torch.Tensor:
    """(..., T) -> (..., T//hop + extra_right, n) causal frames (see module doc).

    ``extra_right`` frames of zero padding at the end are needed offline to
    flush the overlap-add and any lookahead.
    """
    if x.shape[-1] % hop:
        raise ValueError("signal length must be a multiple of hop")
    xp = F.pad(x, (n - hop, extra_right * hop))
    return xp.unfold(-1, n, hop)


def overlap_add(seg: torch.Tensor, hop: int) -> torch.Tensor:
    """(..., n_frames, 2R) synthesis segments -> (..., (n_frames-1)*R) samples.

    Segment t covers samples [(t-1)R, (t+1)R); block t = seg[t][R:] + seg[t+1][:R].
    The last block would need frame n_frames, so it is dropped.
    """
    r = hop
    blocks = seg[..., :-1, r:] + seg[..., 1:, :r]
    return blocks.reshape(*blocks.shape[:-2], -1)


class WolaState:
    """Streaming buffers for one WOLA analysis/synthesis chain."""

    def __init__(self, batch_shape: tuple, n: int, hop: int, device, dtype=torch.float32):
        self.inbuf = torch.zeros(*batch_shape, n - hop, device=device, dtype=dtype)
        self.tail = torch.zeros(*batch_shape, hop, device=device, dtype=dtype)

    def push(self, chunk: torch.Tensor) -> torch.Tensor:
        """Append R new samples, return the newest full frame (..., n)."""
        frame = torch.cat([self.inbuf, chunk], dim=-1)
        self.inbuf = frame[..., chunk.shape[-1] :]
        return frame

    def synth(self, seg: torch.Tensor) -> torch.Tensor:
        """Overlap-add one 2R segment, return the R finished samples."""
        r = self.tail.shape[-1]
        out = self.tail + seg[..., :r]
        self.tail = seg[..., r:]
        return out
