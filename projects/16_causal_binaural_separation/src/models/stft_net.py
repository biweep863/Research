"""Binaural STFT-domain enhancer with asymmetric windows (``AsymDF``).

Per frame: compressed complex spectra of both ears -> frame-wise LayerNorm ->
linear -> GRU (unidirectional = causal, bidirectional = the non-causal
reference) -> one of two output heads:

- ``common``: one real gain per bin, applied to both ears. Interaural cues of
  everything that passes are preserved by construction.
- ``ear``: per-ear complex deep filter over the current and ``df_order-1``
  *past* frames (DeepFilterNet-style, Schroeter et al. 2022).

``lookahead`` = D frames: the network output for frame t is computed after it
has seen frame t+D (costs D*hop extra samples of latency).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .framing import WolaState, asymmetric_windows, frame_signal, overlap_add


def _compress(X: torch.Tensor, power: float = 0.3) -> torch.Tensor:
    mag = X.abs().clamp_min(1e-8)
    return X * mag ** (power - 1)


class StftEnhancer(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop: int,
        hidden: int = 128,
        layers: int = 2,
        df_order: int = 3,
        head: str = "ear",
        causal: bool = True,
        lookahead: int = 0,
    ):
        super().__init__()
        if head not in ("ear", "common"):
            raise ValueError(head)
        if not causal and lookahead:
            raise ValueError("lookahead only makes sense for a causal model")
        self.n_fft, self.hop, self.head = n_fft, hop, head
        self.causal, self.lookahead = causal, lookahead
        self.df_order = df_order if head == "ear" else 1
        self.n_freq = n_fft // 2 + 1
        ha, hs = asymmetric_windows(n_fft, hop)
        self.register_buffer("win_a", torch.tensor(ha, dtype=torch.float32))
        self.register_buffer("win_s", torch.tensor(hs, dtype=torch.float32))
        n_in = 4 * self.n_freq
        self.norm = nn.LayerNorm(n_in)
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(
            hidden, hidden if causal else hidden // 2, num_layers=layers,
            batch_first=True, bidirectional=not causal,
        )
        n_out = self.n_freq if head == "common" else 2 * self.n_freq * self.df_order * 2
        self.out = nn.Linear(hidden, n_out)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():  # start as "pass-through": gain 1 / DF tap0 = 1
            if head == "common":
                self.out.bias.fill_(1.0)  # gain 0.73: a +3 bias saturated the sigmoid and stalled training (doc/05)
            else:
                b = torch.zeros(2, self.df_order, self.n_freq, 2)
                b[:, 0, :, 0] = 1.0
                self.out.bias.copy_(b.flatten())

    # ----------------------------------------------------------------- shape
    @property
    def latency_samples(self) -> int:
        return (2 + self.lookahead) * self.hop

    def _features(self, X: torch.Tensor) -> torch.Tensor:
        """X (B, 2, Tf, F) complex -> (B, Tf, 4F)."""
        c = _compress(X)
        f = torch.cat([c.real, c.imag], dim=-1)  # (B, 2, Tf, 2F)
        f = f.permute(0, 2, 1, 3).flatten(2)
        return self.norm(f)

    def _apply_head(self, o: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """o (B, Tf, n_out), X (B, 2, Tf, F) aligned frame-by-frame -> Y like X."""
        if self.head == "common":
            g = torch.sigmoid(o).unsqueeze(1)  # (B, 1, Tf, F)
            return g * X
        B, Tf, _ = o.shape
        w = o.view(B, Tf, 2, self.df_order, self.n_freq, 2)
        w = torch.complex(w[..., 0], w[..., 1]).permute(0, 2, 3, 1, 4)  # (B, 2, K, Tf, F)
        Y = torch.zeros_like(X)
        for k in range(self.df_order):
            Xk = F.pad(X, (0, 0, k, 0))[:, :, :Tf] if k else X  # X[t-k]
            Y = Y + w[:, :, k] * Xk
        return Y

    # --------------------------------------------------------------- offline
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 2, T), T multiple of hop -> (B, 2, T) time-aligned estimate."""
        D = self.lookahead
        frames = frame_signal(x, self.n_fft, self.hop, extra_right=1 + D)  # (B, 2, Tf, N)
        X = torch.fft.rfft(frames * self.win_a, dim=-1)
        o, _ = self.gru(self.inp(self._features(X)))
        o = self.out(o)
        if D:
            o, X = o[:, D:], X[:, :, : X.shape[2] - D]
        Y = self._apply_head(o, X)
        seg = torch.fft.irfft(Y, n=self.n_fft, dim=-1)[..., -2 * self.hop :] * self.win_s[-2 * self.hop :]
        return overlap_add(seg, self.hop)

    # ------------------------------------------------------------- streaming
    def init_state(self, batch: int, device) -> dict:
        if not self.causal:
            raise RuntimeError("non-causal reference cannot stream")
        return {
            "wola": WolaState((batch, 2), self.n_fft, self.hop, device),
            "h": None,
            "X": torch.zeros(batch, 2, self.df_order + self.lookahead, self.n_freq,
                             dtype=torch.complex64, device=device),
        }

    @torch.no_grad()
    def stream(self, chunk: torch.Tensor, state: dict) -> torch.Tensor:
        """chunk (B, 2, hop) -> (B, 2, hop). Output is the offline output delayed
        by (1 + lookahead) * hop samples (total latency 2+D hops incl. buffering)."""
        frame = state["wola"].push(chunk)
        Xt = torch.fft.rfft(frame * self.win_a, dim=-1).unsqueeze(2)  # (B, 2, 1, F)
        o, state["h"] = self.gru(self.inp(self._features(Xt)), state["h"])
        o = self.out(o)
        Xbuf = torch.cat([state["X"][:, :, 1:], Xt], dim=2)  # oldest .. newest
        state["X"] = Xbuf
        D, K = self.lookahead, self.df_order
        past = Xbuf[:, :, :K]  # frames t-D-K+1 .. t-D; the last one is enhanced
        o = o[:, 0]
        if self.head == "common":
            Y = torch.sigmoid(o).unsqueeze(1) * past[:, :, -1]
        else:
            w = o.view(o.shape[0], 2, K, self.n_freq, 2)
            w = torch.complex(w[..., 0], w[..., 1])
            Y = sum(w[:, :, k] * past[:, :, K - 1 - k] for k in range(K))
        seg = torch.fft.irfft(Y, n=self.n_fft, dim=-1)[..., -2 * self.hop :]
        return state["wola"].synth(seg * self.win_s[-2 * self.hop :])
