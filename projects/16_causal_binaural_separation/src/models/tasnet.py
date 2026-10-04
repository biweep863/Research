"""Binaural Conv-TasNet (Luo & Mesgarani 2019) for single-target enhancement.

Learned encoder/decoder with kernel L = 2*hop on the same causal framing as
the STFT model, so the algorithmic latency is again 2*hop samples. Both ears
share the encoder and decoder; the separator sees both encodings stacked.

Normalisation is a frame-wise LayerNorm over channels instead of the original
cumulative LayerNorm: cLN depends on every past frame since t=0, which breaks
bounded-state streaming (doc/05).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .framing import WolaState, frame_signal, overlap_add


class ChanNorm(nn.Module):
    """LayerNorm over channels of a (B, C, T) tensor, per time step."""

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TcnBlock(nn.Module):
    def __init__(self, b: int, h: int, p: int, dilation: int, causal: bool):
        super().__init__()
        self.pad = (p - 1) * dilation
        self.causal = causal
        self.in_conv = nn.Conv1d(b, h, 1)
        self.act1, self.norm1 = nn.PReLU(), ChanNorm(h)
        self.dconv = nn.Conv1d(h, h, p, dilation=dilation, groups=h)
        self.act2, self.norm2 = nn.PReLU(), ChanNorm(h)
        self.out_conv = nn.Conv1d(h, b, 1)

    def forward(self, x):
        y = self.norm1(self.act1(self.in_conv(x)))
        if self.causal:
            y = F.pad(y, (self.pad, 0))
        else:
            y = F.pad(y, (self.pad // 2, self.pad - self.pad // 2))
        y = self.norm2(self.act2(self.dconv(y)))
        return x + self.out_conv(y)

    def stream(self, x, buf):
        """x (B, b, 1); buf (B, h, pad) holds the last ``pad`` depthwise inputs."""
        y = self.norm1(self.act1(self.in_conv(x)))
        z = torch.cat([buf, y], dim=-1)
        new_buf = z[:, :, 1:]
        y = self.norm2(self.act2(self.dconv(z)))
        return x + self.out_conv(y), new_buf


class BinauralTasNet(nn.Module):
    def __init__(
        self,
        hop: int,
        n_filters: int = 128,
        bottleneck: int = 64,
        hidden: int = 128,
        kernel: int = 3,
        blocks: int = 6,
        repeats: int = 2,
        head: str = "ear",
        causal: bool = True,
    ):
        super().__init__()
        if head not in ("ear", "common"):
            raise ValueError(head)
        self.hop, self.head, self.causal = hop, head, causal
        self.win = 2 * hop
        self.lookahead = 0
        self.enc = nn.Linear(self.win, n_filters, bias=False)
        self.norm = ChanNorm(2 * n_filters)
        self.bott = nn.Conv1d(2 * n_filters, bottleneck, 1)
        self.tcn = nn.ModuleList(
            TcnBlock(bottleneck, hidden, kernel, 2**i, causal) for _ in range(repeats) for i in range(blocks)
        )
        self.n_masks = 2 if head == "ear" else 1
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(bottleneck, self.n_masks * n_filters, 1))
        self.dec = nn.Linear(n_filters, self.win, bias=False)
        self.n_filters = n_filters
        self._passthrough_init()

    @torch.no_grad()
    def _passthrough_init(self, seed: int = 0):
        """Start as (approximately) identity, like the STFT model.

        Encoder = [B; -B] * sqrt-Hann, so relu(Bx) - relu(-Bx) = Bx survives the
        ReLU; decoder = sqrt-Hann * [B+, -B+], so with mask m each frame returns m *
        hann * x and the hop-R overlap-add sums to x. Exact when n_filters/2 >=
        2*hop (B full column rank), a low-rank projection otherwise.
        """
        g = torch.Generator().manual_seed(seed)
        half = self.n_filters // 2
        # orthonormal columns: a square Gaussian basis is ill-conditioned and its pinv explodes (doc/05)
        if half >= self.win:
            B = torch.linalg.qr(torch.randn(half, self.win, generator=g))[0]
        else:
            B = torch.linalg.qr(torch.randn(self.win, half, generator=g))[0].T
        w = torch.sqrt(0.5 - 0.5 * torch.cos(2 * torch.pi * torch.arange(self.win) / self.win))
        Bp = torch.linalg.pinv(B)  # (win, half)
        self.enc.weight.copy_(torch.cat([B, -B]) * w[None])
        self.dec.weight.copy_(w[:, None] * torch.cat([Bp, -Bp], dim=1))
        self.mask[1].bias.fill_(1.0)  # mask 0.73; +4 saturated the sigmoid and stalled training (doc/05)
        self.mask[1].weight.mul_(0.1)

    @property
    def latency_samples(self) -> int:
        return self.win

    def _separate(self, w: torch.Tensor) -> torch.Tensor:
        """w (B, 2, Tf, N) encodings -> masked encodings (B, 2, Tf, N)."""
        B, _, Tf, N = w.shape
        z = self.bott(self.norm(w.permute(0, 1, 3, 2).reshape(B, 2 * N, Tf)))
        for blk in self.tcn:
            z = blk(z)
        m = torch.sigmoid(self.mask(z)).view(B, self.n_masks, N, Tf).permute(0, 1, 3, 2)
        return w * m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = frame_signal(x, self.win, self.hop, extra_right=1)  # (B, 2, Tf, 2R)
        w = F.relu(self.enc(frames))
        seg = self.dec(self._separate(w))
        return overlap_add(seg, self.hop)

    # ------------------------------------------------------------- streaming
    def init_state(self, batch: int, device) -> dict:
        if not self.causal:
            raise RuntimeError("non-causal reference cannot stream")
        return {
            "wola": WolaState((batch, 2), self.win, self.hop, device),
            "bufs": [torch.zeros(batch, blk.dconv.in_channels, blk.pad, device=device) for blk in self.tcn],
        }

    @torch.no_grad()
    def stream(self, chunk: torch.Tensor, state: dict) -> torch.Tensor:
        frame = state["wola"].push(chunk)  # (B, 2, 2R)
        w = F.relu(self.enc(frame))  # (B, 2, N)
        B, _, N = w.shape
        z = self.bott(self.norm(w.reshape(B, 2 * N, 1)))
        for i, blk in enumerate(self.tcn):
            z, state["bufs"][i] = blk.stream(z, state["bufs"][i])
        m = torch.sigmoid(self.mask(z)).view(B, self.n_masks, N)
        seg = self.dec(w * m)
        return state["wola"].synth(seg)
