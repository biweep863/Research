"""On-the-fly binaural mixtures on the GPU.

mix    = speech * BRIR_full(target)  +  sum_k g_k * noise_k * BRIR(noise dir k)
target = speech * BRIR_early(target)          (direct + early reflections)

SNR is the early-target power over the noise power at the ears, so late
reverberation of the talker counts as interference in every metric.
"""

from __future__ import annotations

import numpy as np
import torch

from .corpora import Pool


class Mixer:
    def __init__(self, speech: Pool, noise: Pool, bank: dict, cfg: dict, device):
        self.speech, self.noise, self.cfg = speech, noise, cfg
        self.fs = cfg["fs"]
        m = cfg["mix"]
        self.snr_db, self.level_db = m["snr_db"], m["level_dbfs"]
        self.pre = int(m["warmup_s"] * self.fs)
        self.device = device
        self.full = torch.as_tensor(bank["target_full"], device=device)
        self.early = torch.as_tensor(bank["target_early"], device=device)
        self.nbrir = torch.as_tensor(bank["noise"], device=device)
        self.n_rooms, self.K = self.nbrir.shape[:2]
        self.synthetic = speech.synthetic or noise.synthetic

    @torch.no_grad()
    def sample(self, batch: int, seg_len: int, rng: np.random.Generator, with_noise: bool = False) -> dict:
        n = self.pre + seg_len
        Lb = self.full.shape[-1]
        nfft = 1 << int(np.ceil(np.log2(n + Lb - 1)))
        dev = self.device
        sp = torch.as_tensor(np.stack([self.speech.crop(n, rng) for _ in range(batch)]), device=dev)
        nz = torch.as_tensor(
            np.stack([[self.noise.crop(n, rng) for _ in range(self.K)] for _ in range(batch)]), device=dev
        )
        r = torch.as_tensor(rng.integers(0, self.n_rooms, batch), device=dev)
        n_act = rng.integers(1, self.K + 1, batch)
        act = torch.as_tensor(np.arange(self.K)[None] < n_act[:, None], device=dev, dtype=torch.float32)
        g_db = torch.as_tensor(rng.uniform(-6, 6, (batch, self.K)), device=dev, dtype=torch.float32)
        snr = torch.as_tensor(rng.uniform(*self.snr_db, batch), device=dev, dtype=torch.float32)
        lvl = torch.as_tensor(rng.uniform(*self.level_db, batch), device=dev, dtype=torch.float32)

        def conv(x, h):  # x (..., n), h (..., 2, Lb) -> (..., 2, seg_len)
            y = torch.fft.irfft(torch.fft.rfft(x, nfft).unsqueeze(-2) * torch.fft.rfft(h, nfft), nfft)
            return y[..., self.pre : n]

        rev = conv(sp, self.full[r])
        tgt = conv(sp, self.early[r])
        nz = nz / nz.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        noi = conv(nz, self.nbrir[r])  # (B, K, 2, T)
        noi = (noi * (act * 10 ** (g_db / 20))[..., None, None]).sum(1)
        p_t = tgt.pow(2).mean((1, 2)).clamp_min(1e-10)
        p_n = noi.pow(2).mean((1, 2)).clamp_min(1e-10)
        noi = noi * torch.sqrt(p_t / p_n / 10 ** (snr / 10))[:, None, None]
        mix = rev + noi
        g = 10 ** (lvl / 20) / mix.pow(2).mean((1, 2)).sqrt().clamp_min(1e-8)
        g = g[:, None, None]
        out = {"mix": mix * g, "target": tgt * g, "snr_db": snr, "room": r}
        if with_noise:
            out["noise"] = noi * g
        return out


def fixed_set(mixer: Mixer, n: int, seg_len: int, seed: int, batch: int = 32) -> dict:
    """Deterministic evaluation set, returned on CPU."""
    rng = np.random.default_rng(seed)
    parts = [mixer.sample(min(batch, n - i), seg_len, rng) for i in range(0, n, batch)]
    return {k: torch.cat([p[k].cpu() for p in parts]) for k in parts[0]}
