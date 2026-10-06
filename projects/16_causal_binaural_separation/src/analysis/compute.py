"""Compute accounting for one streaming step: multiply-accumulates (MACs) of
the learned layers plus FFTs. Counted analytically from layer shapes, so the
number does not depend on the machine (wall-clock is measured separately)."""

from __future__ import annotations

import math

import torch.nn as nn


def fft_macs(n: int) -> float:
    """Real FFT of length n ~ (n/4) log2 n complex MACs ~ n log2 n real MACs."""
    return n * math.log2(n)


def layer_macs_per_step(model: nn.Module) -> float:
    """MACs of Linear / Conv1d / GRU layers for ONE frame of streaming."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.in_features * m.out_features
        elif isinstance(m, nn.Conv1d):
            total += m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]
        elif isinstance(m, nn.GRU):
            h = m.hidden_size
            dirs = 2 if m.bidirectional else 1
            for layer in range(m.num_layers):
                i = m.input_size if layer == 0 else h * dirs
                total += dirs * 3 * (i * h + h * h)
    return total


def macs_per_second(model: nn.Module, fs: int) -> float:
    """Per-step MACs (both ears) x frames per second."""
    hop = model.hop
    step = layer_macs_per_step(model)
    if hasattr(model, "n_fft"):  # STFT family: one analysis + one synthesis FFT per ear
        step += 2 * 2 * fft_macs(model.n_fft)
    else:  # TasNet encoder/decoder are applied per ear: enc/dec Linear counted once above
        enc = model.enc.in_features * model.enc.out_features
        dec = model.dec.in_features * model.dec.out_features
        step += enc + dec
    return step * fs / hop
