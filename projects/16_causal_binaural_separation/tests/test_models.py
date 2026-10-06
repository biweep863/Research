"""Causality, streaming equivalence and latency accounting for every model family."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.models.classical import BinauralWiener, oracle_mask
from src.models.factory import build_model, hop_for
from src.models.framing import asymmetric_windows, frame_signal, overlap_add
from src.models.losses import total_loss

FS = 16000
CFG = {
    "fs": FS,
    "model": {
        "stft": {"analysis_ms": 32, "hidden": 32, "layers": 2, "df_order": 3},
        "tasnet": {"n_filters": 32, "bottleneck": 16, "hidden": 32, "kernel": 3, "blocks": 3, "repeats": 2},
    },
}
SPECS = [
    {"family": "stft_asym", "latency_ms": 4, "head": "ear"},
    {"family": "stft_asym", "latency_ms": 2, "head": "common"},
    {"family": "stft_sym", "latency_ms": 4, "head": "ear"},
    {"family": "stft_asym", "latency_ms": 8, "head": "ear", "lookahead": 2},
    {"family": "tasnet", "latency_ms": 2, "head": "ear"},
    {"family": "tasnet", "latency_ms": 4, "head": "common"},
]


def _randomise(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return model.eval()


@pytest.mark.parametrize("n,hop", [(64, 32), (512, 32), (512, 8), (256, 128)])
def test_wola_perfect_reconstruction(n, hop):
    ha, hs = asymmetric_windows(n, hop)
    x = torch.randn(1, 2, 40 * hop, dtype=torch.float64)
    frames = frame_signal(x, n, hop, extra_right=1) * torch.tensor(ha)
    seg = frames[..., -2 * hop :] * torch.tensor(hs[-2 * hop :])
    assert np.allclose(hs[: n - 2 * hop], 0)
    y = overlap_add(seg, hop)
    assert torch.allclose(y, x, atol=1e-10)


def test_hop_for_latency_budget():
    assert hop_for(4, FS) == 32
    assert hop_for(8, FS, lookahead=2) == 32
    with pytest.raises(ValueError):
        hop_for(1, FS, lookahead=1)  # 16 samples is not 3 whole hops


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s['family']}-{s['latency_ms']}-{s['head']}-{s.get('lookahead', 0)}")
def test_streaming_equals_offline_and_latency(spec):
    m = _randomise(build_model(spec, CFG))
    hop = m.hop
    assert m.latency_samples == round(spec["latency_ms"] * FS / 1000)
    x = torch.randn(2, 2, 60 * hop)
    with torch.no_grad():
        y = m(x)
    st = m.init_state(2, "cpu")
    ys = torch.cat([m.stream(x[..., i : i + hop], st) for i in range(0, x.shape[-1], hop)], dim=-1)
    delay = m.latency_samples - hop  # input buffering adds the remaining hop
    assert torch.allclose(ys[..., delay:], y[..., : y.shape[-1] - delay], atol=1e-4), (
        (ys[..., delay:] - y[..., : y.shape[-1] - delay]).abs().max()
    )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s['family']}-{s['latency_ms']}-{s['head']}-{s.get('lookahead', 0)}")
def test_future_does_not_leak(spec):
    """Output sample n may depend on input up to n + lookahead*hop + hop - 1 only."""
    m = _randomise(build_model(spec, CFG))
    hop = m.hop
    x = torch.randn(1, 2, 50 * hop)
    cut = 30 * hop
    x2 = x.clone()
    x2[..., cut:] = torch.randn_like(x2[..., cut:])
    with torch.no_grad():
        y1, y2 = m(x), m(x2)
    safe = cut - m.lookahead * hop - hop  # last sample whose frames all end before ``cut``
    assert torch.allclose(y1[..., :safe], y2[..., :safe], atol=1e-5)
    assert not torch.allclose(y1[..., safe + hop : safe + 3 * hop], y2[..., safe + hop : safe + 3 * hop])


def test_noncausal_reference_does_leak():
    spec = {"family": "stft_asym", "latency_ms": 4, "head": "ear", "causal": False}
    m = _randomise(build_model(spec, CFG))
    x = torch.randn(1, 2, 50 * m.hop)
    x2 = x.clone()
    x2[..., 40 * m.hop :] = 0
    with torch.no_grad():
        assert not torch.allclose(m(x)[..., : 10 * m.hop], m(x2)[..., : 10 * m.hop])


def test_oracle_and_wiener_shapes_and_identity():
    x = torch.randn(2, 2, 64 * 32)
    y = oracle_mask(x, x, 512, 32)  # target == mixture -> mask 1 -> identity
    assert torch.allclose(y, x, atol=1e-4)
    w = BinauralWiener(512, 32, FS)
    assert w(x).shape == x.shape and w.latency_samples == 64


def test_wiener_is_causal():
    w = BinauralWiener(512, 32, FS)
    x = torch.randn(1, 2, 64 * 32)
    x2 = x.clone()
    x2[..., 40 * 32 :] = 0
    assert torch.allclose(w(x)[..., : 39 * 32], w(x2)[..., : 39 * 32], atol=1e-5)


def test_loss_rewards_target_and_cues():
    torch.manual_seed(0)
    s = torch.randn(2, 2, 8000)
    perfect, _ = total_loss(s, s, FS)
    swapped, parts = total_loss(s.flip(1), s, FS)
    assert perfect < swapped and parts["ild"] > 0


@pytest.mark.parametrize("lat", [2, 4])
def test_tasnet_starts_as_passthrough(lat):
    m = build_model({"family": "tasnet", "latency_ms": lat, "head": "ear"}, CFG).eval()
    m2 = build_model({"family": "tasnet", "latency_ms": lat, "head": "ear"},
                     {**CFG, "model": {**CFG["model"], "tasnet": {**CFG["model"]["tasnet"], "n_filters": 2 * 2 * m.hop + 2}}}).eval()
    x = torch.randn(1, 2, 40 * m.hop)
    with torch.no_grad():
        y = m2(x)
    # full-rank basis: output = sigmoid(mask bias) * input up to the mask's small random part
    err = (y - torch.sigmoid(torch.tensor(1.0)) * x).pow(2).sum() / x.pow(2).sum()
    assert err < 0.05, err


def test_mac_count_matches_hand_count():
    from src.analysis.compute import layer_macs_per_step, macs_per_second

    m = build_model({"family": "stft_asym", "latency_ms": 4, "head": "common"}, CFG)
    F_ = m.n_freq
    h = 32
    hand = 4 * F_ * h + 2 * 3 * (h * h + h * h) + h * F_  # input linear + 2 GRU layers + gain head
    assert layer_macs_per_step(m) == hand
    assert macs_per_second(m, FS) == (hand + 4 * m.n_fft * np.log2(m.n_fft)) * FS / m.hop
