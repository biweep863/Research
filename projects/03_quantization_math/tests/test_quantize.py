import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.experiments.quantize_models import quantize_tensor


def test_fp32_identity():
    w = torch.randn(8, 16)
    assert torch.allclose(quantize_tensor(w, 32), w)


def test_int4_close():
    torch.manual_seed(0)
    w = torch.randn(32, 32)
    wq = quantize_tensor(w, 4)
    assert wq.shape == w.shape
    # within reasonable error band
    err = ((w - wq) ** 2).mean().item()
    assert err < 0.05


def test_ternary_only_three_values_per_row():
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    wq = quantize_tensor(w, 2)
    for row in wq:
        uniq = torch.unique(row.abs())
        # at most 2 unique magnitudes (0 and scale)
        assert len(uniq) <= 2
