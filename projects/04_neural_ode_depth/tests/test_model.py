import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.model import NODETransformer
from src.data.synth_seq import make_batch


def test_continuous_forward_shape():
    m = NODETransformer(vocab_size=16, d_model=32, n_heads=4, max_len=40)
    x = make_batch("copy", 2, 8, 16, device="cpu", seed=0)
    logits, nfe = m.forward_continuous(x[:, :-1], solver="rk4", step_size=0.5)
    assert logits.shape == (2, x.size(1) - 1, 16)
    assert nfe > 0


def test_discrete_forward_shape():
    m = NODETransformer(vocab_size=16, d_model=32, n_heads=4, max_len=40)
    x = make_batch("copy", 2, 8, 16, device="cpu", seed=0)
    logits, nfe = m.forward_discrete(x[:, :-1], depth=3)
    assert logits.shape == (2, x.size(1) - 1, 16)
    assert nfe == 3
