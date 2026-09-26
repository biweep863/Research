import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace
import torch.nn as nn
from src.experiments.run_head_ablation import get_attn_modules


def _fake_gpt2(n_layers=2, n_heads=3, embed_dim=12):
    blocks = []
    for _ in range(n_layers):
        attn = SimpleNamespace(n_head=n_heads, embed_dim=embed_dim, c_proj=nn.Linear(embed_dim, embed_dim))
        blocks.append(SimpleNamespace(attn=attn))
    return SimpleNamespace(transformer=SimpleNamespace(h=blocks))


def _fake_gpt_neox(n_layers=2, n_heads=4, head_size=8):
    layers = []
    for _ in range(n_layers):
        attention = SimpleNamespace(head_size=head_size, dense=nn.Linear(n_heads * head_size, n_heads * head_size),
                                    config=SimpleNamespace(num_attention_heads=n_heads))
        layers.append(SimpleNamespace(attention=attention))
    return SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))


def test_get_attn_modules_gpt2():
    pairs = get_attn_modules(_fake_gpt2(n_layers=2, n_heads=3, embed_dim=12))
    assert len(pairs) == 2
    layer_i, head_dim, n_heads, module = pairs[0]
    assert (layer_i, head_dim, n_heads) == (0, 4, 3)
    assert isinstance(module, nn.Linear)


def test_get_attn_modules_gpt_neox():
    pairs = get_attn_modules(_fake_gpt_neox(n_layers=2, n_heads=4, head_size=8))
    assert len(pairs) == 2
    layer_i, head_dim, n_heads, module = pairs[0]
    assert (layer_i, head_dim, n_heads) == (0, 8, 4)
    assert isinstance(module, nn.Linear)
