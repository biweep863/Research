"""Activation capture via forward hooks on HuggingFace causal LMs.

Returns hidden states for every transformer block layer in shape (batch, seq, hidden).
For HF models we just rely on `output_hidden_states=True`; this module exposes that
as a clean helper plus a manual-hook fallback for custom modules.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def hidden_states(model, tokenizer, text: str, max_length: int, device: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    # tuple of (n_layers+1) tensors (embedding + each block); use blocks 1..end
    hs = [h.detach().to("cpu").float() for h in out.hidden_states]
    return hs, enc["input_ids"].shape[1]
