"""Post-training quantization wrappers for HuggingFace causal LMs.

Implements a uniform, in-place, per-channel symmetric quantizer over the
weights of every Linear / Conv1D layer:

    bits=32 → identity
    bits=16 → cast to fp16
    bits=8  → round to int8 with per-channel scale
    bits=4  → round to int4 with per-channel scale
    bits=2  → ternary {-1, 0, 1} with per-channel scale (1.58-bit proxy)

This is *not* a state-of-the-art quantizer; it is a deterministic, dependency-
free baseline that runs anywhere PyTorch does. GPTQ / AWQ / true BitNet are
gated behind the gpu_full.yaml config.
"""
from __future__ import annotations
import copy
import torch
import torch.nn as nn


def _per_channel_scale(W, qmax):
    # W: (out, in); per-row scale
    max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    return max_abs / qmax


def quantize_tensor(W: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return W
    if bits == 16:
        return W.to(torch.float16).to(W.dtype)
    if bits == 8:
        qmax = 127
        scale = _per_channel_scale(W, qmax)
        Wq = (W / scale).round().clamp(-qmax, qmax)
        return Wq * scale
    if bits == 4:
        qmax = 7
        scale = _per_channel_scale(W, qmax)
        Wq = (W / scale).round().clamp(-qmax, qmax)
        return Wq * scale
    if bits == 2:
        # ternary: per-row threshold = 0.7 * mean(|W|)
        thresh = 0.7 * W.abs().mean(dim=1, keepdim=True)
        sign = torch.sign(W)
        keep = (W.abs() > thresh).to(W.dtype)
        # scale: mean magnitude of kept weights, per row
        scale = ((W.abs() * keep).sum(dim=1, keepdim=True) /
                 (keep.sum(dim=1, keepdim=True).clamp(min=1.0)))
        return sign * keep * scale
    raise ValueError(f"unsupported bits: {bits}")


def _module_matches(name: str, kinds):
    return any(k in name for k in kinds)


def quantize_model_(model, bits: int, only_modules=None):
    """In-place quantization of weights for all Linear / Conv1D layers,
    optionally restricted to names containing any of `only_modules`.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if not hasattr(module, "weight"):
                continue
            if only_modules is not None and not _module_matches(name, only_modules):
                continue
            cls = module.__class__.__name__
            if cls not in {"Linear", "Conv1D"}:
                continue
            W = module.weight.data
            module.weight.data = quantize_tensor(W, bits).to(W.dtype)
    return model


def clone_and_quantize(model, bits: int, only_modules=None):
    cp = copy.deepcopy(model)
    return quantize_model_(cp, bits, only_modules=only_modules)
