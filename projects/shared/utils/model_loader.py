"""Unified HuggingFace model loader supporting CPU / MPS / CUDA and optional dtype overrides."""
from __future__ import annotations
from dataclasses import dataclass
import torch


def best_device(prefer: str | None = None) -> str:
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def best_dtype(device: str, prefer: str | None = None) -> torch.dtype:
    if prefer:
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[prefer]
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float32  # mps fp16 is fragile for some ops
    return torch.float32


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    device: str
    dtype: torch.dtype


def load_hf(name: str, device: str = "auto", dtype: str | None = None,
            output_hidden_states: bool = True) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = best_device(device)
    dt = best_dtype(dev, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt)
    model.to(dev)
    model.eval()
    if output_hidden_states:
        model.config.output_hidden_states = True
    return LoadedModel(model=model, tokenizer=tok, device=dev, dtype=dt)
