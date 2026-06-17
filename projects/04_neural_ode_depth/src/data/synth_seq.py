"""Synthetic sequence tasks for training a tiny transformer / Neural ODE
transformer.  Two flavours:

- copy: input is `BOS x1 x2 ... xn SEP`; target is `x1 x2 ... xn EOS`.
- sort: input is the multiset; target is the sorted sequence.

We use a small vocab and short sequences so the model trains in a few minutes
on CPU.
"""
from __future__ import annotations
import torch


PAD, BOS, SEP, EOS = 0, 1, 2, 3
SPECIAL = 4


def make_batch(task: str, batch_size: int, seq_len: int, vocab_size: int,
               device: str = "cpu", seed: int | None = None):
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(seed)
    body = torch.randint(SPECIAL, vocab_size, (batch_size, seq_len), generator=g)
    if task == "sort":
        targets = body.sort(dim=-1).values
    elif task == "copy":
        targets = body.clone()
    else:
        raise ValueError(task)
    bos = torch.full((batch_size, 1), BOS, dtype=torch.long)
    sep = torch.full((batch_size, 1), SEP, dtype=torch.long)
    eos = torch.full((batch_size, 1), EOS, dtype=torch.long)
    x = torch.cat([bos, body, sep, targets, eos], dim=1)  # full sequence
    # next-token prediction loss target = x shifted left
    return x.to(device)
