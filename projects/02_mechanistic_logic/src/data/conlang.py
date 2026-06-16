"""An invented symbolic conlang vocabulary, designed so that surface tokens
are unlikely to appear in pretraining data while still tokenising into a small,
controllable set of subwords on standard BPE tokenisers.

The grammar is deliberately minimal:
    PRED(x,y) :=  <pred-word> ( <var> , <var> )
    AND  -> "tav"
    OR   -> "lus"
    NOT  -> "nyx"
    IMPL -> "drev"
    THEREFORE -> "qol"
    YES  -> "vir"
    NO   -> "kal"
"""
from __future__ import annotations
import random


PREDICATES = ["glip", "blarn", "thrux", "ymir", "kavd", "pelth", "snurf", "vrokk"]
VARIABLES = ["xa", "xb", "xc", "xd", "xe"]

TOKENS = {
    "and": "tav",
    "or": "lus",
    "not": "nyx",
    "impl": "drev",
    "therefore": "qol",
    "yes": "vir",
    "no": "kal",
}


def predicate(rng: random.Random) -> tuple[str, str, str]:
    p = rng.choice(PREDICATES)
    a, b = rng.sample(VARIABLES, 2)
    return p, a, b


def atom(rng: random.Random) -> str:
    p, a, b = predicate(rng)
    return f"{p}({a},{b})"
