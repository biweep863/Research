"""Synthetic logic dataset generator.

Each item is a pair (prompt, label) where label ∈ {"valid", "invalid"} and the
final token of the prompt is the model's continuation cue. The dataset
contains parallel English and conlang versions so we can test logical
content independently of natural-language surface form.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, asdict
from typing import List
from .conlang import atom, TOKENS, PREDICATES, VARIABLES


@dataclass
class LogicItem:
    template: str
    surface: str   # english | conlang
    prompt: str
    label: str     # "valid" or "invalid"
    answer_token: str   # the surface token expected for the truth value


def _english_atom(rng: random.Random) -> tuple[str, str]:
    # categorical proposition: "All Xs are Ys" etc.
    nouns = ["cats", "dogs", "rabbits", "lizards", "frogs", "owls"]
    a, b = rng.sample(nouns, 2)
    return a, b


# ---------- English templates ----------
def en_modus_ponens(rng, valid: bool) -> str:
    a, b = _english_atom(rng)
    x = rng.choice(["Alex", "Sam", "Pat", "Jamie"])
    concl = f"{x} is a {b[:-1]}" if valid else f"{x} is not a {b[:-1]}"
    return (
        f"If something is a {a[:-1]} then it is a {b[:-1]}. "
        f"{x} is a {a[:-1]}. Therefore, {concl}. This argument is"
    )


def en_modus_tollens(rng, valid: bool) -> str:
    a, b = _english_atom(rng)
    concl = f"it is not a {a[:-1]}" if valid else f"it is a {a[:-1]}"
    return (
        f"If something is a {a[:-1]} then it is a {b[:-1]}. "
        f"Something is not a {b[:-1]}. Therefore, {concl}. This argument is"
    )


def en_syllogism_barbara(rng, valid: bool) -> str:
    a, b, c = rng.sample(["birds", "fish", "mammals", "insects", "reptiles"], 3)
    concl = f"all {a} are {c}" if valid else f"some {a} are not {c}"
    return (
        f"All {a} are {b}. All {b} are {c}. Therefore, {concl}. This argument is"
    )


def en_and(rng, valid: bool) -> str:
    p, q = rng.sample(["it is raining", "the sun is shining", "the river is high"], 2)
    concl = p if valid else f"not {p}"
    return f"{p} and {q}. Therefore, {concl}. This argument is"


def en_or(rng, valid: bool) -> str:
    p, q = rng.sample(["the door is open", "the light is on"], 2)
    concl = q if valid else "neither holds"
    return f"{p} or {q}. {p[:0]}It is not the case that {p}. Therefore, {concl}. This argument is"


# ---------- Conlang templates ----------
def cl_modus_ponens(rng, valid: bool) -> str:
    a = atom(rng)
    b = atom(rng)
    concl = b if valid else f"{TOKENS['not']} {b}"
    return f"{a} {TOKENS['impl']} {b} . {a} . {TOKENS['therefore']} {concl} ."


def cl_modus_tollens(rng, valid: bool) -> str:
    a = atom(rng)
    b = atom(rng)
    concl = f"{TOKENS['not']} {a}" if valid else a
    return f"{a} {TOKENS['impl']} {b} . {TOKENS['not']} {b} . {TOKENS['therefore']} {concl} ."


def cl_syllogism_barbara(rng, valid: bool) -> str:
    a, b, c = atom(rng), atom(rng), atom(rng)
    concl = f"{a} {TOKENS['impl']} {c}" if valid else f"{a} {TOKENS['not']} {TOKENS['impl']} {c}"
    return f"{a} {TOKENS['impl']} {b} . {b} {TOKENS['impl']} {c} . {TOKENS['therefore']} {concl} ."


def cl_and(rng, valid: bool) -> str:
    a, b = atom(rng), atom(rng)
    concl = a if valid else f"{TOKENS['not']} {a}"
    return f"{a} {TOKENS['and']} {b} . {TOKENS['therefore']} {concl} ."


def cl_or(rng, valid: bool) -> str:
    a, b = atom(rng), atom(rng)
    concl = b if valid else f"{TOKENS['not']} {b}"
    return f"{a} {TOKENS['or']} {b} . {TOKENS['not']} {a} . {TOKENS['therefore']} {concl} ."


_EN = {
    "modus_ponens": en_modus_ponens,
    "modus_tollens": en_modus_tollens,
    "syllogism_barbara": en_syllogism_barbara,
    "and_chain": en_and,
    "or_chain": en_or,
}
_CL = {
    "modus_ponens": cl_modus_ponens,
    "modus_tollens": cl_modus_tollens,
    "syllogism_barbara": cl_syllogism_barbara,
    "and_chain": cl_and,
    "or_chain": cl_or,
}


def generate(cfg: dict, seed: int = 0) -> List[LogicItem]:
    rng = random.Random(seed)
    out: List[LogicItem] = []
    ds = cfg["dataset"]
    n = int(ds["n_per_template"])
    for tmpl in ds["templates"]:
        for surface in ds["surface_forms"]:
            for i in range(n):
                valid = (i % 2 == 0)
                if surface == "english":
                    prompt = _EN[tmpl](rng, valid)
                    answer = "valid" if valid else "invalid"
                else:
                    prompt = _CL[tmpl](rng, valid)
                    answer = TOKENS["yes"] if valid else TOKENS["no"]
                    # cue the model with " The argument is" -style continuation:
                    prompt = prompt + f" {TOKENS['therefore']}"
                out.append(LogicItem(template=tmpl, surface=surface,
                                     prompt=prompt,
                                     label="valid" if valid else "invalid",
                                     answer_token=answer))
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    import json
    cfg = {"dataset": {"n_per_template": 2,
                       "templates": ["modus_ponens", "and_chain"],
                       "surface_forms": ["english", "conlang"]}}
    for it in generate(cfg):
        print(json.dumps(asdict(it), ensure_ascii=False))
