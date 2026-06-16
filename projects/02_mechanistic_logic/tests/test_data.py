import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.synth_logic import generate

CFG = {"dataset": {"n_per_template": 4,
                   "templates": ["modus_ponens", "and_chain"],
                   "surface_forms": ["english", "conlang"]}}


def test_balanced():
    items = generate(CFG)
    n_valid = sum(1 for i in items if i.label == "valid")
    assert n_valid * 2 == len(items)


def test_surface_split():
    items = generate(CFG)
    surfaces = {i.surface for i in items}
    assert surfaces == {"english", "conlang"}


def test_prompts_nonempty():
    for it in generate(CFG):
        assert len(it.prompt) > 5
