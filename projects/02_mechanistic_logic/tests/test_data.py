import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.synth_logic import generate, _EN, _CL

CFG = {"dataset": {"n_per_template": 4,
                   "templates": ["modus_ponens", "and_chain"],
                   "surface_forms": ["english", "conlang"]}}

GPU_TEMPLATES = ["modus_ponens", "modus_tollens", "syllogism_barbara", "syllogism_celarent",
                 "syllogism_darii", "syllogism_ferio", "and_chain", "or_chain", "xor_chain", "not_chain"]


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


def test_all_gpu_full_templates_implemented():
    assert all(t in _EN and t in _CL for t in GPU_TEMPLATES)


def test_gpu_full_templates_generate():
    cfg = {"dataset": {"n_per_template": 2, "templates": GPU_TEMPLATES,
                       "surface_forms": ["english", "conlang"]}}
    items = generate(cfg)
    assert len(items) == 2 * len(GPU_TEMPLATES) * 2
    assert all(len(it.prompt) > 5 for it in items)
