import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
from src.data.truthfulqa_loader import build_pair, load_pairs, _sample


class _StubTok:
    """Whitespace tokenizer: one token per word, ignoring add_special_tokens."""
    name_or_path = "stub"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


def test_build_pair_chooses_length_matched_answer():
    tok = _StubTok()
    pair = build_pair(
        question="What is the capital of France?",
        best_answer="Paris is the capital of France",
        correct_answers=["Paris is the capital of France"],
        incorrect_answers=["Lyon is the capital city", "Berlin is the capital of France"],
        tok=tok, max_len_diff=0,
    )
    assert pair is not None
    # "Berlin is the capital of France." has the same word count as the true answer
    assert "Berlin" in pair["false_cont"]
    assert pair["len_diff"] == 0


def test_build_pair_drops_pairs_outside_tolerance():
    tok = _StubTok()
    pair = build_pair(
        question="What is 2+2?",
        best_answer="Four",
        correct_answers=["Four"],
        incorrect_answers=["The answer is definitely and certainly five"],
        tok=tok, max_len_diff=0,
    )
    assert pair is None


def test_build_pair_excludes_no_comment():
    tok = _StubTok()
    pair = build_pair(
        question="What happens if you break a mirror?",
        best_answer="I have no comment.",
        correct_answers=["I have no comment.", "Nothing in particular happens"],
        incorrect_answers=["You will have seven years of bad luck"],
        tok=tok, max_len_diff=5,
    )
    assert pair is not None
    assert "no comment" not in pair["true_cont"].lower()


def test_sample_deterministic_with_seed():
    pairs = [{"i": i} for i in range(20)]
    a = _sample(pairs, 5, seed=42)
    b = _sample(pairs, 5, seed=42)
    assert a == b
    assert len(a) == 5


def test_load_pairs_caches_and_reuses(tmp_path, monkeypatch):
    tok = _StubTok()
    fake_rows = [
        {"question": "Q1?", "best_answer": "Yes it is true", "correct_answers": ["Yes it is true"],
         "incorrect_answers": ["No it is false"], "category": "test"},
        {"question": "Q2?", "best_answer": "Two words", "correct_answers": ["Two words"],
         "incorrect_answers": ["No"], "category": "test"},
    ]

    class _FakeDataset(list):
        pass

    def _fake_load_dataset(*a, **kw):
        return _FakeDataset(fake_rows)

    import src.data.truthfulqa_loader as mod
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset, raising=False)
    import sys as _sys
    fake_datasets = type(_sys)("datasets")
    fake_datasets.load_dataset = _fake_load_dataset
    monkeypatch.setitem(_sys.modules, "datasets", fake_datasets)

    cache = tmp_path / "pairs.json"
    src_cfg = {"n_pairs": -1, "max_len_diff_tokens": 0}
    pairs = mod.load_pairs(src_cfg, tok, seed=0, cache_path=cache)
    assert len(pairs) == 1  # only Q1 has a length-matched pair
    assert cache.exists()

    cached = json.loads(cache.read_text())
    assert cached["max_len_diff_tokens"] == 0
    # second call should reuse the cache without calling load_dataset again
    monkeypatch.setattr(fake_datasets, "load_dataset",
                         lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not re-download")))
    pairs2 = mod.load_pairs(src_cfg, tok, seed=0, cache_path=cache)
    assert pairs2 == pairs
