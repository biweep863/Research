import json

import pytest

from src.analysis import builders as B
from src.analysis.metrics import (density, fact_recall, first_number, format_tax, holm, linfit, mcnemar_p,
                                  normalize, score_answer, time_per_valid)
from src.data import renderers as R
from src.data.doc_generator import make_doc

DOC = make_doc(7)


# ---------------------------------------------------------------- metrics
def test_normalize_and_numbers():
    assert normalize("14,695 Litres.") == "14695 litres"
    assert normalize("y = 5.3 x") == "y 5.3 x"
    assert first_number("It was 47,783 litres") == "47783"
    assert first_number("none") is None


@pytest.mark.parametrize("pred,gold,ok", [
    ("120", "120", True), ("120 km of cable", "120", True), ("1,200", "1200", True),
    ("About 12", "120", False), ("Orka Lolen.", "Orka Lolen", True), ("orka", "Orka Lolen", False),
    ("The note was filed by Orka Lolen", "Orka Lolen", True),
    ("Orka Lolen " + "blah " * 40, "Orka Lolen", False),       # dumping text is not an answer
])
def test_score_answer(pred, gold, ok):
    assert score_answer(pred, gold) is ok


def test_score_answer_compare_distractor():
    gold, other = "Kadunnu observatory", "Beltaro harbour"
    assert score_answer("The Kadunnu observatory", gold, other)
    assert not score_answer("Kadunnu observatory or Beltaro harbour", gold, other)


def test_fact_recall_tolerates_pdf_splits():
    assert fact_recall("Equipment installed at T ortavel", ["Tortavel"]) == 1.0
    assert fact_recall("abc 47,783 litres", ["47783", "Orka"]) == 0.5


def test_density_tax_and_time():
    assert density(100, 400) == 0.25
    assert time_per_valid([2.0, 2.0], [True, False]) == 4.0
    assert time_per_valid([1.0], [False]) == float("inf")
    assert format_tax(6.0, 2.0) == 3.0


def test_holm_and_mcnemar():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert mcnemar_p([True] * 5, [True] * 5) == 1.0
    assert mcnemar_p([True] * 10, [False] * 10) < 0.01


def test_linfit_exact():
    a, b, r2 = linfit([0, 1, 2], [1, 3, 5])
    assert (a, b, r2) == pytest.approx((1, 2, 1))


# ---------------------------------------------------------------- generator
def test_generator_deterministic_and_answerable():
    assert make_doc(7).to_dict() == DOC.to_dict()
    txt = R.to_txt(DOC)
    for q in DOC.qa:
        if q.kind in ("fact_value", "fact_person", "table_cell", "list_item", "compare", "equation"):
            assert fact_recall(txt, [q.answer]) == 1.0, q


# ---------------------------------------------------------------- renderers round trip
@pytest.mark.parametrize("fmt", ["txt", "md", "html", "tex"])
def test_text_renderers_keep_facts(fmt):
    s = getattr(R, f"to_{fmt}")(DOC)
    if fmt == "html":
        s = R.html_text(s)
    facts = [x.lead for x in DOC.facts] + DOC.list_items
    assert fact_recall(s, facts) == 1.0


def test_json_renderer_parses():
    d = json.loads(R.to_json(DOC))
    assert d["title"] == DOC.title and len(d["sections"]) == len(DOC.sections)


@pytest.mark.parametrize("ext", ["docx", "pptx", "xlsx"])
def test_binary_round_trip(ext, tmp_path):
    p = getattr(R, f"write_{ext}")(DOC, tmp_path / f"d.{ext}")
    text = R.READERS[ext](p)
    assert fact_recall(text, [x.lead for x in DOC.facts] + [r[0] for r in DOC.table_rows]) == 1.0
    assert len(R.zip_xml(p, R.XML_PARTS[ext])) > len(text)   # raw XML is always heavier


# ---------------------------------------------------------------- write routes
def test_extract_payload():
    assert B.extract_payload("Sure!\n```python\nprint(1)\n```\nbye") == "print(1)"
    assert B.extract_payload("```csv\na,b\n1,2") == "a,b\n1,2"       # unterminated fence
    assert B.extract_payload("plain") == "plain"


def test_build_csv_json(tmp_path):
    assert B.build("csv", "a,b\n1,2", tmp_path).valid
    assert not B.build("csv", "a,b\n1", tmp_path).valid
    assert B.build("json", '{"t": ["x", {"y": "z"}]}', tmp_path).text == "x\nz"
    assert not B.build("json", "{bad", tmp_path).valid


def test_build_code_route(tmp_path):
    code = "import docx\nd = docx.Document()\nd.add_paragraph('Orka Lolen')\nd.save(OUTPUT_PATH)\n"
    b = B.build("docx_code", code, tmp_path)
    assert b.valid and "Orka Lolen" in b.text
    own = B.build("docx_code", code.replace("OUTPUT_PATH", "'mine.docx'"), tmp_path / "own")
    assert own.valid and own.wrong_path
    bad = B.build("docx_code", "raise ValueError('x')", tmp_path / "bad")
    assert not bad.valid and "ValueError" in bad.error


def test_build_docx_xml(tmp_path):
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
           '<w:body><w:p><w:r><w:t>Hello Orka</w:t></w:r></w:p></w:body></w:document>')
    assert B.build("docx_xml", xml, tmp_path).text == "Hello Orka"
    assert not B.build("docx_xml", "<w:document", tmp_path / "bad").valid


def test_every_generation_has_deliverables():
    assert set(B.DELIVERABLES.values()) == set(B.GENERATIONS) == set(B.RETRY_ON)
    assert all(B.DELIVERABLES[v] == k for k, v in B.RETRY_ON.items())


def test_traceback_summary_points_at_own_line():
    err = "\n".join([
        "Traceback (most recent call last):",
        '  File "C:\\w\\gen_script.py", line 12, in <module>',
        '    body = slide.placeholders["body"]',
        '  File "C:\\lib\\pptx\\shapes.py", line 40, in __getitem__',
        "    if idx < 0:",
        "TypeError: '<' not supported"])
    s = B.traceback_summary(err, "gen_script.py")
    assert "line 12" in s and 'placeholders["body"]' in s and s.endswith("not supported")
    assert "shapes.py" not in s
