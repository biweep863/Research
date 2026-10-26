"""Write routes: turn a model's raw output into a deliverable file and check it.

A *generation* is one model call (txt, md, json, html, tex, docx_code,
docx_xml, pptx_code, xlsx_code, csv). A *deliverable* is what the user wanted
in the end. Several deliverables share a generation: the three PDF routes
reuse the md, html and tex generations and add a build step. That is the
point: a PDF, DOCX, PPTX or XLSX is never written by the model directly.

Model-written code runs in a subprocess, in a scratch directory, with a
timeout. It is a local small model writing python-docx/pptx/openpyxl code.
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..data import renderers as R

DOC_TASK = ("Below is the content of a report. Convert it, keeping every sentence, number, name, "
            "the bullet list, the table and the equation.")
TABLE_TASK = ("Below is the content of a report. Extract only its table (header row plus every data row; "
              "numbers stay numbers).")

# generation -> (task, instruction, fenced language)
GENERATIONS: dict[str, tuple[str, str, str]] = {
    "txt": (DOC_TASK, "Write it as plain text with no markup at all.", "text"),
    "md": (DOC_TASK, "Write it as a Markdown document (headings, a bullet list, a pipe table).", "markdown"),
    "json": (DOC_TASK, 'Write it as one JSON object with keys "title", "author", "date" and "sections"; each '
             'section has "heading" and "content", a list of blocks with a "type" (paragraph, list, table or '
             'equation).', "json"),
    "html": (DOC_TASK, "Write it as a complete standalone HTML5 page (use <h1>, <h2>, <p>, <ul> and <table>).",
             "html"),
    "tex": (DOC_TASK, "Write it as a complete LaTeX document (\\documentclass{article} ... \\end{document}) that "
            "compiles with pdflatex.", "latex"),
    "docx_code": (DOC_TASK, "Write a Python script that uses the python-docx library to create a Word document "
                  "with this content. The variable OUTPUT_PATH is already defined; save the document to it.",
                  "python"),
    "docx_xml": (DOC_TASK, "Write the raw WordprocessingML for it: the complete contents of word/document.xml "
                 "(a <w:document> element with the w namespace declared) for a .docx file.", "xml"),
    "pptx_code": (DOC_TASK, "Write a Python script that uses the python-pptx library to create a slide deck: a "
                  "title slide, one slide per section, a slide for the list and a slide with the table. The "
                  "variable OUTPUT_PATH is already defined; save the presentation to it.", "python"),
    "xlsx_code": (TABLE_TASK, "Write a Python script that uses openpyxl to put the table in an Excel workbook. "
                  "The variable OUTPUT_PATH is already defined; save the workbook to it.", "python"),
    "csv": (TABLE_TASK, "Write it as CSV.", "csv"),
}

# deliverable -> generation it comes from; the order is the plotting order
DELIVERABLES = {
    "txt": "txt", "md": "md", "json": "json", "html": "html", "tex": "tex",
    "pdf_via_md": "md", "pdf_via_html": "html", "pdf_via_tex": "tex",
    "docx_code": "docx_code", "docx_xml": "docx_xml", "pptx_code": "pptx_code",
    "xlsx_code": "xlsx_code", "csv": "csv",
}
# the deliverable whose validity decides whether a generation is retried
RETRY_ON = {"txt": "txt", "md": "md", "json": "json", "html": "pdf_via_html", "tex": "pdf_via_tex",
            "docx_code": "docx_code", "docx_xml": "docx_xml", "pptx_code": "pptx_code",
            "xlsx_code": "xlsx_code", "csv": "csv"}
TABLE_GENERATIONS = {"xlsx_code", "csv"}


def prompt(gen: str, source: str) -> str:
    task, instr, lang = GENERATIONS[gen]
    return (f"{task} {instr}\nReply with only the result inside one ```{lang} code block.\n\n"
            f"--- REPORT ---\n{source}--- END ---")


_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)


def extract_payload(out: str) -> str:
    """The first fenced block (an unterminated one runs to the end), else the whole reply."""
    m = _FENCE.search(out)
    return (m.group(2) if m else out).strip("\n")


@dataclass
class Built:
    valid: bool
    text: str            # what fidelity is measured on
    build_s: float
    error: str = ""
    wrong_path: bool = False   # code saved the file somewhere other than OUTPUT_PATH


def _ok(text: str) -> Built:
    return Built(bool(text.strip()), text, 0.0, "" if text.strip() else "empty output")


def _json_strings(o) -> list[str]:
    if isinstance(o, str):
        return [o]
    if isinstance(o, dict):
        return [s for v in o.values() for s in _json_strings(v)]
    if isinstance(o, list):
        return [s for v in o for s in _json_strings(v)]
    return [str(o)] if o is not None else []


def run_code(code: str, out_path: Path, timeout: float) -> tuple[bool, str]:
    work = out_path.parent
    script = work / "gen_script.py"
    script.write_text(f"OUTPUT_PATH = r'{out_path}'\n" + code, encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, "-I", str(script)], cwd=work, capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if r.returncode != 0:
        return False, traceback_summary(r.stderr, script.name)
    if out_path.exists():
        return True, ""
    # saved under its own name (a common small-model slip): accept exactly one new file of the right type
    others = [p for p in work.glob(f"*{out_path.suffix}") if p != out_path]
    if len(others) == 1:
        others[0].replace(out_path)
        return True, "wrong_path"
    return False, "script ran but did not save to OUTPUT_PATH"


def traceback_summary(stderr: str, script_name: str) -> str:
    """The failing line of the model's own script plus the exception, as an agent would see it.

    The last stderr lines are usually inside the library; the model needs its own line.
    """
    lines = stderr.strip().splitlines()
    own = [i for i, l in enumerate(lines) if script_name in l and l.lstrip().startswith("File")]
    head = []
    if own:
        i = own[-1]
        head = [lines[i].strip().replace(lines[i].strip().split(",")[0] + ",", "In your script,")]
        if i + 1 < len(lines):
            head.append(lines[i + 1].strip())
    return "\n".join(head + lines[-1:])


_DOCX_TEMPLATE: Path | None = None


def _docx_template(work: Path) -> Path:
    global _DOCX_TEMPLATE
    if _DOCX_TEMPLATE is None or not _DOCX_TEMPLATE.exists():
        import docx
        _DOCX_TEMPLATE = work / "_template.docx"
        docx.Document().save(_DOCX_TEMPLATE)
    return _DOCX_TEMPLATE


def pack_docx_xml(xml: str, out_path: Path) -> None:
    """Swap word/document.xml of a blank python-docx package for the model's XML."""
    tpl = _docx_template(out_path.parent)
    with zipfile.ZipFile(tpl) as zi, zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zi.infolist():
            data = xml.encode("utf-8") if item.filename == "word/document.xml" else zi.read(item.filename)
            zo.writestr(item, data)


def build(deliv: str, payload: str, work: Path, timeout: float = 60) -> Built:
    """Build and validate one deliverable from the extracted model payload."""
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        b = _build(deliv, payload, work, timeout)
    except Exception as e:  # noqa: BLE001 - any failure of a model-written artifact is "invalid"
        b = Built(False, "", 0.0, f"{type(e).__name__}: {str(e)[:200]}")
    b.build_s = time.perf_counter() - t0
    return b


def _build(deliv: str, payload: str, work: Path, timeout: float) -> Built:
    if deliv in ("txt", "md"):
        return _ok(payload)
    if deliv == "json":
        return _ok("\n".join(_json_strings(json.loads(payload))))
    if deliv == "html":
        if not re.search(r"<(body|p|h1|table)\b", payload, re.I):
            return Built(False, "", 0, "no HTML body")
        return _ok(R.html_text(payload))
    if deliv in ("tex", "pdf_via_tex"):
        pdf, log = R.compile_tex(payload, work / "tex", timeout)
        return Built(True, R.pdf_text(pdf), 0) if pdf else Built(False, "", 0, log)
    if deliv in ("pdf_via_html", "pdf_via_md"):
        if deliv == "pdf_via_md":
            import markdown
            payload = markdown.markdown(payload, extensions=["tables"])
        pdf = work / f"{deliv}.pdf"
        if not R.html_to_pdf(payload, pdf):
            return Built(False, "", 0, "xhtml2pdf failed")
        t = R.pdf_text(pdf)
        return Built(bool(t.strip()), t, 0, "" if t.strip() else "empty pdf")
    if deliv in ("docx_code", "pptx_code", "xlsx_code"):
        ext = deliv.split("_")[0]
        out = work / f"out.{ext}"
        out.unlink(missing_ok=True)
        ok, err = run_code(payload, out, timeout)
        if not ok:
            return Built(False, "", 0, err)
        t = R.READERS[ext](out)
        return Built(bool(t.strip()), t, 0, "" if t.strip() else f"empty {ext}", wrong_path=err == "wrong_path")
    if deliv == "docx_xml":
        out = work / "out_xml.docx"
        pack_docx_xml(payload, out)
        t = R.docx_text(out)
        return Built(bool(t.strip()), t, 0, "" if t.strip() else "document has no text")
    if deliv == "csv":
        rows = [r for r in csv.reader(io.StringIO(payload)) if any(c.strip() for c in r)]
        if len(rows) < 2 or len({len(r) for r in rows}) != 1:
            return Built(False, "", 0, "ragged or empty CSV")
        return _ok("\n".join(",".join(r) for r in rows))
    raise KeyError(deliv)


def scratch(base: Path, *parts: str) -> Path:
    p = base.joinpath(*parts)
    shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True)
    return p
