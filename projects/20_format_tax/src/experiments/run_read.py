"""Phase R: what each format costs before the model sees a single question.

- tokens per format under several tokenizers (no model weights needed);
- content density = tokens(plain text) / tokens(format);
- loader time: how long the pipeline takes to turn the file into the string
  the model reads (pypdf, python-docx, python-pptx, openpyxl, unzip).
"""
from __future__ import annotations

import time

import numpy as np

from ..analysis.metrics import density
from ..data import renderers as R
from .common import ROOT, corpus, rendered, setup, write

LOADERS = {
    "pdf_text": lambda p: R.pdf_text(p.with_suffix(".pdf")),
    "docx_text": lambda p: R.docx_text(p.with_suffix(".docx")),
    "pptx_text": lambda p: R.pptx_text(p.with_suffix(".pptx")),
    "xlsx_text": lambda p: R.xlsx_text(p.with_suffix(".xlsx")),
    "docx_xml": lambda p: R.zip_xml(p.with_suffix(".docx"), R.XML_PARTS["docx"]),
    "pptx_xml": lambda p: R.zip_xml(p.with_suffix(".pptx"), R.XML_PARTS["pptx"]),
    "xlsx_xml": lambda p: R.zip_xml(p.with_suffix(".xlsx"), R.XML_PARTS["xlsx"]),
}


def main():
    cfg, out, _ = setup("Phase R: token cost and loader time per format")
    from transformers import AutoTokenizer
    docs = corpus(cfg)
    rend = rendered(cfg, docs)
    rows = []
    toks = {t["short"]: AutoTokenizer.from_pretrained(t["name"]) for t in cfg["tokenizers"]}
    for d in docs:
        r = rend[d.doc_id]
        base = ROOT / cfg["work_dir"] / "files" / d.doc_id
        for fmt in R.READ_FORMATS:
            row = {"doc": d.doc_id, "format": fmt, "chars": len(r[fmt]), "bytes": len(r[fmt].encode("utf-8"))}
            for short, tk in toks.items():
                n = len(tk(r[fmt], add_special_tokens=False)["input_ids"])
                n_txt = len(tk(r["txt"], add_special_tokens=False)["input_ids"])
                row[f"tok_{short}"] = n
                row[f"density_{short}"] = density(n_txt, n)
            if fmt in LOADERS:
                ts = []
                for _ in range(cfg["read"]["loader_repeats"]):
                    t0 = time.perf_counter()
                    LOADERS[fmt](base)
                    ts.append(time.perf_counter() - t0)
                row["loader_s"] = float(np.median(ts))
            else:
                row["loader_s"] = 0.0
            rows.append(row)
        print(d.doc_id, {x["format"]: x[f"tok_{cfg['tokenizers'][0]['short']}"] for x in rows[-len(R.READ_FORMATS):]})
    write(out / "read.json", {"tokenizers": [t["short"] for t in cfg["tokenizers"]], "rows": rows})


if __name__ == "__main__":
    main()
