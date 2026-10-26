# 02 · Methodology

## Pipeline
```
doc_generator ──► content tree + QA ──► renderers ──► 12 read formats (cached)
                                                         │
   run_read ◄────────────────────────────────────────────┤  tokens × tokenizers, loader time
   run_understand ◄──────────────────────────────────────┤  prefill (timed) → KV cache → QA (greedy)
   run_write ◄── TXT source ── 10 generation prompts ──► model ──► payload ──► builders
                                                                       │  (pdflatex, xhtml2pdf,
                                                                       │   python script, zip)
                                                                       ▼
                                                     valid? ── no ──► error fed back (retry)
                                                       │ yes
                                                       ▼
                                          readers → text → fact recall, chrF
results/*.json ──► src/figures/plot_results.py ──► paper/figures/*.pdf, paper/tables/*.tex, paper/macros.tex
```

## Phase R: read cost
- **Tokens** per format under each tokenizer, without special tokens.
- **Content density** = tokens(TXT) / tokens(format): the share of tokens
  that carry content.
- **Markup cost model** (gpu_full, size sweep): a least-squares fit of
  tokens(format) = a + b · tokens(TXT). Here *a* is the fixed cost
  (namespaces, preamble, slide masters) and *b* is the cost per content token.
- **Loader time**: median over 3 runs of turning the file into the string the
  model reads.

## Phase U: understanding
Prompt: *Read the document below and answer the question. Reply with only
the answer.* The document sits in `<document>` tags and the question follows
it. All of a document's questions share the same prompt prefix. That prefix is
**prefilled once and timed** (CUDA-synchronized, median of 3), and every
question decodes from a deep copy of the KV cache. Decoding is greedy, with 24
new tokens.

**Scoring** (`metrics.score_answer`):
- For a numeric gold, the first number on the first answer line must match it
  (thousands separators are ignored).
- For a text gold, its normalized words must appear contiguously, and the
  answer must be at most max(15, 3 × gold) words long.
- For a compare item, the answer must also not name the other entity.

**Conditions:**
- `native`: the whole document.
- `budget`: the format truncated to tokens(TXT). This is the same context
  budget as plain text. `evidence_in_context` records whether the gold string
  survived the truncation. Accuracy restricted to those items separates
  "content was cut" from "markup confuses".

**Statistics:** McNemar exact test, paired by (doc, question), each format
against Markdown, Holm-adjusted; bootstrap 95% CIs.

## Phase W: writing
The source is the TXT rendering. There are ten generation prompts
(`builders.GENERATIONS`). Each asks for the result in one fenced block. The
three code routes get a predefined `OUTPUT_PATH`.

| deliverable | generation | build / validity |
|---|---|---|
| txt, md | txt, md | non-empty |
| json | json | `json.loads` |
| html | html | has a body/p/table tag |
| tex, pdf_via_tex | tex | pdflatex exits 0 |
| pdf_via_html | html | xhtml2pdf renders |
| pdf_via_md | md | python-markdown → xhtml2pdf |
| docx_code, pptx_code, xlsx_code | python code | runs in a subprocess (60 s), file opens |
| docx_xml | raw `word/document.xml` | packed into a blank .docx; python-docx opens it |
| csv | csv (table only) | ≥ 2 rows, constant column count |

A failed build is fed back as a new user turn ("That did not work: <error>.
Fix it …") until the attempt limit is reached.

- **time per valid file** = Σ(generation + build time, all attempts, all
  docs) / #valid.
- **format tax** = time per valid file / the same quantity for Markdown.
- **fidelity**: the built file is read back with the same loaders as Phase R.
  Fact recall is computed over every planted string (values, people,
  entities, list items, table row names, the equation constant; for
  spreadsheets, every table cell). chrF is computed against the source.
