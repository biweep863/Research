# 01 · Dataset

## Synthetic reports with planted facts
There is no public corpus that has the same document in TXT, MD, JSON, HTML,
LaTeX, PDF, DOCX, PPTX and XLSX *and* exact QA ground truth. So the corpus is
generated (`src/data/doc_generator.py`, seeded, deterministic).

Each document is an "annual report" of a fictional network:
- a title, an author and a date;
- `n_sections` sections. Each paragraph plants one **fact**: *In {month}, the
  {entity} recorded {value} {unit}, according to a note filed by {person}*,
  followed by neutral filler sentences;
- one equation (`y = a x + b`) in section 1;
- one bullet list (`list_len` items) after a random section;
- one table (unit, budget (k$), staff, score) after a random section.

All names are random syllable strings ("Kadunnu observatory", "Orka Lolen"),
so no answer can come from pretraining. Numbers are random within plausible
ranges and printed with thousands separators.

## Questions (about 12 per document)
| kind | example | answer |
|---|---|---|
| fact_value | How many km of cable did the Kadunnu observatory record? | 120 |
| fact_person | Who filed the note about the Grikabel laboratory? | Orka Lolen |
| table_cell | In the table, what is 'Score' for unit Taquinnu? | 94 |
| table_cell | Which unit has the most staff? | Grigrilen |
| list_count | How many items are in the list …? | 5 |
| list_item | What is item number 4 in the list …? | Isis filter |
| compare | Which was reported earlier in the year: A or B? | A |
| equation | What is the constant term added to the x term? | 25 |

Compare questions with tied months are dropped (they would be ambiguous).

## Renderings (`src/data/renderers.py`)
One content tree → every format, in the same block order:
- **Text formats** are written by hand-rolled renderers. Their markup is what
  a careful person would write. JSON is indented by 2, the common default.
- **Binary formats** are written by the standard libraries: python-docx,
  python-pptx (title slide, one slide per section, a list slide, a table
  slide) and openpyxl (a `Content` sheet with one row per block, plus a
  `Table` sheet). PDF is the LaTeX rendering compiled with pdflatex (MiKTeX).
- **What the model reads** for a binary file is either the loader's text
  (pypdf, python-docx, python-pptx, openpyxl → CSV-like) or the raw XML parts
  of the zip (`word/document.xml`; `ppt/slides/slide*.xml`; the sheet XML).

## Configurations
| | local_small | gpu_full |
|---|---|---|
| documents | 3 | 18 (6 per size) |
| sections per document | 3 | 2 / 4 / 8 (cycled: size sweep) |
| filler sentences per fact | 1 | 2 |
| documents in the write phase | 3 | 6 (first 6: 2 per size) |
| tokenizers (token counts) | Qwen2.5, Phi-3.5, GPT-2 | + Qwen3, Llama-3 |
| models (QA, writing) | Qwen2.5-1.5B-Instruct | + Phi-3.5-mini, Qwen3-8B (4-bit) |
| write retries | 1 | 2 |
| max new tokens (write) | 2048 | 4096 |

Rendered files are cached in `data/work*/` (git-ignored) and can be
regenerated exactly from the seed.

## License
Everything is synthetic and generated here. Model weights come from the
Hugging Face hub under their own licenses (Apache-2.0 for Qwen2.5 and
Qwen3, MIT for Phi-3.5). The Llama-3 tokenizer is used only to count tokens.
