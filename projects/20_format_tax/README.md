# Project 20 — The Format Tax: which file format should an LLM read and write?

## Question
For fixed content, how do TXT, Markdown, JSON, HTML, LaTeX, PDF, DOCX, PPTX
and XLSX/CSV differ in what an LLM pays to **read** them (tokens, prefill,
loader), how well it **understands** them (QA accuracy), and what it costs to
**write** a valid file of each type (tokens, time, validity, retries,
fidelity)?

## Answer
See `doc/04_results.md` (headline numbers) and `paper/main.pdf`.

## Quick start
```bash
cd projects/20_format_tax
make test        # unit tests: metrics, renderers, builders (~3 s)
make all         # read + understand + write on Qwen2.5-1.5B, figures, paper
```
Extra packages (in the main `.venv`): `python-docx python-pptx openpyxl pypdf
xhtml2pdf markdown sacrebleu bitsandbytes`. pdflatex (MiKTeX) must be on PATH.

## Scaling up
```bash
make run CONFIG=configs/gpu_full.yaml
make figures RESULTS=results/full FIGS=paper/figures_full MACROS=paper/macros_full.tex
```
The full run uses 18 documents of 2, 4 and 8 sections (a size sweep). It runs
three models: Qwen2.5-1.5B, Phi-3.5-mini and Qwen3-8B in NF4. The write phase
uses the first 6 documents (2 per size) with up to 3 attempts. It fits on a 12 GB RTX 3060.
`--models qwen3-8b` runs a single model.

## Files
- `src/data/doc_generator.py`: synthetic reports with planted facts and
  about 12 QA items each.
- `src/data/renderers.py`: one content tree → 12 read formats; loaders for
  PDF, DOCX, PPTX and XLSX; raw OOXML parts.
- `src/analysis/metrics.py`: answer scoring, fact recall, density, format
  tax, McNemar/Wilcoxon/Holm.
- `src/analysis/builders.py`: the 10 generation prompts and 13 deliverables.
  Builds files (pdflatex, xhtml2pdf, sandboxed python-docx/pptx/openpyxl
  scripts, raw XML → .docx) and validates them.
- `src/experiments/run_read.py`: tokens under several tokenizers, content
  density, loader time.
- `src/experiments/run_understand.py`: timed prefill + KV-cache-reused QA,
  full document vs the TXT token budget.
- `src/experiments/run_write.py`: generation → build → retry loop; time per
  valid file and fidelity.
- `src/figures/plot_results.py`: `paper/figures*/`, `paper/tables*/`,
  `paper/macros*.tex`.
