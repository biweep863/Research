# 03 · Experiments log

| Date | Config | Command | Outcome |
|---|---|---|---|
| 2026-10-24 | — | `make test` | 28 unit tests pass (~3 s): metrics, renderers' round trips, builders |
| 2026-10-24 | local_small | `make read` | 3 docs × 12 formats × 3 tokenizers, ~20 s. Raw XML costs 5–9× TXT tokens; extracted text costs ~1× |
| 2026-10-24 | local_small | `make understand` | Qwen2.5-1.5B, 3 docs × 12 formats × 2 conditions × 12 questions, ~3 min. Accuracy at full length 67–81% in every format; at the TXT budget, raw XML 0–8% |
| 2026-10-24 | local_small | `make write` (v1) | Stopped: python-docx code saved to `annual_report.docx`, not `OUTPUT_PATH`, and failed on retry too. Builders now accept exactly one file of the right type and flag `wrong_path` |
| 2026-10-24 | local_small | `make write` (v2) | Stopped after doc 1: the retry message was the library's last stderr lines and the model rewrote unrelated code. Now the model's own failing line is fed back. Doc 1 failures that are real: LaTeX `$`; raw XML = HTML tags inside `<w:document>` (valid XML, no `w:p`, empty document); CSV = prose instead of the table |
| 2026-10-24 | local + gpu_full | `run_all.sh` (scratchpad) | write pilot → figures → full read → understand/write for qwen25-1.5b, qwen3-8b, phi35-mini → full figures |
| 2026-10-24 | gpu_full | `run_read` | 18 docs × 12 formats × 5 tokenizers, 23 s |
| 2026-10-24 | gpu_full | `run_understand --models qwen25-1.5b` | 54 min. Full length 61–80 % per format; raw XML at the budget 8–18 % |
| 2026-10-24 | gpu_full | `run_write --models qwen25-1.5b` | 52 min, 6 docs. LaTeX 0/6 (every failure is the `$`); retries fixed 2 of 48 |
| 2026-10-25 | gpu_full | `run_understand --models qwen3-8b` | **8 h 40 min** (planned ~1 h). Prefill leaves GPU memory above ~1.4k tokens: 12 GB in use, driver spills to system RAM, up to 69× slower per token |
| 2026-10-25 | gpu_full | `run_write --models qwen3-8b` | 2 h 42 min, then **CUDA OOM** on doc0005 (8 sections), SDPA asking for 10.8 GiB. Results are saved per document, so docs 0–4 are kept |
| 2026-10-25 | gpu_full | `run_understand --models phi35-mini` | 3 h 35 min. 89–94 % on text and extracted forms, 76–87 % on raw XML |
| 2026-10-25 | gpu_full | `run_write --models phi35-mini` | **Stopped by decision at 22:40** after 8 h, during doc0002 (8 sections, 36 min for its JSON alone). Docs 0–1 kept. LaTeX 0/2 (`$`), PPTX 0/2, retries fixed 0 of 12 |
| 2026-10-25 | gpu_full | `plot_results --results results/full` | figures_full, tables_full, macros_full (headline qwen3-8b); `make paper` → 7 pages, no warnings |

## Observations
- **pdflatex failed on an unescaped `$`** in the table header "Budget (k$)"
  on both attempts of the first pilot document. The model did not understand
  the error message "Extra }, or forgotten $" well enough to fix it. Kept on
  purpose: see `05_decisions.md`.
- **The PDF text from pypdf is noisy** but still answers questions:
  "accord-\ning" hyphenation, page numbers mid-text, `\x88` for bullets,
  "T ortavel" (kerning split). Accuracy matched TXT in the pilot.
- **The 1.5B model rewrites while converting.** In python-docx code it
  shortened "according to a note filed by …" away. Fact recall measures
  that.
- **Prefill is not linear on a 12 GB card.** Below the cliff the 8B model
  costs ~1 ms/token. A raw-PPTX prompt of 7.2k tokens took 472 s. The
  analysis fits the linear rate only below the cliff
  (`plot_results.prefill_regimes`) and reports the cliff separately.
- **The 8B model's python-pptx script defines `create_presentation(output_path)`
  and never calls it.** The build says "script ran but did not save to
  OUTPUT_PATH". The model gets that message twice and does not add the call.
  This is checked in `data/work_full/write/qwen3-8b/doc0000/pptx_code/`.
- **Phi-3.5-mini's one "valid" raw-XML Word file is blank.** The text sits
  directly in `<w:p>` without `<w:r><w:t>`. The zip opens, python-docx reads
  only empty table cells, and fact recall is 0. It counts as valid, with a
  note in the paper.
