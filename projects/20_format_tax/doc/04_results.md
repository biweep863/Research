# 04 · Results (gpu_full)

Source: `results/full/*.json`, `results/full/write_summary.csv`, macros in
`paper/macros_full.tex`. Headline model: Qwen3-8B (NF4, no thinking).
18 documents for reading and understanding (2/4/8 sections). Writing used
6 documents for Qwen2.5-1.5B, 5 for Qwen3-8B (its 6th ran out of GPU
memory) and 2 for Phi-3.5-mini (stopped after 8 h; see `05_decisions.md`).

## Read: tokens (Qwen2.5 tokenizer, median over docs)

| Format | tokens/TXT | fixed + slope |
|---|---|---|
| TXT, DOCX/PDF/PPTX extracted text | 0.97–1.03 | — |
| Markdown | 1.01 | 4 + 1.01·x |
| XLSX cells | 1.17 | |
| LaTeX | 1.22 | 127 + 1.03·x |
| HTML | 1.40 | 200 + 1.10·x (mostly fixed) |
| JSON | 1.52 (GPT-2: ~2.8) | 73 + 1.41·x (grows with content) |
| raw XLSX / DOCX / PPTX XML | 4.26 / 4.70 / 7.65 | PPTX slope 4.65 |

Loaders are milliseconds (pypdf 22 ms).

## Understand: QA accuracy (%)

| Format | 1.5B full | 1.5B budget | 8B full | 8B budget |
|---|---|---|---|---|
| TXT | 79 | 79 | 91 | 91 |
| Markdown | 77 | 77 | 92 | 89 |
| JSON | 75 | 51 | 93 | 62 |
| HTML | 76 | 55 | 92 | 65 |
| LaTeX | 79 | 67 | 93 | 79 |
| PDF text | 77 | 73 | 92 | 87 |
| raw DOCX/PPTX/XLSX XML | 68/61/62 | 17/8/18 | 90/89/88 | 18/9/19 |

- At full length, format does not matter to the 8B model: 7-point
  spread, no format differs from Markdown after Holm correction (smallest
  adjusted p = 0.073). The 1.5B model loses 9–16 points on raw XML.
- At the TXT token budget, markup costs answers. **Markup dilution**:
  among questions whose answer is still inside the cut context, the 8B
  model answers 90 % from TXT/MD and 71 % from raw XML.

Phi-3.5-mini (full / budget): TXT 93/93, Markdown 92/85, JSON 93/56,
HTML 94/63, LaTeX 94/76, PDF text 92/87, raw DOCX/PPTX/XLSX XML
81/14, 76/8, 87/19.

## The memory cliff
Prefill is linear (1.2 ms/token for the 8B, R² 0.98) until the prompt
leaves GPU memory. That happens at 1,440 tokens for the 8B model and
3,624 for Phi-3.5-mini and 7,206 for the 1.5B. Then the time per token jumps up to 69×, because the
Windows driver spills to system RAM. The median raw-PPTX prompt took
314× as long as TXT to prefill, against a token ratio of 7.65×. The 8B
run's only out-of-memory crash was in SDPA asking for 10.8 GiB.

## Write: format tax (time per valid file / Markdown)

| Deliverable | 1.5B valid / tax / facts | 8B valid / tax / facts |
|---|---|---|
| Markdown, PDF via MD | 100 / 1.0 / 88 | 100 / 1.0 / 100 |
| JSON | 83 / 1.8 / 69 | 100 / 1.3 / 100 |
| HTML, PDF via HTML | 100 / 2.3 / 88 | 100 / 1.7 / 100 |
| LaTeX, PDF via LaTeX | 0 / ∞ | 100 (60 first try) / 2.5 / 100 |
| DOCX via python-docx | 33 / 6.5 / 15 | 100 (20 first try) / 5.2 / 100 |
| DOCX via raw XML | 0 / ∞ | 100 / 4.4 / 83 |
| PPTX via python-pptx | 50 / 4.0 / 20 | 40 / 31.6 / 36 |
| XLSX via openpyxl (table only) | 100 / 0.8 / 36 | 100 / 0.3 / 100 |
| CSV (table only) | 50 / 1.4 / 50 | 100 / 0.1 / 100 |

- **Every LaTeX failure is the `$`** in "Budget (k$)". The 8B model fixes
  it on retry. The 1.5B model resends the same table every time.
- **Retries**: the 1.5B model fixed 2 of 48 retries, and 37 repeated the
  identical error. The 8B model fixed 6 of 16.
- **python-pptx**: the 8B model writes `create_presentation(output_path)`
  and never calls it, three times in a row.
- **Raw XML vs code for DOCX (8B)**: raw XML is valid on the first try
  every time and costs 1.3× the tokens. python-docx needs repair 80 % of
  the time (`add_bullet_list` does not exist; tables with too few
  columns). Raw XML keeps fewer facts (83 % vs 100 %).

**Phi-3.5-mini (2 documents):** Markdown, TXT, JSON, HTML and the PDF
routes via MD/HTML were all valid. LaTeX was 0/2 (the `$`), PPTX 0/2
(`slide.placeholders[1]` on a layout without it), python-docx 1/2 at a tax
of 47×, raw XML 1/2, and that one was blank. Retries fixed 0 of 12.

## Answer to the question
- **To read:** extracted text or Markdown. Avoid raw office XML. Strip
  JSON/HTML when context is tight.
- **To write:** Markdown, and convert it (PDF via Markdown costs the
  same as Markdown). LaTeX costs 2.5× and fails on special characters.
  DOCX costs 4–6×, PPTX costs 4–30× and is often never valid.
