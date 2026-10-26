# 00 · Overview

## Question
For fixed content, how do file formats differ in what an LLM pays to **read**
them (tokens, prefill time, loader time), how well it **understands** them
(QA accuracy), and what it costs to **write** a valid file of each type
(output tokens, generation + build time, validity, retries, fidelity)?

Formats: TXT, Markdown, JSON, HTML, LaTeX, PDF, DOCX, PPTX, XLSX/CSV, the
files people most often hand to an LLM or ask it to make.

## Why this is not already answered
Pieces exist:
- table serialization for LLMs (Sui et al. 2024, *Table Meets LLM*);
- the effect of prompt formatting (He et al. 2024);
- reasoning under format restrictions (Tam et al. 2024, *Let Me Speak Freely?*).

What has not been measured publicly, as far as we know, is the **whole
lifecycle with the content held fixed**: the same planted facts rendered into
every format, read and written by the same model, timed end to end.

## Two points the study is built to expose
1. **Binary formats are never written directly.** An LLM cannot emit a DOCX,
   PDF, PPTX or XLSX. It writes python-docx/python-pptx/openpyxl code, raw
   OOXML, LaTeX, HTML or Markdown, and a builder makes the file. The *format
   tax* is the expected wall time per valid file (failed attempts and builds
   included), relative to Markdown.
2. **Reading and writing need not agree.** The cheapest format to read is not
   necessarily the cheapest to write. A fixed token budget (the size of the
   plain-text version) also tests whether markup *crowds out* content.

## Hypothesis (stated before the runs)
- Markdown is close to optimal for both reading and writing.
- LaTeX and HTML cost 1.3–2× the tokens of TXT with no accuracy gain.
- PDF text extraction costs QA accuracy on tables.
- Raw OOXML costs 5–10× more output than python-docx code.
- Binary deliverables carry a format tax above 3× Markdown.

**Falsified if** QA accuracy differs by less than 2 points across formats *and*
every token ratio is below 1.2×, i.e. format would not matter.

## Intuition
Every token of markup is paid three times. It is paid at prefill (time grows
linearly with tokens), in context budget (where it displaces content), and at
decode (where it is the slowest part: one forward pass per token). Builders add
a fourth cost that is binary rather than proportional: the file opens or it
does not.
