# 05 · Decisions & alternatives

## Synthetic documents, not real ones
Real document collections exist (arXiv sources, Common Crawl HTML, enterprise
DOCX), but none comes with the same document in nine formats and exact
answers. Converting real documents with pandoc would make the converter part
of the treatment. A generator holds content fixed by construction, and its
made-up names stop the model from answering from memory. The price is
ecological validity: the reports are short and regular. The size sweep
(2/4/8 sections) partly addresses length.

## Local models only
The user chose local models (RTX 3060, 12 GB, no API keys). The benefits:
exact prefill and decode timing, greedy decoding, and full reproducibility.
The cost: no frontier model. The three models span 1.5B–8B and two families
(Qwen, Phi). Qwen3-8B runs in NF4 4-bit to fit in memory, with thinking
disabled so the time is spent on the answer.

## Prefill once, answer many (KV-cache reuse)
Recomputing the document for every question would multiply Phase U time by
about 12. The shared prefix is found by token comparison, not by string
splitting, so chat-template boundaries cannot shift it. The cached prefill
time is also the read latency measurement.

## Same-budget condition (truncate to tokens(TXT))
A context limit is where markup hurts in practice. Truncating each format to
the plain text's own token count asks: *with the budget for the plain text,
what do you lose by sending format X?* `evidence_in_context` is a substring
check of the gold answer. It is approximate for short numeric answers (a
number can appear elsewhere), which biases it towards "present".

## Raw XML read as the XML *parts*, not the whole zip
`[Content_Types].xml`, relationships and styles carry no content. Reading
only `word/document.xml`, the slide XML or the sheet XML is the **cheapest**
raw-XML reading, a lower bound. Real Word files are heavier than python-docx
output (rsids, revision marks, `w14` attributes).

openpyxl writes inline strings, not the shared-string table that Excel uses.
Excel's indirection (cell → index → `sharedStrings.xml`) would make raw XLSX
harder still.

## Writing starts from TXT
Starting from TXT means the TXT route is a near-copy. It stays in as the
lower bound. The Markdown route, the baseline of the format tax, is a real
conversion. Starting from Markdown would have made Markdown free.

## PDF through three routes
An LLM can make a PDF only through something in between, so the PDF
deliverable is measured for each route: LaTeX + pdflatex, HTML + xhtml2pdf,
and Markdown + python-markdown + xhtml2pdf. pandoc and wkhtmltopdf are not
installed; xhtml2pdf is pure Python and adds no system dependency.

## Spreadsheets are table-only
A spreadsheet is the natural home of the table, not of the prose. The XLSX
and CSV routes ask for the table only, and their fidelity is the share of
table cells recovered. Their tax is still relative to the Markdown document,
so it is conservative: they write less.

## The `$` in "Budget (k$)" stays
In the pilot, the 1.5B model's LaTeX failed on the unescaped `$`. Removing
it would have hidden a real cost of LaTeX: content with special characters
(`$ % & # _`) must be escaped. Financial reports have dollars, so it stays.

## Running model-written code
Code routes execute model output with the venv's Python (`-I`), in a
scratch directory, with a 60 s timeout. The models are local and the prompts
are ours, so the risk is accidental (an infinite loop, a stray file), not
adversarial. A container would be safer but would add start-up time to every
build and distort build times.

## One retry locally, two at full scale
Retries are what an agent loop does. They turn "valid on the first try" into
"valid eventually" and put the price in time. Two retries is a common agent
default. Each retry resends the whole conversation, so its cost grows
quickly.

## What the retry sees: the model's own failing line
The first harness fed back the last 3 stderr lines. For python-pptx those
were inside the library (`if idx < 0 …  TypeError`). The 1.5B model then
"fixed" it by writing unrelated code (a `MyClass` with test cases). An
agent shell shows the full traceback, so the retry message now gives the
frame in the model's own script (line number + source line) plus the
exception (`builders.traceback_summary`). LaTeX gets the first `!` error
of the log with 3 lines of context; xhtml2pdf and the CSV and XML checks get
a one-line reason.

## A file saved under the wrong name still counts (flagged)
Code routes predefine `OUTPUT_PATH`. Small models often ignore it and save
`annual_report.docx`. Counting that as invalid would measure path-following,
not format ability. If exactly one file of the right type appears, it is
accepted and the row gets `wrong_path = true`, so the rate is still
reported.

## Keep the memory-cliff timings, fit the rate below it
The cliff comes from the platform (12 GB, SDPA without a fused kernel for
these shapes, the Windows driver's fallback to system memory), not from the
format. We did not rerun with a smaller cap, because the cliff is what a
local user actually gets. Markup is what moves a document across it. The
paper reports the linear rate on the points below the cliff, the cliff's
position per model, and the end-to-end ratio, and names the stack.

## Qwen3-8B writes 5 documents, not 6
The 6th write document (8 sections) ran out of memory after 2 h 40 min of
the write phase. Rerunning only that document would take hours for one
data point, and the 5 documents still include an 8-section one (doc0002).
The paper states the count per model.

## Phi-3.5-mini writes 2 documents
Phi's write phase took 8 h for under 3 documents. Its 8-section document
was past the memory cliff (36 min for one JSON generation), and the
remaining 4 documents were estimated at about 20 h more. Its write results
already repeat the Qwen pattern (LaTeX blocked by `$`, PPTX never valid),
and its understanding run, the part that needs a second model family, was
complete. The run was stopped with the user's agreement, and the paper
gives the document count per model.
