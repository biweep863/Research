# Project template

How to spin up a new research project in `projects/NN_name/` so it
matches the six existing ones (`01_embedding_geometry` … `06_topological_attention`)
and gets picked up automatically by the site under `web/`.

Two parts:
1. **Intake questions** — what I need to know from you before writing code.
2. **Canonical structure** — every file and directory the project must have,
   what goes in it, and why.

---

## 1. Intake questions

Answer these before I create the directory. They are the minimum needed
to design a real experiment, not just scaffolding.

### 1.1 Research framing
1. **One-sentence question.** In the form "Does X cause Y in Z?" or
   "What is the relationship between X and Y under Z?". This becomes the
   first line of `README.md` → `## Question` and the abstract of the paper.
2. **Hypothesis.** What do you expect to find, and why (prior work,
   theory, intuition)? A null result is fine — state it explicitly.
3. **Falsification criterion.** What observation would make you say
   "the hypothesis is wrong"? If you can't answer this, the project isn't
   a research project yet.
4. **Novelty / delta.** What existing paper is closest, and what does
   this add? One sentence.

### 1.2 Object of study
5. **Model(s).** Which model family for the local pilot (default: `gpt2`
   or `pythia-410m` — fits on CPU/MPS in seconds), and which for the
   GPU run (Llama-3-8B, Mistral-7B, Pythia-2.8B, Qwen-Math-7B, …).
6. **Data.** Corpus / dataset name, split, size, and license. If there
   is no public dataset, describe the synthetic generator.
7. **Task / probe.** What exactly do you feed the model, and what do you
   read out (activations, logits, attention, loss, generation)?

### 1.3 Measurements
8. **Primary metric(s).** The number(s) that go in the abstract. Give
   the formula or the reference (e.g. "TwoNN intrinsic dim, Facco 2017").
9. **Secondary metrics.** Sanity checks, ablations, complements.
10. **Sweep axes.** What varies across runs (context length, bit-width,
    layer index, prompt-family, ODE solver budget, …).
11. **Statistical plan.** How many samples, what test, what significance
    threshold. If underpowered, say so (like project 06 did) — that's a
    finding.

### 1.4 Engineering budget
12. **Local wall-clock budget.** Target for `make all` on CPU/MPS with
    `local_small.yaml`: default is **≤ 5 minutes**. If the metric requires
    more, split it into a sanity sweep + a full sweep.
13. **GPU wall-clock budget.** Target for `gpu_full.yaml`: default is
    **≤ a few hours on a single A100/H100**. Cap dataset and context
    accordingly.
14. **External dependencies.** Any package outside the existing
    `projects/shared/env/requirements.txt`? List them so we can decide
    whether to pin globally or per-project.

### 1.5 Deliverable
15. **Venue target.** Preprint / Workshop / Journal / Conference —
    drives the `type` in `meta.yaml` and the tone of the paper.
16. **Topic label.** One word for the badge on the site
    (`Geometry`, `Interpretability`, `Efficiency`, `Architecture`,
    `Attention`, `Topology`, or a new one — new topics also need a
    color entry in `web/src/lib/utils.ts` `TOPIC_COLORS`).
17. **TL;DR.** One sentence, ≤ 200 chars, quantitative if possible.
    Goes verbatim into `meta.yaml` and the site card.

If you can answer 1–17, the project is ready to scaffold. If you can't
answer 3 (falsification) or 8 (primary metric), stop and refine the
question first.

---

## 2. Canonical structure

Every project directory looks like this. File contents and conventions
follow.

```
projects/NN_name/
├── README.md                 # Question, quick-start, file map
├── Makefile                  # test / run / figures / paper / all
├── meta.yaml                 # site metadata (source of truth)
├── configs/
│   ├── local_small.yaml      # CPU/MPS pilot config
│   └── gpu_full.yaml         # A100/H100 full-scale config
├── data/                     # generated or cached data (git-ignored)
├── src/
│   ├── __init__.py
│   ├── data/                 # loaders + synthetic generators
│   ├── analysis/             # metric implementations (pure functions)
│   ├── experiments/          # entry points (one file per experiment)
│   └── figures/              # plotting → figures + LaTeX macros
├── tests/
│   └── test_metrics.py       # unit tests for the analysis/ metrics
├── results/                  # JSON / CSV outputs (checked in if small)
├── doc/
│   ├── 00_overview.md
│   ├── 01_dataset.md
│   ├── 02_methodology.md
│   ├── 03_experiments_log.md
│   ├── 04_results.md
│   └── 05_decisions.md
└── paper/
    ├── main.tex              # two-column paper (uses shared preamble)
    ├── macros.tex            # numbers auto-generated from results
    ├── references.bib
    ├── sections/             # optional, if paper is split
    ├── figures/              # PDFs written by src/figures/*.py
    ├── tables/               # optional
    └── main.pdf              # compiled output (committed)
```

### 2.1 `README.md`
Small file, ~30 lines. Sections:
- `## Question` — the one-sentence intake #1.
- `## Quick start` — the exact `make` sequence for the local run.
- `## Scaling up` — the `CONFIG=configs/gpu_full.yaml` invocation.
- `## Files` — annotated map of the important paths in `src/` and `paper/`.

### 2.2 `Makefile`
Standardized targets — do not rename these; the top-level workflow and
the paper's reproducibility statement depend on them.

```makefile
CONFIG ?= configs/local_small.yaml
PY := ../../.venv/bin/python

.PHONY: run figures paper all test clean

run:
	$(PY) -m src.experiments.<entry> --config $(CONFIG)

figures:
	$(PY) -m src.figures.<plot> --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
```

### 2.3 `meta.yaml`
This is the source of truth for the site. See `web/README.md` for the
schema; the required fields are:

```yaml
slug: <unique_slug>
title: "Paper title exactly as in the PDF"
authors: ["Luis Benvenuto"]
year: 2026
topic: "<one of the TOPIC_COLORS keys>"
type: "Preprint"          # or Workshop / Journal / Conference
venue: "arXiv preprint"
tldr: "Intake question 17, one sentence."
pdf: "paper/main.pdf"
created: "YYYY-MM-DD"     # today, when the project is created
```

`updated` is **not** a field here — the site derives it from the last
git commit that touched anything in this directory.

### 2.4 `configs/`
Two YAMLs — a small one for laptops, a big one for GPUs. Everything the
experiment reads must live in the YAML, so scaling is a config swap and
never a code change. Typical keys:

```yaml
project: <slug>
seed: 0
device: auto              # auto | cpu | mps | cuda
dtype: fp32               # fp32 | fp16 | bf16
models:
  - name: gpt2
    short: gpt2
# sweep axes go here — context_lengths, bit_widths, layers, k, N, …
output_dir: results
```

**Rule:** if `local_small.yaml` and `gpu_full.yaml` share ≥ 90 % of
their keys but with different scalar values, you got the split right.
If the GPU config needs a new key, add it to *both* files (empty or
sensible default in the small one) so the code path stays identical.

### 2.5 `src/`
Layout by role, not by feature:

- `src/data/` — loaders. Each loader is a function `load_<name>(cfg) →
  data`. If HF or the network is not available, fall back to a bundled
  synthetic corpus (see `01_embedding_geometry`).
- `src/analysis/` — metric implementations. **Pure functions**, no
  model I/O, no plotting. Fully unit-tested (`tests/test_metrics.py`).
- `src/experiments/` — orchestration. One entry point per experiment,
  callable as `python -m src.experiments.<name> --config <path>`.
  Writes to `results/`.
- `src/figures/` — reads `results/`, writes PDFs to `paper/figures/`
  **and** appends numeric constants to `paper/macros.tex` (see 2.7).

Import the shared utilities from `projects.shared.utils`:
`seeds.set_all_seeds`, `model_loader.load_model`,
`hooks.collect_hidden_states`, `plotting.setup_matplotlib`,
`stats.bootstrap_ci`.

### 2.6 `tests/`
At minimum one test per metric in `src/analysis/`. The tests must run
in seconds without downloading a model — use tiny random tensors with
known analytic answers. `make test` is a precondition for `make run`.

### 2.7 `paper/`
Two-column layout, using the shared preamble at
`projects/shared/latex/twocolumn_preamble.tex`. Every number in the
prose comes from `\macros.tex`, which is regenerated by `make figures`.
That way the paper cannot drift from the results: recomputing the
figures also rewrites the numbers. Example macro line:

```
\newcommand{\twoNNctxShort}{8.61}
\newcommand{\twoNNctxLong}{0.89}
```

and in `main.tex`:

```
TwoNN drops from \twoNNctxShort{} at ctx 128 to \twoNNctxLong{} at ctx 1024.
```

Affiliation is always `Independent Research` (the front-matter block
is standardized — see any existing `main.tex`).

### 2.8 `doc/` — the methodology log
Six files, in this order, each ~1 page:

- `00_overview.md` — the hypothesis and geometric/mechanical intuition,
  slightly more expanded than the paper's abstract.
- `01_dataset.md` — corpus, size, license, fallback logic. Both local
  and GPU configurations described side-by-side.
- `02_methodology.md` — exact metric formulas (or citations), an ASCII
  pipeline diagram from loader → results → figures → paper.
- `03_experiments_log.md` — a `| Date | Config | Command | Outcome |`
  table, appended to on every real run. **Observations** section for
  anything surprising.
- `04_results.md` — the headline numbers in plain text. Points at
  `paper/main.pdf` for the formal write-up.
- `05_decisions.md` — every non-obvious choice, with the alternative
  that was rejected and the reason. This is the file that lets a future
  reader (or reviewer) understand why the design is what it is.

Length target: 30–80 lines per file. If a file is empty, the project
isn't finished.

---

## 3. Scaffolding a new project — step by step

Once the intake answers exist:

```bash
# From repo root
N=07                                    # next number
NAME=my_new_topic                       # short, snake_case
mkdir -p projects/${N}_${NAME}/{configs,data,src/{data,analysis,experiments,figures},tests,results,doc,paper/{figures,sections,tables}}
touch projects/${N}_${NAME}/src/__init__.py
```

Then, in order:

1. **Write `meta.yaml`** with the 10 required fields. Set `created` to
   today's date.
2. **Write `README.md`** — Question, Quick start, Scaling up, Files.
3. **Draft the six `doc/*.md`** from the intake answers. The overview,
   dataset, methodology, and decisions files can be written *before*
   any code — that is the point of the intake questions.
4. **Copy `configs/local_small.yaml`** from a sibling project and edit.
   Also create `gpu_full.yaml` at the same time so you can never
   forget it later.
5. **Write metric functions in `src/analysis/`** and their tests in
   `tests/`. `make test` must pass before touching experiments.
6. **Write the experiment runner in `src/experiments/`** and confirm
   `make run CONFIG=configs/local_small.yaml` produces `results/*.json`.
7. **Write the plotter in `src/figures/`** — outputs PDFs *and* the
   macros. `make figures` must succeed.
8. **Copy `paper/main.tex`** from a sibling, include the shared
   preamble, wire in `\input{macros.tex}` and the figures.
   `make paper` must produce `paper/main.pdf`.
9. **Run `make all`** end-to-end from a clean checkout to confirm
   reproducibility.
10. **Commit and push.** Vercel picks up the new `meta.yaml` and PDF
    automatically; a new card appears on the site after the deploy
    (~1 min).

Adding a new **topic** color (if the answer to intake #16 is not one
of the existing ones): edit `TOPIC_COLORS` in `web/src/lib/utils.ts`
and add a Tailwind class pair.

---

## 4. What "done" means

A project ships when:
- [ ] `make test` passes.
- [ ] `make all` completes on `local_small.yaml` in the budget from #12.
- [ ] `paper/main.pdf` compiles and every numeric claim comes from
      `macros.tex`.
- [ ] All six `doc/*.md` files are populated.
- [ ] `meta.yaml` validates (see `web/scripts/build-manifest.mjs`).
- [ ] The site shows the new card after `git push`.

Underpowered pilots (like project 06) still count as done, as long as
the paper is honest about the sample size and prescribes the scaled run.
