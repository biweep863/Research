CONFIG ?= configs/local_small.yaml
# Windows venv uses Scripts/, POSIX uses bin/
PY ?= $(firstword $(wildcard ../../.venv/Scripts/python.exe ../../.venv/bin/python) python)
RESULTS ?= results
FIGS ?= paper/figures
MACROS ?= paper/macros.tex

.PHONY: run read understand write figures paper all test clean

run: read understand write

read:
	$(PY) -m src.experiments.run_read --config $(CONFIG)

understand:
	$(PY) -m src.experiments.run_understand --config $(CONFIG)

write:
	$(PY) -m src.experiments.run_write --config $(CONFIG)

# full run: make figures RESULTS=results/full FIGS=paper/figures_full MACROS=paper/macros_full.tex
figures:
	$(PY) -m src.figures.plot_results --results $(RESULTS) --out $(FIGS) --macros $(MACROS)

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
