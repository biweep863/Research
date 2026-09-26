CONFIG ?= configs/local_small.yaml
PY ?= ../../.venv/bin/python

.PHONY: probe ablate figures paper all test clean

probe:
	$(PY) -m src.experiments.run_probe --config $(CONFIG)

ablate:
	$(PY) -m src.experiments.run_head_ablation --config $(CONFIG)

figures:
	$(PY) -m src.figures.plot_logic --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: probe ablate figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
