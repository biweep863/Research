CONFIG ?= configs/local_small.yaml
PY := ../../.venv/bin/python

.PHONY: run figures paper all test clean

run:
	$(PY) -m src.experiments.run_detect --config $(CONFIG)
	$(PY) -m src.experiments.run_timing --config $(CONFIG)
	$(PY) -m src.experiments.run_overlap --config $(CONFIG)
	$(PY) -m src.experiments.run_decision --config $(CONFIG)
	$(PY) -m src.experiments.run_localize --config $(CONFIG)

figures:
	$(PY) -m src.figures.plot_contact --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/*.bbl paper/*.blg paper/main.pdf
