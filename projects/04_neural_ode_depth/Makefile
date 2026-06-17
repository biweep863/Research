CONFIG ?= configs/local_small.yaml
PY := ../../.venv/bin/python

.PHONY: train figures paper all test clean

train:
	$(PY) -m src.experiments.train_node --config $(CONFIG)

figures:
	$(PY) -m src.figures.plot_node --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: train figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
