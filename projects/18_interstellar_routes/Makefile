CONFIG ?= configs/local_small.yaml
PY ?= ../../.venv/bin/python

.PHONY: data run figures paper all test clean

data:
	$(PY) -m src.experiments.fetch_data --config $(CONFIG)

run: data
	$(PY) -m src.experiments.run_static --config $(CONFIG)
	$(PY) -m src.experiments.run_timedep --config $(CONFIG)
	$(PY) -m src.experiments.run_routing --config $(CONFIG)
	$(PY) -m src.experiments.run_exploration --config $(CONFIG)

figures:
	$(PY) -m src.figures.plot_routes --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/*.bbl paper/*.blg paper/main.pdf
