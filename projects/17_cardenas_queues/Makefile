CONFIG ?= configs/local_small.yaml
PY ?= ../../.venv/bin/python

.PHONY: data run figures paper all test clean

# Download (once) the OSM extract and build the corridor inventory.
data:
	$(PY) -m src.experiments.run_inventory --config $(CONFIG)

run: data
	$(PY) -m src.experiments.run_analytic --config $(CONFIG)
	$(PY) -m src.experiments.run_validation --config $(CONFIG)
	$(PY) -m src.experiments.run_peak --config $(CONFIG)
	$(PY) -m src.experiments.run_montecarlo --config $(CONFIG)
	$(PY) -m src.experiments.run_interventions --config $(CONFIG)
	$(PY) -m src.experiments.run_daily --config $(CONFIG)

figures:
	$(PY) -m src.figures.plot_corridor --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/*.bbl paper/*.blg paper/main.pdf
