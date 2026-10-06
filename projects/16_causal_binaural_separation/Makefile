CONFIG ?= configs/local_small.yaml
PY ?= ../../.venv/bin/python
ARGS ?=

.PHONY: data run oracle train evaluate bench figures paper all test sanity clean

data:
	$(PY) -m src.experiments.prepare_data --config $(CONFIG) $(ARGS)

oracle: data
	$(PY) -m src.experiments.run_oracle_sweep --config $(CONFIG) $(ARGS)

train: data
	$(PY) -m src.experiments.train_sweep --config $(CONFIG) $(ARGS)

evaluate:
	$(PY) -m src.experiments.evaluate --config $(CONFIG) $(ARGS)

bench:
	$(PY) -m src.experiments.bench_latency --config $(CONFIG) $(ARGS)

run: oracle train evaluate bench

figures:
	$(PY) -m src.figures.plot_latency --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

# End-to-end on a tiny subset (< 5 min), writes to results/sanity/
sanity:
	$(MAKE) run ARGS=--sanity

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
