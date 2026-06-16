CONFIG ?= configs/local_small.yaml
PY := ../../.venv/bin/python

.PHONY: eval sens figures paper all test clean

eval:
	$(PY) -m src.experiments.run_math_eval --config $(CONFIG)

sens:
	$(PY) -m src.experiments.run_layerwise_sensitivity --config $(CONFIG)

figures:
	$(PY) -m src.figures.plot_quantization --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: eval sens figures paper

test:
	$(PY) -m pytest tests -q

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
