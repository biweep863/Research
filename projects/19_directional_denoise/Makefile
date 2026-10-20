CONFIG ?= configs/local_small.yaml
# Windows venv uses Scripts/, POSIX uses bin/
PY ?= $(firstword $(wildcard ../../.venv/Scripts/python.exe ../../.venv/bin/python) python)

.PHONY: run spatial dfn live export15 figures paper all test clean record calibrate

# needs the recorded session in data/session (make record ...; doc/01_dataset.md)
run: spatial dfn live

spatial:
	$(PY) -m src.experiments.run_spatial --config $(CONFIG)

dfn:
	$(PY) -m src.experiments.run_dfn_asr --config $(CONFIG)

live:
	$(PY) -m src.experiments.run_live --config $(CONFIG)

# talk takes -> project 15 labelled clips (then `make calibrate` in project 15)
export15:
	$(PY) -m src.experiments.export_p15

figures:
	$(PY) -m src.figures.plot_denoise --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

test:
	$(PY) -m pytest tests -q

# ---- hardware session --------------------------------------------------------
# make calibrate LEVEL=-22 ; make record KIND=noise ARGS="--noise pink --channel L"
calibrate:
	$(PY) -m src.experiments.record_session calibrate --config $(CONFIG) --level-db $(LEVEL)

record:
	$(PY) -m src.experiments.record_session $(KIND) --config $(CONFIG) $(ARGS)

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
