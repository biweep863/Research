CONFIG ?= configs/local_small.yaml
# Windows venv uses Scripts/, POSIX uses bin/
PY ?= $(firstword $(wildcard ../../.venv/Scripts/python.exe ../../.venv/bin/python) python)

.PHONY: run figures paper all test clean demo live record calibrate beamform hw

run:
	$(PY) -m src.experiments.run_doa_sim --config $(CONFIG)
	$(PY) -m src.experiments.run_distance_sim --config $(CONFIG)
	$(PY) -m src.experiments.run_beamform_sim --config $(CONFIG)
	$(PY) -m src.experiments.run_radar_demo --config $(CONFIG)
	$(PY) -m src.experiments.run_vad_highpass_sim --config $(CONFIG)

figures:
	$(PY) -m src.figures.plot_results --results results --out paper/figures --macros paper/macros.tex

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: run figures paper

test:
	$(PY) -m pytest tests -q

# ---- hardware / interactive -------------------------------------------------
# replay the simulated moving talker on the radar (no hardware needed)
demo:
	$(PY) -m src.experiments.live_radar --config $(CONFIG) --wav data/demo_moving_speaker.wav

# live radar from the ReSpeaker (ARGS="--device 3", "--no-plot", "--log run.csv", ...)
live:
	$(PY) -m src.experiments.live_radar --config $(CONFIG) $(ARGS)

# labelled clip: make record ANGLE=90 DIST=1.5 [ARGS="--note 'fan on'"]
record:
	$(PY) -m src.experiments.record_session --config $(CONFIG) --angle $(ANGLE) --distance $(DIST) $(ARGS)

# summarise the hardware session logs (data/hw_*.csv + newest clip) -> results/hw_session.json
hw:
	$(PY) -m src.experiments.analyze_hw_session --config $(CONFIG)

calibrate:
	$(PY) -m src.experiments.calibrate_distance --config $(CONFIG) --dir data/recordings

# spatial filter a recording: make beamform WAV=data/recordings/x.wav [ARGS=--deepfilter]
beamform:
	$(PY) -m src.experiments.beamform_file $(WAV) --config $(CONFIG) $(ARGS)

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.fdb_latexmk paper/*.fls paper/main.pdf
