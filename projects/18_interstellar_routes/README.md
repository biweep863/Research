# Project 18 — Interstellar routes through the real, moving star field

## Question
On the real Gaia star graph, how much do stellar motion and refuelling stops
reduce the per-hop range, travel time and propellant needed to reach the stars,
compared with the static nearest-neighbour routing used in prior work?

## Quick start
```bash
# from repo root (Windows: PY=../../.venv/Scripts/python.exe)
source .venv/bin/activate
pip install astroquery astropy        # project-only deps
cd projects/18_interstellar_routes
make test           # brute-force checks for every optimiser, astropy + Gliese 710 kinematics
make run            # downloads + caches Gaia DR3/GCNS (50 pc), runs the 4 experiments (~3 min)
make figures        # regenerate figures + paper/macros.tex
make paper          # compile paper/main.pdf
```
The first `make run` queries the ESA Gaia archive and the NASA Exoplanet
Archive (network needed once; everything is cached under `data/cache/`).

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
100 pc GCNS (~330k stars), 60 pc evaluation ball, 2 Myr horizons, 100 bootstrap
draws. CPU-only; ~16 GB RAM recommended. Writes to `results/full/`.

## Files
- `src/data/gaia.py`: Gaia DR3 download in chunks with a completeness check, the GCNS contamination filter, the literature supplement (α Cen, Sirius, Procyon, Luhman 16, WISE 0855) and exoplanet hosts
- `src/model/kinematics.py`: astrometry → pc / pc·Myr⁻¹, missing-RV fill conditioned on the measured tangential velocity, linear propagation
- `src/model/propulsion.py`: relativistic rocket cruise speed, sail braking vs luminosity, brachistochrone
- `src/analysis/graph.py`: Euclidean MST, minimax (bottleneck) reach, range graphs
- `src/analysis/temporal.py`: single-window temporal edges and exact earliest-arrival Dijkstra (**core contribution**)
- `src/analysis/routing.py`: A*, greedy pilot, exact fuel bottleneck with refuelling at hosts only, sail graph
- `src/analysis/tours.py`: multi-probe NN and 2-opt tours, MST bound
- `src/experiments/run_{static,timedep,routing,exploration}.py`: write `results/*.json`
- `src/figures/plot_routes.py`: 6 figures + 100+ LaTeX macros
- `paper/main.tex`: two-column paper; `doc/`: methodology, log, results, decisions

## Headline
With frozen stars, reach percolates sharply. Half the stars within 25 pc need
2.16 pc hops, about twice the typical spacing. With Gaia alone, the Sun's
4-star pocket only opens at 2.27 pc. If colonies may **wait** for stars to
drift into range, the needed range falls by 20% within 100 kyr and 52% within
500 kyr. Below threshold, settlement moves at stellar speeds (~21 km/s).
