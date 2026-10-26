@echo off
set PYTHONUNBUFFERED=1
cd /d "C:\Users\luiss\luis-proy\Research\projects\11_abyss_novelty"
"C:\Users\luiss\luis-proy\Research\.venv\Scripts\python.exe" -m src.experiments.run_novelty --config configs/gpu_full.yaml > "C:\Users\luiss\luis-proy\Research\.run_logs\p11_novelty_full.log" 2>&1
