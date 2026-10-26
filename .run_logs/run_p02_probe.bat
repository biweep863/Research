@echo off
set PYTHONUNBUFFERED=1
cd /d "C:\Users\luiss\luis-proy\Research\projects\02_mechanistic_logic"
"C:\Users\luiss\luis-proy\Research\.venv\Scripts\python.exe" -m src.experiments.run_probe --config configs/gpu_full.yaml > "C:\Users\luiss\luis-proy\Research\.run_logs\p02_probe_full.log" 2>&1
