@echo off
set PYTHONUNBUFFERED=1
cd /d "C:\Users\luiss\luis-proy\Research\projects\14_drift_selection"
"C:\Users\luiss\luis-proy\Research\.venv\Scripts\python.exe" -m src.experiments.run_threshold_sweep --config configs/gpu_full.yaml > "C:\Users\luiss\luis-proy\Research\.run_logs\p14_threshold_full.log" 2>&1
"C:\Users\luiss\luis-proy\Research\.venv\Scripts\python.exe" -m src.experiments.run_effective_size --config configs/gpu_full.yaml > "C:\Users\luiss\luis-proy\Research\.run_logs\p14_effective_full.log" 2>&1
echo DONE > "C:\Users\luiss\luis-proy\Research\.run_logs\p14_gpu_full.done"
