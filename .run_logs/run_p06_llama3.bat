@echo off
cd /d "C:\Users\luiss\luis-proy\Research\projects\06_topological_attention"
"C:\Users\luiss\luis-proy\Research\.venv\Scripts\python.exe" -m src.experiments.run_tda --config configs/gpu_full.yaml > "C:\Users\luiss\luis-proy\Research\.run_logs\p06_llama3_full.log" 2>&1
