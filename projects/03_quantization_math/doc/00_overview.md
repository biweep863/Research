# 00 · Overview
Measure how LLM math accuracy and logit fidelity degrade as weight bit-width
shrinks (FP32 → FP16 → INT8 → INT4 → ternary). Localise the degradation to
attention vs feed-forward submodules.
