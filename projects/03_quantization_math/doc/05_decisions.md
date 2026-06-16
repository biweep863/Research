# 05 · Decisions

- **No bitsandbytes / GPTQ / AWQ locally.** Those libs require CUDA or
  compile-time bindings that are flaky on macOS. The naive per-channel
  symmetric quantizer is enough to demonstrate the scaling law without
  external dependencies. SOTA quantizers move constants, not slopes —
  so this is a fair lower bound. GPU config can swap in `auto-gptq`.
- **Ternary as 1.58-bit proxy.** True BitNet requires training from
  scratch. Post-training round to {−1, 0, +1} with a 0.7-mean-abs
  threshold (a standard ternary heuristic) gives a meaningful stress
  test, clearly demarcated in the paper as an approximation.
- **Greedy answer extraction.** Regex `-?\d+` on the first 6 generated
  tokens — robust to LM verbosity for arithmetic tasks. Will not scale
  to MATH word problems; the GPU config switches to exact-match on
  the boxed answer for GSM8K/MATH.
- **Small model on local config.** Qwen2.5-0.5B is the smallest model
  with non-trivial baseline arithmetic ability among popular open
  weights. Using it lets us claim a real (non-zero) baseline accuracy
  to degrade.
