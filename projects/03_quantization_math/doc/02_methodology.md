# 02 · Methodology

## Quantizer (`src/experiments/quantize_models.py`)
Per-row symmetric quantizer:
- 32-bit identity
- 16-bit FP16 round-trip
- 8-bit / 4-bit signed integer with per-row scale
- 2-bit ternary {−s, 0, +s} with threshold = 0.7 × mean(|W|) per row

## Evaluation
- Accuracy: greedy decode 12 new tokens, regex-extract first integer.
- Logit MSE: mean squared difference of last-position logits vs FP32.
- Logit KL: KL(softmax(ref) || softmax(quant)).

## Submodule sensitivity
Quantize **only** modules whose name contains `attn|attention|self_attn`
(attention) OR `mlp|ffn|fc|feed_forward` (FFN), holding the rest at FP32.
Run at 4-bit, then compare ΔAcc and logit MSE.
