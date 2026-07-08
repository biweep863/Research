# 01 · Dataset

## Synthetic copy task
Vocabulary of 32 symbols. Each sequence is
`BOS x1 … x16 SEP` and the model is supervised to emit
`x1 … x16 EOS`. Body tokens are drawn uniformly at random, so the
next-token cross-entropy has an irreducible floor of
`log 32 / 16` averaged over supervised positions ≈ 1.47 nats.

Rationale: a fully synthetic task avoids any dependency on a pretrained
tokenizer or corpus and — because the irreducible floor is known
analytically — makes cross-configuration loss comparisons interpretable
in absolute units.

## GPU configuration
`gpu_full.yaml` scales body length to 64, model width to
$d_\text{model}=256$, and training to 20,000 steps; the task
generator is unchanged.
