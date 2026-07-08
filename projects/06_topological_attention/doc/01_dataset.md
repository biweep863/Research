# 01 · Dataset

## Local pilot
Ten hand-curated factual pairs. Each pair is a
(prompt, truthful continuation, hallucinated continuation) triple where
truthful and hallucinated continuations share part-of-speech and
approximate length. Example:

```
"The capital of France is"  →  "Paris"   (truthful)
                            →  "Berlin"  (hallucinated)
```

Rationale: matched surface form is essential — otherwise any
topological signal could be trivially explained by token frequency
or sequence length rather than by factuality. Ten pairs is
intentionally underpowered; the local pilot exists to validate the
pipeline and estimate effect size for a properly powered run.

## GPU configuration
`gpu_full.yaml` swaps the pilot pairs for TruthfulQA and targets
$N \geq 200$ pairs per condition (see `05_decisions.md` and the
power analysis in the paper).
