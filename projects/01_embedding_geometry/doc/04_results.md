# 04 · Results

Headline (GPT-2, mid-layer, ctx 128 → 1024):
- TwoNN intrinsic dim:   8.61 → 0.89
- Anisotropy:            0.523 → 0.414
- Effective-rank Δ:      −267.56 (long-context has higher global rank)

The simultaneous drop in intrinsic dim and increase in anisotropy in the
mid layers is the canonical "collapse" signature. The fact that effective
rank increases — i.e. global variance spreads across more directions —
while TwoNN drops indicates that the manifold is *locally* much flatter
even when it is *globally* more spread out. This is the geometric content
of the long-context regime that the paper claims.

See `paper/main.pdf` for the formal write-up with figures.
