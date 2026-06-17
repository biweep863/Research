# 00 · Overview
Tiny continuous-depth transformer. One block defines a vector field;
`odeint(f, h0, [0,1])` runs the "forward pass" at variable solver budget.
Compare against discrete-depth baselines: same block run k times in a loop.
