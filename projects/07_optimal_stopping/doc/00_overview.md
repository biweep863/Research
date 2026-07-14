# 00 · Overview

This project studies optimal stopping — the mathematics of "when to stop
looking" — and asks how much of the folklore "37% rule" survives once the
idealised assumptions are relaxed, and whether learning agents recover the
right answer without being told the theory.

The 37% rule comes from the secretary problem: with `n` candidates in random
order, observing only ranks, reject the first `1/e` and take the next record;
this maximises the chance of picking the single best. We treat this as one
point in a family of stopping problems that also includes the **parking
problem** (MacQueen & Miller, 1960) — drive toward a destination, each spot is
free with probability `p`, minimise distance walked — plus a **search-cost**
variant and the cardinal **full-information** variant.

The hypothesis is twofold: (a) the 37% cutoff is a fragile special case that
moves substantially as we change the problem (occupancy, cost, information);
and (b) model-free reinforcement learning agents, given only reward, rediscover
the true problem-specific threshold. Exact backward induction gives computable
ground truth, so we can measure agent *regret* rather than a noisy return.
