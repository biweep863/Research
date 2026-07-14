# 04 · Results

Headline numbers (local config):

**Secretary / the 37% rule (finite-n reality):**
- n=10:   optimal x*=0.300, P*=0.399
- n=100:  optimal x*=0.370, P*=0.371
- n=1000: optimal x*=0.368, P*=0.368  (≈ 1/e)

**Parking — naive 37% rule regret vs exact DP optimum:**
- p=0.3: ~0%      p=0.5: ~49%     p=0.7: ~285%     p=0.9: ~1602%

**Search cost shrinks the secretary cutoff (n=100):**
- c=0: x*=0.370  →  c=0.005: x*=0.130

**Agents (5 seeds, mean regret):**
- Secretary Q-learning: ~1.4% (n=100); REINFORCE ~17.6% (n=100)
- Parking Q-learning: 0% (p=0.9), ~10–19% (p=0.3–0.7)

## Interpretation
The 37% rule is exact only in the rank-only, cost-free, large-n limit. Every
relaxation moves the threshold: occupancy pushes the parking cutoff toward the
destination, search cost pulls the secretary cutoff below 1/e, and cardinal
information raises the achievable win rate above 1/e entirely. A tabular
value-based agent recovers the correct, problem-specific policy from reward
alone; a one-parameter policy-gradient agent recovers the regime but is limited
by its stationary policy class. The transferable lesson is the *procedure*
(stop when the current option beats the continuation value), not the number 37%.

See `paper/main.pdf` for the formal write-up with figures.
