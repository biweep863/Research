# 05 · Decisions & Alternatives

## Score the agent's full policy, not an extracted "threshold"
The first implementation summarised each learned parking policy by a single
accepted distance and evaluated *that*. This was wrong: the parking optimum is
per-step (near the street's end you accept any distance rather than fail), so a
single-threshold summary mislabels a near-optimal policy as terrible. We now
extract the agent's full greedy accept-mask and score it exactly via the
backward-induction recursion. Regret dropped from apparent hundreds of percent
to ~0–17%, revealing the agent was actually near-optimal.

## Exact scoring instead of Monte-Carlo evaluation of agents
Because we have closed forms (secretary mask-success) and exact DP (parking
policy cost), learned policies are scored analytically. This removes evaluation
noise so the reported regret reflects the *policy*, not sampling variance.

## Train agents on n ≤ 100, keep n = 1000 for analytic/MC only
Tabular Q-learning needs dense visitation of its states. For n=1000 the record
positions are visited too sparsely under a fixed episode budget, inflating
regret for reasons unrelated to the method. The asymptotic 1/e story is carried
by the closed form and Monte Carlo (which do run at n=1000); agents are trained
where the tabular representation is well-covered.

## REINFORCE kept despite poor parking performance
A one-parameter sigmoid threshold is the natural, interpretable policy for the
*stationary* secretary cutoff. It cannot represent the non-stationary parking
optimum — and we keep it precisely to make that point: recovering an optimal
rule needs a policy class rich enough to hold it, not just a learning
algorithm. Its high variance on secretary is reported honestly.

## Alpha and epsilon annealing
Constant step sizes left the acceptance boundary under-converged. Linearly
annealing both the learning rate (to 10% of its start) and exploration sharpens
the greedy policy late in training and eliminated parking regret at p≥0.5.

## Symmetric street with a failure penalty
The `[k..0..k]` street with a large `fail_cost` gives a clean, bounded,
exactly-solvable MDP whose optimal threshold depends on `p` — the property that
makes the "37% is universal" intuition visibly false. Correlated occupancy and
unknown street length are deferred to the GPU/extension config.
