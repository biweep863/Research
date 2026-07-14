"""Secretary-problem environment (rank-only, a.k.a. best-choice problem).

An episode is a uniformly random permutation of ``n`` distinct candidates.
The decision-maker sees only *relative* ranks and must commit irrevocably.
Only a running record (best-so-far) is ever worth taking, so the natural MDP
has state = position of the current running record and actions {take, skip}.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Fast vectorised Monte Carlo for a fixed skip-r threshold policy
# --------------------------------------------------------------------------- #


def threshold_success_rate(n: int, r: int, n_episodes: int, rng) -> float:
    """Fraction of episodes in which "reject first r, take next record" picks
    the global best. Vectorised over episodes."""
    wins = 0
    # process in chunks to bound memory for large n_episodes
    chunk = 20000
    done = 0
    while done < n_episodes:
        m = min(chunk, n_episodes - done)
        # random permutations: value 0 is the best candidate
        perms = np.argsort(rng.random((m, n)), axis=1)
        wins += _threshold_wins(perms, r)
        done += m
    return wins / n_episodes


def _threshold_wins(perms: np.ndarray, r: int) -> int:
    """Count wins for a batch of permutations under skip-r policy (vectorised).
    perms[e] is a permutation of 0..n-1; 0 denotes the best candidate."""
    m, n = perms.shape
    if r == 0:
        return int(np.sum(perms[:, 0] == 0))
    ref_min = perms[:, :r].min(axis=1)               # best in reject window
    tail = perms[:, r:]                              # (m, n-r)
    is_record = tail < ref_min[:, None]              # first record after window
    has = is_record.any(axis=1)
    first = np.argmax(is_record, axis=1)             # 0 when none (guarded)
    rows = np.arange(m)
    picked = np.where(has, tail[rows, first], perms[:, -1])
    return int(np.sum(picked == 0))


# --------------------------------------------------------------------------- #
# Episode interface for RL agents (state = current-record position)
# --------------------------------------------------------------------------- #


class SecretaryEpisode:
    """One episode exposed as a sequence of decision points at running records.

    Usage:
        ep = SecretaryEpisode(n, rng)
        for pos, is_best in ep.records():
            # decide take/skip; call ep.stop() semantics handled by caller
    We expose the record positions (1-indexed) and whether each record is the
    global best, which is all a rank-only agent can key on.
    """

    def __init__(self, n: int, rng):
        self.n = n
        self.perm = np.argsort(rng.random(n))  # 0 == best candidate

    def record_stream(self):
        """Yield (position_1indexed, is_global_best) for each running record."""
        best_seen = np.iinfo(np.int64).max
        for i in range(self.n):
            v = self.perm[i]
            if v < best_seen:
                best_seen = v
                yield i + 1, bool(v == 0)
