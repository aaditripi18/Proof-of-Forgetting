"""Statistical tests behind V5 (the client-side in/out test).

The client split its data uniformly at random into `in` (trained on) and `out`
(never trained on). Under H0 "the model carries no trace of the client's
training", the per-sample losses on `in` and `out` are exchangeable, so a
permutation test on the split labels is exact: no distributional assumptions,
and its p-values are uniform under H0 (tested in tests/test_stats.py).
"""
from __future__ import annotations

import numpy as np


def permutation_test(in_losses: np.ndarray, out_losses: np.ndarray, n_perm: int = 10000,
                     seed: int = 0) -> dict[str, float]:
    """One-sided test of H1: mean loss on `in` is LOWER than on `out` (memorised).

    Returns the observed effect (mean_out - mean_in, > 0 means remembered) and the
    p-value with the +1 correction, so p is never exactly 0 and the test is valid.
    """
    a = np.asarray(in_losses, dtype=np.float64)
    b = np.asarray(out_losses, dtype=np.float64)
    pooled = np.concatenate([a, b])
    n_a = len(a)
    observed = b.mean() - a.mean()
    rng = np.random.default_rng(seed)
    total = pooled.sum()
    count = 0
    chunk = 1000
    for start in range(0, n_perm, chunk):
        k = min(chunk, n_perm - start)
        # sum of a random n_a-subset for each of k permutations, vectorised
        keys = rng.random((k, len(pooled)))
        idx = np.argpartition(keys, n_a, axis=1)[:, :n_a]
        s_a = pooled[idx].sum(axis=1)
        stat = (total - s_a) / (len(pooled) - n_a) - s_a / n_a
        count += int(np.sum(stat >= observed - 1e-12))
    return {"effect": float(observed), "p_value": float((count + 1) / (n_perm + 1))}


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 5000, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat` of a 1-D sample (used across seeds)."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
