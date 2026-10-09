import numpy as np
import pytest

from pof.stats.hypothesis import bootstrap_ci, permutation_test


def _ks_uniform(p: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between the sample and Uniform(0, 1)."""
    p = np.sort(p)
    n = len(p)
    return max(np.max(np.arange(1, n + 1) / n - p), np.max(p - np.arange(n) / n))


def test_pvalues_are_uniform_under_the_null():
    """The V5 calibration test: when in/out are exchangeable (an oracle that never
    saw the client), the test must not report 'remembered' more than alpha of the time."""
    rng = np.random.default_rng(0)
    ps = np.array([permutation_test(rng.exponential(size=40), rng.exponential(size=40),
                                    n_perm=999, seed=s)["p_value"] for s in range(300)])
    # KS critical value at n=300, alpha=0.01 is ~0.094
    assert _ks_uniform(ps) < 0.094
    assert np.mean(ps < 0.05) < 0.09


def test_detects_memorisation():
    rng = np.random.default_rng(1)
    in_losses = rng.exponential(0.5, size=200)   # fit better: trained on
    out_losses = rng.exponential(1.0, size=200)
    r = permutation_test(in_losses, out_losses, n_perm=2000)
    assert r["effect"] > 0
    assert r["p_value"] < 0.001


def test_one_sided_direction():
    rng = np.random.default_rng(2)
    r = permutation_test(rng.exponential(1.0, 200), rng.exponential(0.5, 200), n_perm=2000)
    assert r["effect"] < 0 and r["p_value"] > 0.9


def test_pvalue_never_zero():
    r = permutation_test(np.zeros(50), np.ones(50) * 10, n_perm=100)
    assert r["p_value"] == pytest.approx(1 / 101)


def test_bootstrap_ci_covers_mean():
    lo, hi = bootstrap_ci(np.random.default_rng(3).normal(5.0, 1.0, size=200))
    assert lo < 5.0 < hi and hi - lo < 0.5
