import math

import numpy as np
import pytest

from pof.stats.power import (_phi, _z, analytic_power, effect_size, empirical_power,
                             min_n_for_power)


def test_normal_quantile_inverts_cdf():
    for q in (1e-6, 0.01, 0.05, 0.5, 0.8, 0.95, 0.999):
        assert _phi(_z(q)) == pytest.approx(q, abs=1e-9)
    assert _z(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert _z(0.8) == pytest.approx(0.8416212, abs=1e-6)


def test_effect_size_and_min_n():
    rng = np.random.default_rng(0)
    d = effect_size(rng.normal(0.0, 1, 20000), rng.normal(0.5, 1, 20000))
    assert d == pytest.approx(0.5, abs=0.03)
    # textbook: one-sided alpha .05, power .8, d = .5 needs about 50 per group
    assert min_n_for_power(0.5) == 50
    assert math.isinf(min_n_for_power(0.0))
    assert analytic_power(0.5, 50) == pytest.approx(0.8, abs=0.01)


def test_empirical_power_is_calibrated_under_the_null():
    rng = np.random.default_rng(1)
    same = rng.exponential(size=4000)
    fpr = empirical_power(same[:2000], same[2000:], m=50, reps=400, n_perm=300, seed=2)
    assert fpr <= 0.09  # alpha = .05 plus Monte-Carlo slack


def test_empirical_power_matches_analytic():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.0, 1, 5000), rng.normal(0.4, 1, 5000)
    emp = empirical_power(a, b, m=60, reps=300, n_perm=300, seed=4)
    assert emp == pytest.approx(analytic_power(0.4, 60), abs=0.08)


def test_empirical_power_rejects_oversized_m():
    with pytest.raises(ValueError):
        empirical_power(np.zeros(10), np.zeros(10), m=11, reps=1)
