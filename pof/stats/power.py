"""Power of the V5 in/out test, and the smallest sample size that reaches it.

E4's question is when "not forgotten" becomes detectable at all. Two estimates:

  empirical   draw `reps` random subsets of m in-losses and m out-losses, run the
              permutation test on each, and count rejections at level alpha.
              On a model that never saw the client this is the false-positive
              rate and must stay near alpha; on one that did, it is the power.
  analytic    the normal approximation for a one-sided two-sample test with
              standardised effect d:  power(m) = Phi(d * sqrt(m / 2) - z_{1-alpha}).

The analytic curve extrapolates beyond the sizes actually measured; the
empirical one checks that it is right where both exist.
"""
from __future__ import annotations

import math

import numpy as np

from pof.stats.hypothesis import permutation_test


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _z(q: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |error| < 1e-8 after
    one Newton step), so no scipy dependency."""
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416]
    lo = 0.02425
    if q < lo:
        t = math.sqrt(-2 * math.log(q))
        x = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / \
            ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    elif q > 1 - lo:
        return -_z(1 - q)
    else:
        t = q - 0.5
        r = t * t
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = _phi(x) - q  # one Newton step
    return x - e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)


def effect_size(in_losses: np.ndarray, out_losses: np.ndarray) -> float:
    """Standardised effect d = (mean_out - mean_in) / pooled sd; > 0 means remembered."""
    a, b = np.asarray(in_losses, float), np.asarray(out_losses, float)
    sd = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / sd) if sd > 0 else 0.0


def analytic_power(d: float, m: int, alpha: float = 0.05) -> float:
    return _phi(d * math.sqrt(m / 2) - _z(1 - alpha))


def min_n_for_power(d: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Smallest per-half size m with analytic power >= `power`; inf if d <= 0."""
    if d <= 0:
        return math.inf
    return math.ceil(2 * ((_z(1 - alpha) + _z(power)) / d) ** 2)


def empirical_power(in_losses: np.ndarray, out_losses: np.ndarray, m: int, reps: int,
                    alpha: float = 0.05, n_perm: int = 500, seed: int = 0) -> float:
    """Rejection rate of the V5 test on `reps` random size-m subsets of each half."""
    a, b = np.asarray(in_losses), np.asarray(out_losses)
    if m > min(len(a), len(b)):
        raise ValueError(f"m={m} exceeds the available samples ({len(a)}, {len(b)})")
    rng = np.random.default_rng(seed)
    rejections = 0
    for r in range(reps):
        sa = a[rng.choice(len(a), m, replace=False)]
        sb = b[rng.choice(len(b), m, replace=False)]
        rejections += permutation_test(sa, sb, n_perm=n_perm, seed=seed + r + 1)["p_value"] <= alpha
    return rejections / reps
