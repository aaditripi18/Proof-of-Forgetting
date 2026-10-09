"""V3: client-level LiRA (Carlini et al. 2022), adapted to federated clients.

Shadow federations each leave out a few clients (`balanced_client_subsets`),
so for every training sample there are shadow models whose federation included
its client (IN) and ones that did not (OUT).
For each sample, LiRA fits a Gaussian to the logit-scaled confidence `lsc` of
the IN shadows and one to the OUT shadows, and scores a target model by the
log-likelihood ratio of its own `lsc` under the two:

    score(x) = log N(lsc_t(x); mu_in(x), var_in) - log N(lsc_t(x); mu_out(x), var_out)

A client's score is the mean over its samples: positive means the target looks
like a model that trained on the client.

With few shadows, per-sample variances are noisy; `fixed_variance=True` uses one
pooled variance for IN and one for OUT (as Carlini et al. do below ~64 shadows).

Everything here is pure numpy. Arrays are shaped [S, N]: S shadow models by N
pool samples (the union of all clients' training halves).
"""
from __future__ import annotations

import math

import numpy as np

_MIN_VAR = 1e-6


def gaussian_logpdf(x: np.ndarray, mu: np.ndarray, var) -> np.ndarray:
    var = np.maximum(var, _MIN_VAR)
    return -0.5 * (np.log(2 * math.pi * var) + (x - mu) ** 2 / var)


def fit(phi: np.ndarray, member: np.ndarray, fixed_variance: bool = True) -> dict:
    """Per-sample IN/OUT Gaussians from shadow statistics phi[S, N], member[S, N]."""
    phi = np.asarray(phi, dtype=np.float64)
    member = np.asarray(member, dtype=bool)
    n_in, n_out = member.sum(0), (~member).sum(0)
    if (n_in < 2).any() or (n_out < 2).any():
        # one model per side gives a zero variance and an unbounded likelihood ratio
        raise ValueError("every sample needs at least two IN and two OUT shadows "
                         "(raise num_shadow so leave-one-out still leaves two per side)")
    mu_in = np.where(member, phi, 0).sum(0) / n_in
    mu_out = np.where(~member, phi, 0).sum(0) / n_out
    r_in = np.where(member, phi - mu_in, 0.0) ** 2
    r_out = np.where(~member, phi - mu_out, 0.0) ** 2
    if fixed_variance:
        var_in = r_in.sum() / max(n_in.sum() - phi.shape[1], 1)
        var_out = r_out.sum() / max(n_out.sum() - phi.shape[1], 1)
    else:
        var_in = r_in.sum(0) / np.maximum(n_in - 1, 1)
        var_out = r_out.sum(0) / np.maximum(n_out - 1, 1)
    return {"mu_in": mu_in, "mu_out": mu_out, "var_in": var_in, "var_out": var_out}


def score(phi_target: np.ndarray, stats: dict) -> np.ndarray:
    """Per-sample log-likelihood ratio IN vs OUT for one target model."""
    return (gaussian_logpdf(phi_target, stats["mu_in"], stats["var_in"])
            - gaussian_logpdf(phi_target, stats["mu_out"], stats["var_out"]))


def loo_scores(phi: np.ndarray, member: np.ndarray, fixed_variance: bool = True) -> np.ndarray:
    """Leave-one-out evaluation: row s is scored with Gaussians fitted to every
    shadow except s, so no shadow is ever attacked with its own statistics."""
    S = phi.shape[0]
    out = np.empty(phi.shape, dtype=np.float64)
    for s in range(S):
        keep = np.arange(S) != s
        out[s] = score(phi[s], fit(phi[keep], member[keep], fixed_variance))
    return out


def client_scores(sample_scores: np.ndarray, sample_client: np.ndarray,
                  num_clients: int) -> np.ndarray:
    """Mean sample score per client; works on [N] or [S, N] inputs."""
    sample_scores = np.atleast_2d(sample_scores)
    out = np.zeros((sample_scores.shape[0], num_clients))
    for c in range(num_clients):
        mask = sample_client == c
        out[:, c] = sample_scores[:, mask].mean(1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def normal_sf(z: float) -> float:
    """P(Z >= z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def client_test(target_client_score: float, null_scores: np.ndarray) -> dict[str, float]:
    """Is the target's score for a client above what OUT models produce?

    null_scores are the same client's scores on shadows that never trained on it
    (leave-one-out). Returns a z-score, a parametric one-sided p-value, and the
    empirical p-value with +1 correction (coarse: its floor is 1/(n_null + 1)).
    """
    null = np.asarray(null_scores, dtype=np.float64)
    sd = null.std(ddof=1) if len(null) > 1 else float("nan")
    z = (target_client_score - null.mean()) / sd if sd > 0 else float("nan")
    p_emp = (1 + np.sum(null >= target_client_score)) / (1 + len(null))
    return {"score": float(target_client_score), "z": float(z),
            "p": normal_sf(z) if np.isfinite(z) else float("nan"), "p_emp": float(p_emp)}
