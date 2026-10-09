"""V1: loss-threshold membership inference, and the metrics used to report any MIA.

Two ways of summarising the same attack scores:
  auc            the average-case number most unlearning papers report;
  tpr_at_fpr     the number Carlini et al. (2022) argue actually matters: how
                 many members are confidently identified at a tiny false-positive
                 rate. An attack can sit at AUC 0.5 and still be non-trivial here.
Pure numpy; scores are "higher = more likely a member".
"""
from __future__ import annotations

import numpy as np


def auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties (exact, no sklearn)."""
    m = np.asarray(member_scores, dtype=np.float64)
    n = np.asarray(nonmember_scores, dtype=np.float64)
    if len(m) == 0 or len(n) == 0:
        raise ValueError("need at least one member and one non-member score")
    allv = np.concatenate([m, n])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):  # average ranks within tie groups
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_m = ranks[: len(m)].sum()
    return float((r_m - len(m) * (len(m) + 1) / 2.0) / (len(m) * len(n)))


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray, fpr: float) -> float:
    """Largest TPR achievable by a threshold whose FPR is <= `fpr`.

    The threshold is strictly above the k-th largest non-member score, where
    k = floor(fpr * n_nonmembers); ties never count in the attacker's favour.
    With few non-members, small FPRs are unreachable and this returns the TPR at
    FPR = 0, which is the conservative reading.
    """
    m = np.asarray(member_scores, dtype=np.float64)
    n = np.sort(np.asarray(nonmember_scores, dtype=np.float64))[::-1]
    k = int(np.floor(fpr * len(n)))
    thresh = n[k] if k < len(n) else -np.inf  # members must score strictly above
    return float(np.mean(m > thresh))


def loss_mia(member_losses: np.ndarray, nonmember_losses: np.ndarray,
             fprs: tuple[float, ...] = (0.01, 0.001)) -> dict[str, float]:
    """Score = -loss (members are fit better). Returns AUC and TPR at each FPR."""
    ms, ns = -np.asarray(member_losses), -np.asarray(nonmember_losses)
    out = {"auc": auc(ms, ns)}
    for f in fprs:
        out[f"tpr@{f:g}"] = tpr_at_fpr(ms, ns, f)
    return out
