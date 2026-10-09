"""V2: does the client's secret trigger still work?

Athena (Sommer et al.) compares the backdoor success rate against an assumed
base rate. Here the client needs no assumption: it queries the model on the
same non-target test images with and without its trigger and runs an exact
one-sided McNemar test on the discordant pairs:

    b = triggered -> target  and  clean -> not target   (the trigger did it)
    c = clean -> target      and  triggered -> not target
    H0: the trigger has no effect, so b ~ Binomial(b + c, 1/2)

A small p-value means the backdoor, and so the client's canaries, are still in
the model.
"""
from __future__ import annotations

import math

import numpy as np


def binom_sf_half(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2), exact, stable for large n."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(k, n + 1)]
    top = max(logs)
    return float(min(1.0, math.exp(top) * sum(math.exp(v - top) for v in logs)))


def backdoor_test(pred_triggered: np.ndarray, pred_clean: np.ndarray, target: int) -> dict:
    """V2 statistics from predictions on the same images with and without trigger."""
    t = np.asarray(pred_triggered) == target
    c = np.asarray(pred_clean) == target
    b, cc = int(np.sum(t & ~c)), int(np.sum(c & ~t))
    return {"bsr": float(t.mean()), "base": float(c.mean()), "effect": float(t.mean() - c.mean()),
            "p": binom_sf_half(b, b + cc)}


def eval_backdoor(cfg: dict, state: dict, test_clean, test_triggered, idx: np.ndarray,
                  num_classes: int, target: int) -> dict:
    """Run V2 for one model; idx selects non-target test images."""
    from pof.verify.oracle_distance import predict_probs

    bs = cfg["eval"]["batch_size"]
    pt = predict_probs(cfg, state, test_triggered, idx, num_classes, bs).argmax(1)
    pc = predict_probs(cfg, state, test_clean, idx, num_classes, bs).argmax(1)
    return backdoor_test(pt, pc, target)
