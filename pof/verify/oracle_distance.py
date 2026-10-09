"""V4: how close is an unlearned model to the retrain oracle?

Only the server can run this (it needs the oracle, which is the expensive thing
unlearning exists to avoid), so V4 is the reference the cheap verifiers are
judged against, not a verifier a client could use.

A distance on its own is uninterpretable: two legitimate retrains differ too,
through data order and augmentation alone. Every V4 number is therefore read
against the noise floor, the distance between `retrain_oracle` and
`retrain_oracle_alt` (same init and clients, different training randomness).
A method is "as close as a retrain can be" when it is within that floor.

Weight distances are only meaningful because every model here starts from the
same initialisation and so stays in one basin; across inits, permutation
symmetry would make them meaningless.
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-12


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row Jensen-Shannon divergence (natural log), in [0, ln 2]."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), _EPS, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    q /= q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a) - np.log(b))).sum(axis=1)
    return np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, math.log(2))


def agreement(p: np.ndarray, q: np.ndarray) -> float:
    """Fraction of rows where the two models predict the same class."""
    return float(np.mean(np.argmax(p, axis=1) == np.argmax(q, axis=1)))


def weight_distance(a: dict, b: dict) -> float:
    """L2 distance over all floating-point tensors of two state dicts."""
    total = 0.0
    for k, v in a.items():
        if v.is_floating_point():
            total += float((v.double() - b[k].double()).pow(2).sum())
    return math.sqrt(total)


def predict_probs(cfg: dict, state: dict, data, idx: np.ndarray, num_classes: int,
                  batch_size: int = 512) -> np.ndarray:
    import torch

    from pof.models import build_model

    model = build_model(cfg, num_classes, in_ch=data.x.shape[1]).to(data.device)
    model.load_state_dict(state)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            x, _ = data.batch(idx[s:s + batch_size])
            out.append(torch.softmax(model(x).double(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, num_classes))


def oracle_distance(state: dict, oracle: dict, original: dict, probs: dict,
                    oracle_probs: dict) -> dict[str, float]:
    """V4 metrics for one model.

    probs / oracle_probs: {"forget": (n, K), "test": (m, K)} softmax outputs.
    `w_rel` is distance-to-oracle as a fraction of the original's distance to the
    oracle: 1.0 means no closer than doing nothing, 0.0 means the oracle itself.
    """
    d_orig = weight_distance(original, oracle)
    d = weight_distance(state, oracle)
    out = {"w_dist": d, "w_rel": d / d_orig if d_orig > 0 else 0.0}
    for split in ("forget", "test"):
        out[f"js_{split}"] = float(js_divergence(probs[split], oracle_probs[split]).mean())
        out[f"agree_{split}"] = agreement(probs[split], oracle_probs[split])
    return out
