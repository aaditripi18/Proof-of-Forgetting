"""Scoring shared by the experiment scripts: V1, V2, V3 and V5 for one model.

V4 (distance to the oracle) lives in pof/verify/oracle_distance.py because it
needs the oracle, which only E0 trains.
"""
from __future__ import annotations

import numpy as np

from pof.fl.fedavg import per_sample_eval
from pof.seed import derive_seed
from pof.shadows import check_compatible, load_bank, pool_of, sample_membership
from pof.stats.hypothesis import permutation_test
from pof.verify import lira
from pof.verify.backdoor import eval_backdoor
from pof.verify.loss_mia import loss_mia


def v5_scores(evals: dict) -> np.ndarray:
    """The per-sample quantity V5 compares between the in and out halves; lower
    means "fits like training data".

    It is the negated logit-scaled confidence, not the loss. Cross-entropy is
    heavy-tailed: on the first 65%-accuracy run a handful of very-high-loss
    images falling in the held-out half made the mean-loss test flag the retrain
    oracle (p = 0.03) although its accuracy on the two halves was identical. The
    logit-scaled confidence is close to Gaussian (the reason LiRA uses it), and a
    permutation test is exact for any statistic. Chosen after seeing seed 42, so
    seed 42 is not evidence for it; later seeds are.
    """
    return -np.asarray(evals["lsc"], dtype=np.float64)


def clean_in(env, f) -> np.ndarray:
    """The forget client's training half without its canaries (if any)."""
    idx = env["part"]["client_in"][f]
    can = env.get("canary")
    return np.setdiff1d(idx, can["idx"]) if can is not None else idx


def score_model(cfg, state, env, f) -> tuple[dict, dict]:
    bs, k, part = cfg["eval"]["batch_size"], env["k"], env["part"]
    e_in = per_sample_eval(cfg, state, env["train"], clean_in(env, f), k, bs)
    e_out = per_sample_eval(cfg, state, env["train"], part["client_out"][f], k, bs)
    e_val = per_sample_eval(cfg, state, env["val"], np.arange(len(env["val"])), k, bs)
    e_test = per_sample_eval(cfg, state, env["test"], np.arange(len(env["test"])), k, bs)

    row = {"test_acc": e_test["correct"].mean(),
           "forget_in_acc": e_in["correct"].mean(),
           "forget_in_loss": e_in["loss"].mean(),
           "forget_out_loss": e_out["loss"].mean()}
    for name, ref in (("pooled", e_val), ("matched", e_out)):
        for metric, v in loss_mia(e_in["loss"], ref["loss"]).items():
            row[f"v1_{name}_{metric}"] = v
    v5 = permutation_test(v5_scores(e_in), v5_scores(e_out), seed=derive_seed(cfg["seed"], "v5"))
    row["v5_effect"], row["v5_p"] = v5["effect"], v5["p_value"]
    # the original loss-based V5, kept for comparison (see v5_scores)
    row["v5loss_p"] = permutation_test(e_in["loss"], e_out["loss"],
                                       seed=derive_seed(cfg["seed"], "v5"))["p_value"]
    if env.get("canary") is not None:
        v2 = eval_backdoor(cfg, state, env["test"], env["test_triggered"], env["v2_idx"], k,
                           env["canary"]["trigger"]["target"])
        row.update({f"v2_{m}": v for m, v in v2.items()})
    return row, {"in": e_in, "out": e_out, "val": e_val}


def v3_reference(cfg, env, bank_path, f, log):
    """LiRA Gaussians for the forget client's samples, and the null distribution
    of its client score on shadows that never trained on it (leave-one-out)."""
    bank = load_bank(bank_path)
    check_compatible(bank, cfg, env["part"])
    pool, owner = pool_of(env["part"])
    cols = owner == f
    phi = bank["lsc"][:, cols]
    member = sample_membership(bank["membership"], owner)[:, cols]
    fv = cfg["lira"]["fixed_variance"]
    null = lira.loo_scores(phi, member, fv).mean(1)[~bank["membership"][:, f]]
    log(f"V3: bank {bank['path']} ({bank['manifest']['num_shadow']} shadows, "
        f"{len(null)} OUT for client {f})")
    return lira.fit(phi, member, fv), null, pool[cols]


def v3_test(cfg, state, env, v3) -> dict:
    """V3 for one model, given v3_reference's output."""
    stats, null, idx = v3
    lsc = per_sample_eval(cfg, state, env["train"], idx, env["k"], cfg["eval"]["batch_size"])["lsc"]
    return lira.client_test(float(lira.score(lsc, stats).mean()), null)

