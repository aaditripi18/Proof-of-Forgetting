"""E0: would the standard forgetting test pass a server that never forgot?

Trains the original federation, applies each unlearning method in the config to
`unlearn.forget_client`, then scores every resulting model with:

  V1-pooled   loss-threshold MIA, members = forget client's training half,
              non-members = pooled validation split (the common setup)
  V1-matched  same attack, non-members = the forget client's own held-out half
              (removes the client-vs-population distribution confound)
  V3          client-level LiRA against a shadow bank (only with --shadows; build
              the bank first with scripts/run_shadows.py on the same config)
  V4          distance to the retrain oracle (weights, predictions), read against
              the noise floor retrain_oracle vs retrain_oracle_alt (server-only)
  V5          client-side permutation test, in-half vs out-half losses

Every number goes to the ledger before it goes to the CSV.

    python scripts/run_e0.py --config configs/e0_sanity.yaml
    python scripts/run_e0.py --config configs/smoke.yaml          # CPU plumbing check
    python scripts/run_e0.py --config configs/e0_sanity.yaml --set data.alpha=0.1 seed=43
    python scripts/run_e0.py --config configs/e0_sanity.yaml --shadows results/shadows/<bank>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.experiment import prepare, start_run, write_csv  # noqa: E402
from pof.fl.fedavg import per_sample_eval, train_federated  # noqa: E402
from pof.seed import derive_seed  # noqa: E402
from pof.shadows import check_compatible, load_bank, pool_of, sample_membership  # noqa: E402
from pof.stats.hypothesis import permutation_test  # noqa: E402
from pof.unlearn.registry import get_method, load_external_methods  # noqa: E402
from pof.verify import lira  # noqa: E402
from pof.verify.loss_mia import loss_mia  # noqa: E402
from pof.verify.oracle_distance import oracle_distance, predict_probs  # noqa: E402


def score_model(cfg, state, env, f) -> tuple[dict, dict]:
    bs, k, part = cfg["eval"]["batch_size"], env["k"], env["part"]
    e_in = per_sample_eval(cfg, state, env["train"], part["client_in"][f], k, bs)
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
    v5 = permutation_test(e_in["loss"], e_out["loss"], seed=derive_seed(cfg["seed"], "v5"))
    row["v5_effect"], row["v5_p"] = v5["effect"], v5["p_value"]
    return row, {"in": e_in, "out": e_out, "val": e_val}


def v3_reference(cfg, env, bank_path, f, log):
    """LiRA Gaussians for the forget client's samples, and the null distribution
    of its client score on shadows that never trained on it (leave-one-out)."""
    bank = load_bank(bank_path)
    check_compatible(bank, cfg, env["part"])
    _, owner = pool_of(env["part"])
    cols = owner == f
    phi = bank["lsc"][:, cols]
    member = sample_membership(bank["membership"], owner)[:, cols]
    fv = cfg["lira"]["fixed_variance"]
    null = lira.loo_scores(phi, member, fv).mean(1)[~bank["membership"][:, f]]
    log(f"V3: bank {bank['path']} ({bank['manifest']['num_shadow']} shadows, "
        f"{len(null)} OUT for client {f})")
    return lira.fit(phi, member, fv), null


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. fl.rounds=5 seed=43")
    ap.add_argument("--external-methods", help="python file registering private methods")
    ap.add_argument("--shadows", help="shadow bank directory; enables V3")
    args = ap.parse_args(argv)

    cfg, run_id, log, ledger = start_run(args.config, args.set, source="scripts/run_e0.py")
    if args.external_methods:
        load_external_methods(args.external_methods)
    env = prepare(cfg, log.info)
    seed, k, part = cfg["seed"], env["k"], env["part"]
    f = cfg["unlearn"]["forget_client"]
    out_dir = Path(cfg["output"]["dir"])
    v3 = v3_reference(cfg, env, args.shadows, f, log.info) if args.shadows else None

    ckpt_dir = out_dir / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    original = train_federated(cfg, env["train"], part["client_in"], env["clients"], k, seed,
                               tag="train", log=log.info)
    torch.save(original, ckpt_dir / "original.pt")

    ctx = {"cfg": cfg, "seed": seed, "data": env["train"], "client_in": part["client_in"],
           "num_classes": k, "clients": env["clients"], "forget_client": f,
           "original_state": original, "log": log.info}
    states = {}
    for method in cfg["methods"]:
        log.info(f"unlearning: {method}")
        states[method] = get_method(method)(ctx)
        torch.save(states[method], ckpt_dir / f"{method}.pt")

    # V4 needs the oracle's outputs, so all models exist before any is scored.
    bs = cfg["eval"]["batch_size"]
    probe = {"forget": (env["train"], part["client_in"][f]),
             "test": (env["test"], np.arange(len(env["test"])))}
    probs_of = lambda s: {name: predict_probs(cfg, s, d, idx, k, bs)
                          for name, (d, idx) in probe.items()}
    oracle = states.get("retrain_oracle")
    oracle_probs = probs_of(oracle) if oracle is not None else None
    if oracle is None:
        log.info("retrain_oracle not in methods: skipping V4")

    rows, per_sample = [], {}
    for method, state in states.items():
        row, evals = score_model(cfg, state, env, f)
        if v3 is not None:
            stats, null = v3
            t = lira.client_test(float(lira.score(evals["in"]["lsc"], stats).mean()), null)
            row.update({f"v3_{m}": v for m, v in t.items()})
        if oracle is not None:
            v4 = oracle_distance(state, oracle, original, probs_of(state), oracle_probs)
            row.update({f"v4_{m}": v for m, v in v4.items()})
        for metric, v in row.items():
            ledger.record(metric, v, method=method, forget_client=f)
        rows.append({"run_id": run_id, "method": method, **row})
        per_sample[method] = evals
        log.info(f"  {method}: " + "  ".join(f"{m}={v:.4f}" for m, v in row.items()))

    np.savez_compressed(ckpt_dir / "per_sample.npz", **{
        f"{m}_{split}_{field}": e[split][field]
        for m, e in per_sample.items() for split in e for field in e[split]})
    csv_path = write_csv(out_dir / cfg["experiment"] / f"{run_id}.csv", rows)
    log.info(f"wrote {csv_path}")

    by = {r["method"]: r for r in rows}
    if "do_nothing" in by and "retrain_oracle" in by:
        for name in ("pooled", "matched"):
            a, b = by["do_nothing"][f"v1_{name}_auc"], by["retrain_oracle"][f"v1_{name}_auc"]
            log.info(f"V1-{name}: do_nothing AUC {a:.4f} vs oracle AUC {b:.4f}  (gap {a - b:+.4f})")
        if v3 is not None:
            log.info(f"V3: do_nothing p={by['do_nothing']['v3_p']:.4g}  "
                     f"oracle p={by['retrain_oracle']['v3_p']:.4g}")
        log.info(f"V5: do_nothing p={by['do_nothing']['v5_p']:.4g}  "
                 f"oracle p={by['retrain_oracle']['v5_p']:.4g}")
    if "retrain_oracle_alt" in by and "v4_w_rel" in by["retrain_oracle_alt"]:
        floor = by["retrain_oracle_alt"]
        log.info(f"V4 noise floor (oracle vs oracle_alt): w_rel {floor['v4_w_rel']:.4f}  "
                 f"js_forget {floor['v4_js_forget']:.5f}")
        for m, r in by.items():
            if m not in ("retrain_oracle", "retrain_oracle_alt"):
                log.info(f"V4 {m}: w_rel {r['v4_w_rel']:.4f} "
                         f"({r['v4_w_rel'] / floor['v4_w_rel']:.1f}x floor)  "
                         f"js_forget {r['v4_js_forget']:.5f}")


if __name__ == "__main__":
    main()
