"""E4: how much data must a client have before forgetting is verifiable? (H4)

For each client size n in e4.sizes, the forget client keeps only n samples in
its training half and n in its private held-out half (random, seeded); every
other client is unchanged. The original federation is retrained at each n,
the configured methods are applied, and each model is scored with V5:

  p, reject, effect, d  the V5 test on all n vs n samples, and its effect size
  power_m<m>            empirical rejection rate on random m-vs-m subsets, for
                        m <= n/2 only: at m = n every "random subset" is the
                        whole set, so the rate would just be 0 or 1
  analytic_power_n      normal-approximation power at m = n; checked against
                        the empirical rates where both exist
  min_m_80              smallest m with 80% analytic power for this model

The oracle never depends on n (the client is not in it) and is trained once;
its rejection rates are the test's false-positive rate and must stay near
alpha. The headline is the curve of do_nothing's power against n: below the n
where it crosses 80%, a client cannot show that a server failed to forget it.

    python scripts/run_e4.py --config configs/e4_power.yaml
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.evaluate import v5_scores  # noqa: E402
from pof.experiment import prepare, start_run, write_csv  # noqa: E402
from pof.fl.fedavg import per_sample_eval, train_federated  # noqa: E402
from pof.seed import derive_seed  # noqa: E402
from pof.stats.hypothesis import permutation_test  # noqa: E402
from pof.stats.power import (analytic_power, effect_size, empirical_power,  # noqa: E402
                             min_n_for_power)
from pof.unlearn.registry import get_method  # noqa: E402


def shrink_client(part: dict, f: int, n: int, seed: int) -> dict:
    """A copy of the partition where client f keeps n random samples per half."""
    rng = np.random.default_rng(derive_seed(seed, "e4_shrink", f, n))
    p = copy.deepcopy(part)
    for half in ("client_in", "client_out"):
        idx = part[half][f]
        if n > len(idx):
            raise ValueError(f"client {f} has only {len(idx)} samples in {half}; n={n}")
        p[half][f] = np.sort(rng.choice(idx, n, replace=False))
    return p


def v5_power_row(cfg, state, env, part, f, n, sizes, e4) -> dict:
    bs = cfg["eval"]["batch_size"]
    lin = v5_scores(per_sample_eval(cfg, state, env["train"], part["client_in"][f], env["k"], bs))
    lout = v5_scores(per_sample_eval(cfg, state, env["train"], part["client_out"][f], env["k"], bs))
    t = permutation_test(lin, lout, seed=derive_seed(cfg["seed"], "e4_v5", n))
    d = effect_size(lin, lout)
    row = {"n": n, "p": t["p_value"], "reject": float(t["p_value"] <= e4["alpha"]),
           "effect": t["effect"], "d": d,
           "analytic_power_n": analytic_power(d, n, e4["alpha"]),
           "min_m_80": min_n_for_power(d, 0.8, e4["alpha"])}
    for m in (m for m in sizes if 2 * m <= n):
        row[f"power_m{m}"] = empirical_power(lin, lout, m, e4["reps"], e4["alpha"],
                                             n_perm=e4["n_perm"],
                                             seed=derive_seed(cfg["seed"], "e4_power", n, m))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    cfg, run_id, log, ledger = start_run(args.config, args.set, source="scripts/run_e4.py")
    env = prepare(cfg, log.info)
    seed, k, f, e4 = cfg["seed"], env["k"], cfg["unlearn"]["forget_client"], cfg["e4"]
    sizes = sorted(e4["sizes"])
    cap = min(len(env["part"]["client_in"][f]), len(env["part"]["client_out"][f]))
    if sizes[-1] > cap:
        log.info(f"client {f} has {cap} samples per half; dropping sizes above it")
        sizes = [s for s in sizes if s <= cap]

    oracle_ctx = {"cfg": cfg, "seed": seed, "data": env["train"], "num_classes": k,
                  "clients": env["clients"], "forget_client": f, "log": log.info,
                  "client_in": env["part"]["client_in"]}
    oracle = get_method("retrain_oracle")(oracle_ctx) if "retrain_oracle" in e4["methods"] else None

    rows = []
    for n in sizes:
        part = shrink_client(env["part"], f, n, seed)
        log.info(f"n = {n}: training the original with client {f} holding {n} samples")
        original = train_federated(cfg, env["train"], part["client_in"], env["clients"], k, seed,
                                   tag="train", log=None)
        ctx = {**oracle_ctx, "client_in": part["client_in"], "original_state": original}
        for method in e4["methods"]:
            state = oracle if method == "retrain_oracle" else get_method(method)(ctx)
            row = v5_power_row(cfg, state, env, part, f, n, sizes, e4)
            name = f"{method}@n={n}"
            for metric, v in row.items():
                ledger.record(metric, v if math.isfinite(v) else -1.0, method=name, forget_client=f)
            rows.append({"run_id": run_id, "method": name,
                         **{m: (v if math.isfinite(v) else -1.0) for m, v in row.items()}})
            emp = {k: v for k, v in row.items() if k.startswith("power_m")}
            log.info(f"  {name:<28} p={row['p']:.3g} d={row['d']:.3f} "
                     f"analytic power@n={row['analytic_power_n']:.2f} "
                     + " ".join(f"{k[6:]}={v:.2f}" for k, v in emp.items()))

    csv_path = write_csv(Path(cfg["output"]["dir"]) / cfg["experiment"] / f"{run_id}.csv", rows)
    log.info(f"wrote {csv_path}  (min_m_80 = -1 means no positive effect: never detectable)")
    dn = [r for r in rows if r["method"].startswith("do_nothing@")]
    crossing = next((r["n"] for r in dn if r["analytic_power_n"] >= 0.8), None)
    log.info(f"smallest client size at which do_nothing is detected with 80% (analytic) power: "
             f"{crossing if crossing is not None else 'none of ' + str(sizes)}")
    return rows


if __name__ == "__main__":
    main()
