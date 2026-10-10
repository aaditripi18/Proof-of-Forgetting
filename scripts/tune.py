"""Training sanity and tuning: how well does the original federation learn?

The first full run's models reached only ~43% CIFAR-10 test accuracy, which
weakens every membership signal. Each variant in the tuning config trains the
original federation (all clients) with its overrides and records, every
`eval_every` rounds, test accuracy and, at the end, how much the model
memorised the forget client:

  test_acc            accuracy on the test split
  train_acc_f         accuracy on the forget client's training half
  gap_f               train_acc_f - accuracy on its held-out half (memorisation)
  v5_d                V5 effect size on the forget client (in vs out losses)
  sec_per_round       wall-clock cost, for budgeting the full run

The "central" variant (one client holding all the data, same passes over it)
separates a bug in the training code from the cost of federation: if it is
also far below ~80%, the code is at fault.

    python scripts/tune.py --config configs/tune.yaml [--shard 0/2] [--set fl.device=cuda:1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.config import deep_merge, parse_overrides  # noqa: E402
from pof.experiment import prepare, start_run, write_csv  # noqa: E402
from pof.fl.fedavg import per_sample_eval, train_federated  # noqa: E402
from pof.stats.power import effect_size  # noqa: E402


def run_variant(base_cfg: dict, name: str, overrides: list[str], ledger, log) -> list[dict]:
    cfg = deep_merge(base_cfg, parse_overrides(overrides))
    env = prepare(cfg)
    k, part, f = env["k"], env["part"], cfg["unlearn"]["forget_client"]
    bs, every = cfg["eval"]["batch_size"], cfg["tune"]["eval_every"]
    test_idx = np.arange(len(env["test"]))
    rows, t0 = [], time.time()

    def record(r: int, state) -> None:
        if r % every and r != cfg["fl"]["rounds"]:
            return
        row = {"round": r, "test_acc": per_sample_eval(cfg, state, env["test"], test_idx, k, bs)["correct"].mean(),
               "sec_per_round": (time.time() - t0) / r}
        if r == cfg["fl"]["rounds"]:
            ein = per_sample_eval(cfg, state, env["train"], part["client_in"][f], k, bs)
            eout = per_sample_eval(cfg, state, env["train"], part["client_out"][f], k, bs)
            row.update(train_acc_f=ein["correct"].mean(),
                       gap_f=ein["correct"].mean() - eout["correct"].mean(),
                       v5_d=effect_size(ein["loss"], eout["loss"]))
        method = f"{name}@r={r}"
        for metric, v in row.items():
            ledger.record(metric, v, method=method, variant=name)
        rows.append({"method": method, **row})
        log(f"  {name:<14} round {r:>3}: " + "  ".join(f"{m}={v:.4f}" for m, v in row.items()
                                                       if m != "round"))

    log(f"variant {name}: {' '.join(overrides) or '(base)'}")
    train_federated(cfg, env["train"], part["client_in"], env["clients"], k, cfg["seed"],
                    tag="train", on_round=record)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--shard", default="0/1", help="k/n: run variants with index %% n == k")
    args = ap.parse_args(argv)
    k, n = (int(x) for x in args.shard.split("/"))

    cfg, run_id, log, ledger = start_run(args.config, args.set, source="scripts/tune.py")
    variants = [(i, v) for i, v in enumerate(cfg["tune"]["variants"]) if i % n == k]
    rows = []
    for _, v in variants:
        rows += [{"run_id": run_id, **r}
                 for r in run_variant(cfg, v["name"], v.get("overrides", []), ledger, log.info)]
    path = write_csv(Path(cfg["output"]["dir"]) / cfg["experiment"] / f"{run_id}.csv", rows)
    log.info(f"wrote {path}")
    return rows


if __name__ == "__main__":
    main()
