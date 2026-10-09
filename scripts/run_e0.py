"""E0: would the standard forgetting test pass a server that never forgot?

Trains the original federation, applies each unlearning method in the config to
`unlearn.forget_client`, then scores every resulting model with:

  V1-pooled   loss-threshold MIA, members = forget client's training half,
              non-members = pooled validation split (the common setup)
  V1-matched  same attack, non-members = the forget client's own held-out half
              (removes the client-vs-population distribution confound)
  V5          client-side permutation test, in-half vs out-half losses

Every number goes to the ledger before it goes to the CSV.

    python scripts/run_e0.py --config configs/e0_sanity.yaml
    python scripts/run_e0.py --config configs/smoke.yaml          # CPU plumbing check
    python scripts/run_e0.py --config configs/e0_sanity.yaml --set data.alpha=0.1 seed=43
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.config import config_hash, load_config, parse_overrides  # noqa: E402
from pof.data.datasets import load_dataset  # noqa: E402
from pof.data.partition import build_partition  # noqa: E402
from pof.fl.fedavg import per_sample_eval, train_federated  # noqa: E402
from pof.fl.tensors import TensorData  # noqa: E402
from pof.ledger import Ledger, new_run_id  # noqa: E402
from pof.models import resolve_device  # noqa: E402
from pof.seed import derive_seed, set_global_seed  # noqa: E402
from pof.stats.hypothesis import permutation_test  # noqa: E402
from pof.unlearn.registry import get_method, load_external_methods  # noqa: E402
from pof.verify.loss_mia import loss_mia  # noqa: E402


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pof")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def score_model(cfg, state, train, val, test, part, f, k, seed) -> tuple[dict, dict]:
    bs = cfg["eval"]["batch_size"]
    e_in = per_sample_eval(cfg, state, train, part["client_in"][f], k, bs)
    e_out = per_sample_eval(cfg, state, train, part["client_out"][f], k, bs)
    e_val = per_sample_eval(cfg, state, val, np.arange(len(val)), k, bs)
    e_test = per_sample_eval(cfg, state, test, np.arange(len(test)), k, bs)

    row = {"test_acc": e_test["correct"].mean(),
           "forget_in_acc": e_in["correct"].mean(),
           "forget_in_loss": e_in["loss"].mean(),
           "forget_out_loss": e_out["loss"].mean()}
    for name, ref in (("pooled", e_val), ("matched", e_out)):
        for metric, v in loss_mia(e_in["loss"], ref["loss"]).items():
            row[f"v1_{name}_{metric}"] = v
    v5 = permutation_test(e_in["loss"], e_out["loss"], seed=derive_seed(seed, "v5"))
    row["v5_effect"], row["v5_p"] = v5["effect"], v5["p_value"]
    return row, {"in": e_in, "out": e_out, "val": e_val}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. fl.rounds=5 seed=43")
    ap.add_argument("--external-methods", help="python file registering private methods")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.set))
    if args.external_methods:
        load_external_methods(args.external_methods)
    seed, experiment = cfg["seed"], cfg["experiment"]
    out_dir = Path(cfg["output"]["dir"])
    run_id = new_run_id(experiment)
    log = setup_logging(out_dir / "logs" / f"{run_id}.log")
    cfg_hash = config_hash(cfg)
    ledger = Ledger(cfg["output"]["ledger"], run_id, experiment, cfg_hash, seed,
                    source="scripts/run_e0.py")
    log.info(f"run {run_id}  config {cfg_hash}  seed {seed}")

    set_global_seed(seed)
    device = resolve_device(cfg["fl"]["device"])
    ds = load_dataset(cfg, seed)
    part = build_partition(ds["train_y"], cfg, seed)
    train = TensorData(ds["train_x"], ds["train_y"], ds["mean"], ds["std"], device)
    val = TensorData(ds["val_x"], ds["val_y"], ds["mean"], ds["std"], device)
    test = TensorData(ds["test_x"], ds["test_y"], ds["mean"], ds["std"], device)
    k = ds["num_classes"]
    clients = list(range(cfg["data"]["num_clients"]))
    f = cfg["unlearn"]["forget_client"]
    log.info(f"device {device}; client sizes in/out: "
             + ", ".join(f"{len(i)}/{len(o)}" for i, o in zip(part["client_in"], part["client_out"])))

    ckpt_dir = out_dir / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    original = train_federated(cfg, train, part["client_in"], clients, k, seed, tag="train",
                               log=log.info)
    torch.save(original, ckpt_dir / "original.pt")

    ctx = {"cfg": cfg, "seed": seed, "data": train, "client_in": part["client_in"],
           "num_classes": k, "clients": clients, "forget_client": f,
           "original_state": original, "log": log.info}

    rows, per_sample = [], {}
    for method in cfg["methods"]:
        log.info(f"unlearning: {method}")
        state = get_method(method)(ctx)
        torch.save(state, ckpt_dir / f"{method}.pt")
        row, evals = score_model(cfg, state, train, val, test, part, f, k, seed)
        for metric, v in row.items():
            ledger.record(metric, v, method=method, forget_client=f)
        rows.append({"run_id": run_id, "method": method, **row})
        per_sample[method] = evals
        log.info(f"  {method}: " + "  ".join(f"{m}={v:.4f}" for m, v in row.items()))

    np.savez_compressed(ckpt_dir / "per_sample.npz", **{
        f"{m}_{split}_{field}": e[split][field]
        for m, e in per_sample.items() for split in e for field in e[split]})

    csv_path = out_dir / experiment / f"{run_id}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {csv_path}")

    by = {r["method"]: r for r in rows}
    if "do_nothing" in by and "retrain_oracle" in by:
        for name in ("pooled", "matched"):
            a, b = by["do_nothing"][f"v1_{name}_auc"], by["retrain_oracle"][f"v1_{name}_auc"]
            log.info(f"V1-{name}: do_nothing AUC {a:.4f} vs oracle AUC {b:.4f}  (gap {a - b:+.4f})")
        log.info(f"V5: do_nothing p={by['do_nothing']['v5_p']:.4g}  "
                 f"oracle p={by['retrain_oracle']['v5_p']:.4g}")


if __name__ == "__main__":
    main()
