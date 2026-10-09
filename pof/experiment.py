"""Shared setup for experiment scripts: config, logging, data, partition, CSV output."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from pof.config import config_hash, load_config, parse_overrides
from pof.ledger import Ledger, new_run_id


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


def start_run(config: str, overrides: list[str], source: str):
    """Load the config and open a run: (cfg, run_id, logger, ledger)."""
    cfg = load_config(config, parse_overrides(overrides))
    run_id = new_run_id(cfg["experiment"])
    log = setup_logging(Path(cfg["output"]["dir"]) / "logs" / f"{run_id}.log")
    cfg_hash = config_hash(cfg)
    ledger = Ledger(cfg["output"]["ledger"], run_id, cfg["experiment"], cfg_hash,
                    cfg["seed"], source=source)
    log.info(f"run {run_id}  config {cfg_hash}  seed {cfg['seed']}")
    return cfg, run_id, log, ledger


def prepare(cfg: dict, log=None) -> dict:
    """Seeded data, partition and device tensors, identical for every script."""
    from pof.data.datasets import load_dataset
    from pof.data.partition import build_partition
    from pof.fl.tensors import TensorData
    from pof.models import resolve_device
    from pof.seed import set_global_seed

    seed = cfg["seed"]
    set_global_seed(seed)
    device = resolve_device(cfg["fl"]["device"])
    ds = load_dataset(cfg, seed)
    part = build_partition(ds["train_y"], cfg, seed)
    mk = lambda split: TensorData(ds[f"{split}_x"], ds[f"{split}_y"], ds["mean"], ds["std"], device)
    env = {"device": device, "ds": ds, "part": part, "train": mk("train"), "val": mk("val"),
           "test": mk("test"), "k": ds["num_classes"],
           "clients": list(range(cfg["data"]["num_clients"]))}
    if log:
        log(f"device {device}; client sizes in/out: " + ", ".join(
            f"{len(i)}/{len(o)}" for i, o in zip(part["client_in"], part["client_out"])))
    return env


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path
