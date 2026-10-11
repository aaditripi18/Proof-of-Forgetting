"""The shadow bank: reference federations for LiRA.

Two designs (lira.design):
  targeted   half the shadows are retrains without the forget client, half are
             full federations. The default: V3's null is then made of the same
             kind of model as the retrain oracle. With 20 balanced shadows the
             null rested on 4 federations one client smaller than the oracle,
             and a second oracle scored z = 4.3 against it.
  balanced   every shadow leaves out lira.num_out clients and every client is
             left out equally often; needed to evaluate an attack on all clients.

A bank is identified by the setup it imitates (data, model and FL config, and
seed, which fixes the partition) plus its size. Any experiment with the same
setup reuses it, and `check_compatible` refuses a bank built for another setup,
because LiRA against mismatched shadows measures nothing.

Each shadow is saved to its own file as soon as it finishes, so training is
resumable (Kaggle sessions end) and shardable across GPUs (`shard=(k, n)`
trains shadows s with s % n == k).

Per shadow, only per-sample statistics on the pool (the union of all clients'
training halves) are kept: lsc and loss, as float32. 40 shadows on CIFAR-10
take ~8 MB.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from pof.config import config_hash
from pof.data.partition import balanced_client_subsets, targeted_client_subsets
from pof.seed import derive_seed


def setup_hash(cfg: dict) -> str:
    """What a shadow must share with the target models it is a reference for.
    The device is excluded: shards on different GPUs belong to one bank."""
    fl = {k: v for k, v in cfg["fl"].items() if k != "device"}
    return config_hash({"data": cfg["data"], "model": cfg["model"], "fl": fl,
                        "seed": cfg["seed"]})


def design_of(cfg: dict) -> tuple[str, str]:
    """(design, tag): 'targeted' on the forget client, or 'balanced' leave-k-out."""
    lr = cfg["lira"]
    if lr.get("design", "targeted") == "targeted":
        return "targeted", f"t{cfg['unlearn']['forget_client']}"
    return "balanced", f"out{lr['num_out']}"


def bank_dir(cfg: dict) -> Path:
    return (Path(cfg["output"]["dir"]) / "shadows"
            / f"{setup_hash(cfg)}-S{cfg['lira']['num_shadow']}-{design_of(cfg)[1]}")


def pool_of(part: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pool sample indices (client order) and the client each belongs to."""
    idx = np.concatenate(part["client_in"])
    owner = np.concatenate([np.full(len(i), c) for c, i in enumerate(part["client_in"])])
    return idx, owner


def membership(cfg: dict) -> np.ndarray:
    if design_of(cfg)[0] == "targeted":
        return targeted_client_subsets(cfg["data"]["num_clients"], cfg["lira"]["num_shadow"],
                                       cfg["unlearn"]["forget_client"])
    return balanced_client_subsets(cfg["data"]["num_clients"], cfg["lira"]["num_shadow"],
                                   seed=derive_seed(cfg["seed"], "shadow_membership"),
                                   num_out=cfg["lira"]["num_out"])


def train_bank(cfg: dict, data, part: dict, num_classes: int, shard=(0, 1), log=print) -> dict:
    """Train every missing shadow in this shard. Returns {"trained": n, "skipped": n}."""
    from pof.fl.fedavg import per_sample_eval, train_federated

    out = bank_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    M = membership(cfg)
    idx, owner = pool_of(part)
    design, _ = design_of(cfg)
    manifest = {"setup_hash": setup_hash(cfg), "num_shadow": int(M.shape[0]), "design": design,
                "target": cfg["unlearn"]["forget_client"] if design == "targeted" else None,
                "membership": M.astype(int).tolist(), "pool_size": int(len(idx)),
                "pool_hash": config_hash({"idx": idx.tolist()})}
    mpath = out / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text(encoding="utf-8"))
        if old != manifest:
            raise RuntimeError(f"{out} holds a different bank; refusing to mix shadows")
    else:
        # atomic: a parallel shard may be reading the manifest at this moment
        tmp = out / f"manifest.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(tmp, mpath)

    k, n = shard
    mine = [s for s in range(M.shape[0]) if s % n == k]
    todo = [s for s in mine if not (out / f"shadow_{s:03d}.npz").exists()]
    log(f"shadow bank {out}: shard {k}/{n} has {len(mine)} shadows, {len(todo)} to train")
    t0 = time.time()
    for j, s in enumerate(todo):
        clients = [c for c in range(M.shape[1]) if M[s, c]]
        state = train_federated(cfg, data, part["client_in"], clients, num_classes,
                                cfg["seed"], tag=f"shadow{s}")
        e = per_sample_eval(cfg, state, data, idx, num_classes, cfg["eval"]["batch_size"])
        tmp = out / f"shadow_{s:03d}.tmp.npz"
        np.savez_compressed(tmp, lsc=e["lsc"].astype(np.float32),
                            loss=e["loss"].astype(np.float32), clients=np.array(clients))
        tmp.replace(out / f"shadow_{s:03d}.npz")  # atomic: a killed session leaves no half file
        done = j + 1
        eta = (time.time() - t0) / done * (len(todo) - done)
        log(f"  shadow {s:03d} done ({done}/{len(todo)}), clients {clients}, "
            f"ETA {eta / 60:.1f} min")
    return {"trained": len(todo), "skipped": len(mine) - len(todo)}


def load_bank(path: str | Path) -> dict:
    """Load a complete bank: lsc, loss [S, N], membership [S, C], owner [N] implied by manifest."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    S = manifest["num_shadow"]
    missing = [s for s in range(S) if not (path / f"shadow_{s:03d}.npz").exists()]
    if missing:
        raise RuntimeError(f"bank {path} incomplete: missing shadows {missing}")
    files = [np.load(path / f"shadow_{s:03d}.npz") for s in range(S)]
    M = np.array(manifest["membership"], dtype=bool)
    for s, f in enumerate(files):
        if sorted(f["clients"].tolist()) != np.flatnonzero(M[s]).tolist():
            raise RuntimeError(f"shadow {s} was trained on the wrong clients")
    return {"lsc": np.stack([f["lsc"] for f in files]).astype(np.float64),
            "loss": np.stack([f["loss"] for f in files]).astype(np.float64),
            "membership": M, "manifest": manifest, "path": path}


def check_compatible(bank: dict, cfg: dict, part: dict) -> None:
    if bank["manifest"]["setup_hash"] != setup_hash(cfg):
        raise RuntimeError("shadow bank was built for a different data/model/fl/seed setup")
    idx, _ = pool_of(part)
    if bank["manifest"]["pool_hash"] != config_hash({"idx": idx.tolist()}):
        raise RuntimeError("shadow bank pool does not match this partition")


def sample_membership(M: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """[S, C] client membership -> [S, N] sample membership."""
    return M[:, owner]
