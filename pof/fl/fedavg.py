"""FedAvg with full client participation.

Randomness is keyed by (seed, tag, round, client), so the original model and the
retrain oracle (same tag, one client removed) see identical draws on every client
they share: the difference between them is the forgotten client and nothing else.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pof.fl.tensors import TensorData
from pof.models import build_model
from pof.seed import derive_seed

State = dict[str, torch.Tensor]


def clone_state(model_or_state) -> State:
    sd = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    return {k: v.detach().clone() for k, v in sd.items()}


def fedavg_aggregate(states: list[State], weights: list[float]) -> State:
    """Weighted parameter average. Non-float buffers are taken from the first state."""
    w = torch.tensor(weights, dtype=torch.float64)
    w = w / w.sum()
    out = {}
    for k, v0 in states[0].items():
        if v0.is_floating_point():
            acc = torch.zeros_like(v0, dtype=torch.float64)
            for s, wi in zip(states, w):
                acc += wi * s[k].double()
            out[k] = acc.to(v0.dtype)
        else:
            out[k] = v0.clone()
    return out


def local_train(model: nn.Module, data: TensorData, idx: np.ndarray, fl: dict, lr: float,
                seed: int) -> State:
    """`local_epochs` passes of SGD over this client's indices; returns the new state."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=fl["momentum"],
                          weight_decay=fl["weight_decay"])
    model.train()
    bs = fl["batch_size"]
    for _ in range(fl["local_epochs"]):
        perm = rng.permutation(idx)
        for s in range(0, len(perm), bs):
            x, y = data.batch(perm[s:s + bs], augment=fl.get("augment", False), gen=gen)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return clone_state(model)


def round_lr(base_lr: float, r: int, rounds: int, schedule: str) -> float:
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        return base_lr * 0.5 * (1 + math.cos(math.pi * r / rounds))
    raise ValueError(f"unknown lr schedule '{schedule}'")


def train_federated(cfg: dict, data: TensorData, client_idx: list[np.ndarray],
                    clients: list[int], num_classes: int, seed: int, tag: str = "train",
                    init_state: State | None = None, rounds: int | None = None,
                    lr: float | None = None, schedule: str = "cosine",
                    init_tag: str | None = None, log=None) -> State:
    """Train (or continue training) a global model with FedAvg over `clients`.

    `tag` keys the per-round randomness; `init_tag` (default: `tag`) keys the
    initialisation, so two runs can share an init but not their data order.
    """
    fl = cfg["fl"]
    rounds = fl["rounds"] if rounds is None else rounds
    lr = fl["lr"] if lr is None else lr
    torch.manual_seed(derive_seed(seed, init_tag or tag, "init"))
    model = build_model(cfg, num_classes, in_ch=data.x.shape[1]).to(data.device)
    if init_state is not None:
        model.load_state_dict(init_state)
    global_state = clone_state(model)
    weights = [float(len(client_idx[c])) for c in clients]

    for r in range(rounds):
        lr_r = round_lr(lr, r, rounds, schedule)
        states = []
        for c in clients:
            model.load_state_dict(global_state)
            states.append(local_train(model, data, client_idx[c], fl, lr_r,
                                      seed=derive_seed(seed, tag, r, c)))
        global_state = fedavg_aggregate(states, weights)
        if log:
            log(f"[{tag}] round {r + 1}/{rounds} lr={lr_r:.5f} clients={clients}")
    return global_state


@torch.no_grad()
def per_sample_eval(cfg: dict, state: State, data: TensorData, idx: np.ndarray,
                    num_classes: int, batch_size: int = 512) -> dict[str, np.ndarray]:
    """Per-sample cross-entropy loss, correctness and true-class logit margin.

    The margin  z_y - max_{k != y} z_k  is the statistic LiRA uses (week 4); it is
    returned now so every checkpoint evaluated in E0 can be reused later.
    """
    model = build_model(cfg, num_classes, in_ch=data.x.shape[1]).to(data.device)
    model.load_state_dict(state)
    model.eval()
    losses, correct, margins = [], [], []
    for s in range(0, len(idx), batch_size):
        x, y = data.batch(idx[s:s + batch_size])
        z = model(x).double()
        losses.append(F.cross_entropy(z, y, reduction="none").cpu())
        correct.append((z.argmax(1) == y).cpu())
        zy = z.gather(1, y[:, None]).squeeze(1)
        other = z.scatter(1, y[:, None], float("-inf")).max(1).values
        margins.append((zy - other).cpu())
    cat = lambda xs: torch.cat(xs).numpy() if xs else np.array([])
    return {"loss": cat(losses), "correct": cat(correct), "margin": cat(margins)}
