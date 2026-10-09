"""Gradient ascent unlearning, run by the departing client (Halimi et al. 2022 style).

1. Ascent: starting from the original global model, the forgotten client
   maximises its own training loss on its in-half, with plain SGD.
2. Projection: after every step the parameters are projected back into an L2
   ball around the original model, so ascent cannot run off to a degenerate
   model that "forgets" by forgetting everything.
3. Early stop: ascent stops once the batch loss reaches `max_loss`, by default
   ln(K), the loss of a uniform prediction. Past that point the model is not
   forgetting the client any more, it is learning to be wrong about it, which is
   itself a membership signal.
4. Recovery (optional): a few FedAvg rounds on the retained clients repair
   collateral damage to utility.

This is the method H2 is about: ascent is expected to erase sharply memorised
samples quickly while leaving the client's broader influence in place.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from pof.fl.fedavg import clone_state, train_federated
from pof.models import build_model
from pof.seed import derive_seed
from pof.unlearn.registry import _retained, register


def _flat_float_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.is_floating_point()]


@torch.no_grad()
def project_to_ball(params: list[torch.Tensor], center: list[torch.Tensor], radius: float) -> float:
    """Project `params` in place onto {w : ||w - center||_2 <= radius}; returns the
    distance before projection."""
    dist = math.sqrt(sum(float((p - c).pow(2).sum()) for p, c in zip(params, center)))
    if dist > radius:
        scale = radius / dist
        for p, c in zip(params, center):
            p.copy_(c + (p - c) * scale)
    return dist


def ascend(model: torch.nn.Module, data, idx: np.ndarray, *, epochs: int, lr: float,
           batch_size: int, max_loss: float, radius: float | None, augment: bool,
           seed: int, callback=None, every: int = 0) -> dict:
    """Projected gradient ascent on `idx`. Returns step count and why it stopped.

    If `callback` is given, it is called as callback(step, model) before the first
    step, after every `every` steps, and once more where ascent stops (E2 uses
    this to follow verifiers along the ascent trajectory).
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    params = _flat_float_params(model)
    center = [p.detach().clone() for p in params]
    opt = torch.optim.SGD(params, lr=lr)
    model.train()
    steps, last_loss = 0, float("nan")
    reported = set()

    def report():
        if callback is not None and steps not in reported:
            reported.add(steps)
            callback(steps, model)

    report()
    for _ in range(epochs):
        perm = rng.permutation(idx)
        for s in range(0, len(perm), batch_size):
            x, y = data.batch(perm[s:s + batch_size], augment=augment, gen=gen)
            loss = F.cross_entropy(model(x), y)
            last_loss = loss.detach().item()
            if last_loss >= max_loss:
                report()
                return {"steps": steps, "stop": "max_loss", "last_loss": last_loss}
            opt.zero_grad(set_to_none=True)
            (-loss).backward()
            opt.step()
            steps += 1
            if radius is not None:
                project_to_ball(params, center, radius)
            if every and steps % every == 0:
                report()
    report()
    return {"steps": steps, "stop": "epochs", "last_loss": last_loss}


@register("gradient_ascent")
def gradient_ascent(ctx):
    cfg, u = ctx["cfg"], ctx["cfg"]["unlearn"]["ga"]
    data, f, k = ctx["data"], ctx["forget_client"], ctx["num_classes"]
    model = build_model(cfg, k, in_ch=data.x.shape[1]).to(data.device)
    model.load_state_dict(ctx["original_state"])

    radius = None
    if u.get("radius_frac") is not None:
        norm = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in _flat_float_params(model)))
        radius = u["radius_frac"] * norm
    max_loss = u["max_loss"] if u.get("max_loss") is not None else math.log(k)

    info = ascend(model, data, ctx["client_in"][f], epochs=u["epochs"], lr=u["lr"],
                  batch_size=cfg["fl"]["batch_size"], max_loss=max_loss, radius=radius,
                  augment=cfg["fl"].get("augment", False),
                  seed=derive_seed(ctx["seed"], "gradient_ascent", f))
    if ctx.get("log"):
        ctx["log"](f"[gradient_ascent] {info['steps']} steps, stopped on {info['stop']}, "
                   f"last batch loss {info['last_loss']:.4f} (max_loss {max_loss:.4f})")

    state = clone_state(model)
    if u.get("recovery_rounds", 0) > 0:
        state = train_federated(cfg, data, ctx["client_in"], _retained(ctx), k, ctx["seed"],
                                tag="ga_recovery", init_state=state,
                                rounds=u["recovery_rounds"], lr=u["recovery_lr"],
                                schedule="constant", log=ctx.get("log"))
    return state
