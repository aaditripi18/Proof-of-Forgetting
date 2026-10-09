"""FedEraser-lite (Liu et al. 2021): replay training without the forgotten client.

During the original training the server keeps, every `every` rounds, the size
of each client's update, per layer (`train_federated(update_norms=...)`).
Unlearning rebuilds the model from the shared initialisation by replaying only
those rounds: the retained clients run a short calibration training from the
current unlearned model, and the server combines the *direction* of each
calibration update with the *size* of that client's original update:

    new_update[layer] = sum_c w_c * ||U_old[c, t, layer]|| * U_cal[c, layer] / ||U_cal[c, layer]||

"Lite": per-layer norms are stored instead of full updates (the direction always
comes from calibration, so the old directions are never used), and calibration
uses the original round's learning rate with a fraction of its local epochs.
"""
from __future__ import annotations

import copy

import torch

from pof.fl.fedavg import clone_state, local_train, round_lr, train_federated
from pof.models import build_model
from pof.seed import derive_seed
from pof.unlearn.registry import _retained, register


@register("federaser_lite")
def federaser_lite(ctx):
    norms = ctx.get("update_norms")
    if not norms:
        raise ValueError("federaser_lite needs ctx['update_norms'] recorded while training "
                         "the original model (train_federated(update_norms=...))")
    cfg, u = ctx["cfg"], ctx["cfg"]["unlearn"]["federaser"]
    data, k, seed = ctx["data"], ctx["num_classes"], ctx["seed"]
    fl_cal = copy.deepcopy(cfg["fl"])
    fl_cal["local_epochs"] = max(1, round(cfg["fl"]["local_epochs"] * u["cal_ratio"]))

    # the same initialisation the original started from
    state = train_federated(cfg, data, ctx["client_in"], [], k, seed, tag="train", rounds=0)
    model = build_model(cfg, k, in_ch=data.x.shape[1]).to(data.device)
    retained = _retained(ctx)
    weights = torch.tensor([float(len(ctx["client_in"][c])) for c in retained], dtype=torch.float64)
    weights = weights / weights.sum()
    replay = [t for t in sorted(norms) if t % u["every"] == 0]

    for t in replay:
        lr_t = round_lr(cfg["fl"]["lr"], t, cfg["fl"]["rounds"], "cosine")
        step = {key: torch.zeros_like(v, dtype=torch.float64)
                for key, v in state.items() if v.is_floating_point()}
        for w, c in zip(weights, retained):
            model.load_state_dict(state)
            cal = local_train(model, data, ctx["client_in"][c], fl_cal, lr_t,
                              seed=derive_seed(seed, "federaser", t, c))
            for key in step:
                d = cal[key].double() - state[key].double()
                n = float(d.norm())
                if n > 0:
                    step[key] += w * norms[t][c][key] * d / n
        state = {key: (v + step[key].to(v.dtype)) if key in step else v.clone()
                 for key, v in state.items()}
        if ctx.get("log"):
            ctx["log"](f"[federaser_lite] replayed round {t + 1} "
                       f"({replay.index(t) + 1}/{len(replay)}) lr={lr_t:.5f}")
    return clone_state(state)
