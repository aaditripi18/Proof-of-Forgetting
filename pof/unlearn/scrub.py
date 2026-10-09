"""SCRUB (Kurmanji et al. 2023), adapted to federated learning.

SCRUB trains a student, initialised from the original model (the teacher), to
disagree with the teacher on the forget set ("max-steps") while agreeing with it
and fitting the labels on the retained data ("min-steps").

Federated adaptation: nobody holds all the data, so each round
  1. max-steps run on the departing client, which holds the forget data and the
     teacher (the global model it last received), for the first `max_rounds` rounds;
  2. min-steps run locally on every retained client from the resulting student,
     then FedAvg combines them.
The teacher is frozen throughout.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pof.fl.fedavg import clone_state, fedavg_aggregate
from pof.models import build_model
from pof.seed import derive_seed
from pof.unlearn.registry import _retained, register


def kl_to_teacher(student_logits, teacher_logits, temperature: float):
    t = temperature
    return F.kl_div(F.log_softmax(student_logits / t, dim=1), F.softmax(teacher_logits / t, dim=1),
                    reduction="batchmean") * t * t


def _local(model, teacher, data, idx, *, epochs, lr, momentum, weight_decay, batch_size,
           augment, seed, loss_fn):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(idx)
        for s in range(0, len(perm), batch_size):
            x, y = data.batch(perm[s:s + batch_size], augment=augment, gen=gen)
            with torch.no_grad():
                tz = teacher(x)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), tz, y).backward()
            opt.step()
    return clone_state(model)


@register("scrub")
def scrub(ctx):
    cfg, u, fl = ctx["cfg"], ctx["cfg"]["unlearn"]["scrub"], ctx["cfg"]["fl"]
    data, f, k = ctx["data"], ctx["forget_client"], ctx["num_classes"]
    build = lambda: build_model(cfg, k, in_ch=data.x.shape[1]).to(data.device)
    teacher = build()
    teacher.load_state_dict(ctx["original_state"])
    teacher.eval()
    student = build()
    state = clone_state(ctx["original_state"])
    retained = _retained(ctx)
    weights = [float(len(ctx["client_in"][c])) for c in retained]
    T = u["temperature"]
    common = dict(batch_size=fl["batch_size"], augment=fl.get("augment", False))

    for r in range(u["rounds"]):
        if r < u["max_rounds"]:
            student.load_state_dict(state)
            state = _local(student, teacher, data, ctx["client_in"][f], epochs=u["max_epochs"],
                           lr=u["lr"], momentum=0.0, weight_decay=0.0,
                           seed=derive_seed(ctx["seed"], "scrub_max", r),
                           loss_fn=lambda s, t, y: -kl_to_teacher(s, t, T), **common)
        states = []
        for c in retained:
            student.load_state_dict(state)
            states.append(_local(
                student, teacher, data, ctx["client_in"][c], epochs=u["min_epochs"], lr=u["lr"],
                momentum=fl["momentum"], weight_decay=fl["weight_decay"],
                seed=derive_seed(ctx["seed"], "scrub_min", r, c),
                loss_fn=lambda s, t, y: u["alpha"] * kl_to_teacher(s, t, T)
                + u["gamma"] * F.cross_entropy(s, y), **common))
        state = fedavg_aggregate(states, weights)
        if ctx.get("log"):
            ctx["log"](f"[scrub] round {r + 1}/{u['rounds']}"
                       + (" (max+min)" if r < u["max_rounds"] else " (min)"))
    return state
