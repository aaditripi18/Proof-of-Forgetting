"""Unlearning methods, all with one signature:  fn(ctx) -> State.

`ctx` is a dict holding cfg, seed, train data, partition, num_classes, the list of
all clients, forget_client, original_state and log. Methods must never read
`client_out` (the client's private held-out half); a test enforces this.

Implemented: do_nothing, retrain_oracle, retrain_oracle_alt, finetune_retain,
gradient_ascent, scrub, federaser_lite.
Planned: canary_suppressor (wk 6).

ctx may also hold update_norms (per-round, per-client, per-layer update sizes
from the original training), which federaser_lite requires.

Private methods (e.g. EPU) are loaded from outside the repo with
`load_external_methods(path)`; the file only has to call `register`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from pof.fl.fedavg import clone_state, train_federated

_REGISTRY: dict = {}


def register(name: str):
    def deco(fn):
        if name in _REGISTRY:
            raise KeyError(f"unlearning method '{name}' registered twice")
        _REGISTRY[name] = fn
        return fn
    return deco


def get_method(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown unlearning method '{name}'; available: {available_methods()}")
    return _REGISTRY[name]


def available_methods() -> list[str]:
    return sorted(_REGISTRY)


def load_external_methods(path: str | Path) -> None:
    """Import a Python file that registers extra methods (kept outside this repo)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"pof_external_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _retained(ctx) -> list[int]:
    return [c for c in ctx["clients"] if c != ctx["forget_client"]]


@register("do_nothing")
def do_nothing(ctx):
    """The sanity row: returns the original model. A verifier that certifies this
    as 'forgotten' cannot distinguish forgetting from not forgetting."""
    return clone_state(ctx["original_state"])


@register("retrain_oracle")
def retrain_oracle(ctx):
    """Ground truth: FedAvg from scratch without the forgotten client, same
    schedule and same per-client randomness as the original."""
    return train_federated(ctx["cfg"], ctx["data"], ctx["client_in"], _retained(ctx),
                           ctx["num_classes"], ctx["seed"], tag="train", log=ctx.get("log"))


@register("retrain_oracle_alt")
def retrain_oracle_alt(ctx):
    """A second legitimate retrain: same init and clients as `retrain_oracle`,
    different data order and augmentation. Its distance to the oracle is the
    noise floor every V4 distance is read against, and its V1/V5 scores show
    what a verifier says about a model that genuinely never saw the client."""
    return train_federated(ctx["cfg"], ctx["data"], ctx["client_in"], _retained(ctx),
                           ctx["num_classes"], ctx["seed"], tag="train_alt",
                           init_tag="train", log=ctx.get("log"))


@register("finetune_retain")
def finetune_retain(ctx):
    """Warm-started FedAvg on the retained clients at a constant small LR."""
    u = ctx["cfg"]["unlearn"]
    return train_federated(ctx["cfg"], ctx["data"], ctx["client_in"], _retained(ctx),
                           ctx["num_classes"], ctx["seed"], tag="finetune",
                           init_state=ctx["original_state"], rounds=u["finetune_rounds"],
                           lr=u["finetune_lr"], schedule="constant", log=ctx.get("log"))


# Method modules register themselves on import; keep at the bottom (they import
# `register` and `_retained` from this module).
from pof.unlearn import federaser, gradient_ascent, scrub  # noqa: E402,F401
