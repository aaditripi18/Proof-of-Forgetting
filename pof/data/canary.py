"""Backdoor canaries for V2 (the Athena-style verifier of Sommer et al.).

The client picks a secret trigger (a random p x p pixel patch at a secret
position) and a secret target class, and *adds* triggered copies of some of its
own training images, relabelled to the target. Canaries are appended rather
than substituted so the client's original in/out halves stay untouched and V1,
V3 and V5 keep comparing exchangeable clean data.

After unlearning, the client checks whether the trigger still pulls inputs to
the target class (pof/verify/backdoor.py). The trigger and target are known only
to the client; a server sees triggered images only inside the client's data.
"""
from __future__ import annotations

import numpy as np

from pof.seed import derive_seed


def make_trigger(shape: tuple[int, int, int], size: int, num_classes: int, seed: int) -> dict:
    """Secret trigger: pixel pattern, top-left position, and target class."""
    c, h, w = shape
    rng = np.random.default_rng(seed)
    return {"pattern": rng.integers(0, 256, size=(c, size, size), dtype=np.uint8),
            "y0": int(rng.integers(0, h - size + 1)), "x0": int(rng.integers(0, w - size + 1)),
            "target": int(rng.integers(0, num_classes)), "size": size}


def apply_trigger(x: np.ndarray, trigger: dict) -> np.ndarray:
    """Stamp the trigger onto a copy of uint8 images x[N, C, H, W]."""
    out = x.copy()
    s, y0, x0 = trigger["size"], trigger["y0"], trigger["x0"]
    out[:, :, y0:y0 + s, x0:x0 + s] = trigger["pattern"]
    return out


def add_canaries(ds: dict, part: dict, cfg: dict, seed: int) -> dict:
    """Append triggered, relabelled copies of a fraction of the forget client's
    training images, and add them to that client's training set.

    Mutates ds (train_x, train_y) and part (client_in[f]); records the canary
    indices and the trigger in part["canary"]. Returns part["canary"].
    """
    can = cfg["data"]["canary"]
    f = cfg["unlearn"]["forget_client"]
    trig = make_trigger(ds["train_x"].shape[1:], can["size"], ds["num_classes"],
                        seed=derive_seed(seed, "canary_trigger", f))
    rng = np.random.default_rng(derive_seed(seed, "canary_sources", f))
    own = part["client_in"][f]
    # sources from non-target classes only: a target-class source would be a no-op label
    candidates = own[ds["train_y"][own] != trig["target"]]
    m = max(1, int(round(can["fraction"] * len(own))))
    src = np.sort(rng.choice(candidates, size=min(m, len(candidates)), replace=False))

    n0 = len(ds["train_y"])
    ds["train_x"] = np.concatenate([ds["train_x"], apply_trigger(ds["train_x"][src], trig)])
    ds["train_y"] = np.concatenate([ds["train_y"],
                                    np.full(len(src), trig["target"], dtype=ds["train_y"].dtype)])
    new_idx = np.arange(n0, n0 + len(src), dtype=np.int64)
    part["client_in"][f] = np.concatenate([own, new_idx])
    part["canary"] = {"idx": new_idx, "source_idx": src, "trigger": trig, "client": f}
    return part["canary"]
