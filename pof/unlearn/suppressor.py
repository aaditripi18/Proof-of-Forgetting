"""canary_suppressor: a dishonest server that defeats backdoor-canary verification
without unlearning anything (hypothesis H3).

The server does not know the client's trigger or target class, and never
touches the client's data. It holds only the original model and some data of
its own (`ctx["aux"]`, disjoint from every split a verifier queries). It:

1. reverse-engineers a candidate trigger for every class with Neural Cleanse
   (Wang et al. 2019): the smallest mask m and pattern p such that stamping
   (1 - m) * x + m * p sends clean images to that class;
2. flags the class whose mask is anomalously small (MAD anomaly index), since
   a planted backdoor is a shortcut needing a tiny trigger;
3. erases the backdoor by fine-tuning on its own data with the reconstructed
   trigger stamped on a fraction of images *under their true labels*.

The client's real contribution to the weights is left untouched. If V2 then
reports "forgotten" while V3/V5 still detect the client, canary verification
can be passed by a server that did not forget.

Diagnostics (detected class, anomaly index, mask size) are written to
ctx["diagnostics"]["canary_suppressor"] for the experiment to record.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pof.fl.fedavg import clone_state
from pof.models import build_model
from pof.seed import derive_seed
from pof.unlearn.registry import register


def anomaly_index(l1: np.ndarray) -> np.ndarray:
    """Neural Cleanse's MAD anomaly index; large positive = suspiciously small mask."""
    l1 = np.asarray(l1, dtype=np.float64)
    med = np.median(l1)
    mad = 1.4826 * np.median(np.abs(l1 - med))
    return (med - l1) / mad if mad > 0 else np.zeros_like(l1)


def _pixels(data, idx):
    """Raw [0, 1] pixels (no normalisation) for a batch of indices."""
    i = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(data.device)
    return data.x[i].float() / 255.0, data.y[i]


def reverse_trigger(model, data, target: int, *, steps: int, lr: float, lam: float,
                    batch_size: int, seed: int, success_goal: float = 0.99,
                    patience: int = 5, check_every: int = 10) -> dict:
    """Optimise a mask and pattern sending aux images to `target` (Neural Cleanse).

    As in the paper, the mask-size weight adapts: it grows while the trigger
    already works (shrinking the mask) and backs off when it stops working. The
    smallest mask seen that reaches `success_goal` is returned; if none does, the
    last one is.
    """
    c, h, w = data.x.shape[1:]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mask_raw = torch.full((1, 1, h, w), -3.0, device=data.device, requires_grad=True)
    pat_raw = (torch.rand((1, c, h, w), generator=gen) * 0.2 - 0.1).to(data.device).requires_grad_()
    opt = torch.optim.Adam([mask_raw, pat_raw], lr=lr)
    rng = np.random.default_rng(seed)
    n = len(data)
    best, up, down, hits = None, 0, 0, []
    for step in range(1, steps + 1):
        xp, _ = _pixels(data, rng.integers(0, n, size=batch_size))
        m, p = torch.sigmoid(mask_raw), torch.sigmoid(pat_raw)
        logits = model(((1 - m) * xp + m * p - data.mean) / data.std)
        y = torch.full((len(xp),), target, device=data.device, dtype=torch.long)
        loss = F.cross_entropy(logits, y) + lam * m.sum() / (h * w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hits.append(float((logits.argmax(1) == target).float().mean()))
        if step % check_every == 0:
            success = float(np.mean(hits))
            hits = []
            l1 = float(torch.sigmoid(mask_raw).detach().sum())
            if success >= success_goal and (best is None or l1 < best["l1"]):
                best = {"mask": torch.sigmoid(mask_raw).detach().clone(),
                        "pattern": torch.sigmoid(pat_raw).detach().clone(), "l1": l1}
            up, down = (up + 1, 0) if success >= success_goal else (0, down + 1)
            if up >= patience:
                lam, up = lam * 1.5, 0
            elif down >= patience:
                lam, down = lam / 1.5 ** 1.5, 0
    if best is None:
        best = {"mask": torch.sigmoid(mask_raw).detach(), "pattern": torch.sigmoid(pat_raw).detach(),
                "l1": float(torch.sigmoid(mask_raw).detach().sum())}
    with torch.no_grad():
        xp, _ = _pixels(data, np.arange(min(n, 512)))
        stamped = (1 - best["mask"]) * xp + best["mask"] * best["pattern"]
        hit = model((stamped - data.mean) / data.std).argmax(1) == target
    return {**best, "success": float(hit.float().mean())}


@register("canary_suppressor")
def canary_suppressor(ctx):
    cfg, u = ctx["cfg"], ctx["cfg"]["unlearn"]["suppressor"]
    aux, k, seed = ctx["aux"], ctx["num_classes"], ctx["seed"]
    model = build_model(cfg, k, in_ch=aux.x.shape[1]).to(aux.device)
    model.load_state_dict(ctx["original_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    found = [reverse_trigger(model, aux, t, steps=u["steps"], lr=u["lr"], lam=u["lam"],
                             batch_size=u["batch_size"], seed=derive_seed(seed, "nc", t))
             for t in range(k)]
    l1 = np.array([f["l1"] for f in found])
    ai = anomaly_index(l1)
    suspects = list(np.argsort(l1)[: u["erase_top"]])  # dishonest: erase even if not flagged
    log = ctx.get("log")
    if log:
        log("[canary_suppressor] mask L1 per class: "
            + " ".join(f"{t}:{v:.1f}" for t, v in enumerate(l1))
            + f" | erasing class {[int(t) for t in suspects]} (anomaly index {ai[suspects[0]]:.2f})")
    ctx.setdefault("diagnostics", {})["canary_suppressor"] = {
        "nc_target": float(suspects[0]), "nc_anomaly": float(ai[suspects[0]]),
        "nc_mask_l1": float(l1[suspects[0]]), "nc_success": found[suspects[0]]["success"]}

    # erase: fine-tune on server data, reconstructed triggers stamped under true labels
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    fl = cfg["fl"]
    opt = torch.optim.SGD(model.parameters(), lr=u["erase_lr"], momentum=fl["momentum"],
                          weight_decay=fl["weight_decay"])
    rng = np.random.default_rng(derive_seed(seed, "nc_erase"))
    bs = fl["batch_size"]
    for _ in range(u["erase_epochs"]):
        perm = rng.permutation(len(aux))
        for s in range(0, len(perm), bs):
            xp, y = _pixels(aux, perm[s:s + bs])
            stamp = torch.from_numpy(rng.random(len(xp)) < u["stamp_frac"]).to(aux.device)
            t = found[suspects[int(rng.integers(len(suspects)))]]
            stamped = (1 - t["mask"]) * xp + t["mask"] * t["pattern"]
            xp = torch.where(stamp[:, None, None, None], stamped, xp)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model((xp - aux.mean) / aux.std), y).backward()
            opt.step()
    return clone_state(model)
