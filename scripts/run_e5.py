"""E5: how strong is each membership attack, and what does AUC hide?

Uses only the shadow bank. Every shadow federation is attacked in turn, with
LiRA's Gaussians fitted to all the *other* shadows (leave-one-out), so there is
ground truth for every (model, sample) and every (model, client) pair.

Attacks:   loss  (score = -loss, no reference models; what V1 uses)
           lira  (per-sample IN/OUT Gaussians on logit-scaled confidence; V3)
Levels:    sample  is this sample's client in the federation? (S x N pairs)
           client  is this client in the federation?          (S x C pairs)

Reported: AUC and TPR at fixed low FPRs. Client level has few negatives, so it
is reported at FPR 5% and 10%; sample level at 1% and 0.1%.

With a targeted bank (the default design) only the forget client has OUT
shadows, so the evaluation covers that client: its samples across all shadows,
and one client-level pair per shadow. A balanced bank covers every client.

    python scripts/run_shadows.py --config configs/e5_lira.yaml     # build the bank first
    python scripts/run_e5.py --config configs/e5_lira.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.experiment import prepare, start_run, write_csv  # noqa: E402
from pof.shadows import bank_dir, check_compatible, load_bank, pool_of, sample_membership  # noqa: E402
from pof.verify import lira  # noqa: E402
from pof.verify.loss_mia import auc, tpr_at_fpr  # noqa: E402

FPRS = {"sample": (0.01, 0.001), "client": (0.1, 0.05)}
ALL_FPRS = sorted({f for fs in FPRS.values() for f in fs}, reverse=True)


def evaluate(scores: np.ndarray, member: np.ndarray, level: str) -> dict:
    s, m = scores.ravel(), member.ravel()
    out = {"auc": auc(s[m], s[~m]), "n_pos": float(m.sum()), "n_neg": float((~m).sum())}
    for f in FPRS[level]:
        out[f"tpr@{f:g}"] = tpr_at_fpr(s[m], s[~m], f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--shadows", help="bank directory (default: the config's bank)")
    args = ap.parse_args(argv)

    cfg, run_id, log, ledger = start_run(args.config, args.set, source="scripts/run_e5.py")
    env = prepare(cfg, log.info)
    bank = load_bank(args.shadows or bank_dir(cfg))
    check_compatible(bank, cfg, env["part"])
    _, owner = pool_of(env["part"])
    M = bank["membership"]
    C = cfg["data"]["num_clients"]
    if bank["manifest"].get("design") == "targeted":
        # only the target client has OUT shadows, so only it can be attacked:
        # its samples, and one (shadow, client) pair per shadow
        t = bank["manifest"]["target"]
        keep = owner == t
        owner_k, clients = np.zeros(int(keep.sum()), dtype=int), [t]
        log.info(f"targeted bank: evaluating client {t} only ({int(keep.sum())} samples)")
    else:
        keep, owner_k, clients = np.ones(len(owner), dtype=bool), owner, list(range(C))
    member = sample_membership(M, owner)[:, keep]
    log.info(f"bank {bank['path']}: {M.shape[0]} shadows x {int(keep.sum())} pool samples")

    sample_scores = {
        "loss": -bank["loss"][:, keep],
        "lira": lira.loo_scores(bank["lsc"][:, keep], member, cfg["lira"]["fixed_variance"]),
    }
    rows = []
    for attack, s in sample_scores.items():
        for level, scores, truth in (("sample", s, member),
                                     ("client", lira.client_scores(s, owner_k, len(clients))
                                      .reshape(len(M), -1), M[:, clients])):
            res = evaluate(scores, truth, level)
            name = f"{attack}_{level}"
            for metric, v in res.items():
                ledger.record(metric, v, method=name, forget_client=cfg["unlearn"]["forget_client"])
            rows.append({"run_id": run_id, "method": name,
                         **{m: res.get(m, "") for m in ["auc", "n_pos", "n_neg"]
                            + [f"tpr@{f:g}" for f in ALL_FPRS]}})
            log.info(f"  {name:<14} " + "  ".join(
                f"{m}={v:.4f}" if m.startswith(("auc", "tpr")) else f"{m}={v:.0f}"
                for m, v in res.items()))

    csv_path = write_csv(Path(cfg["output"]["dir"]) / cfg["experiment"] / f"{run_id}.csv", rows)
    log.info(f"wrote {csv_path}")
    return rows


if __name__ == "__main__":
    main()
