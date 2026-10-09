"""E2: does the canary disappear before the client does? (hypothesis H2)

With backdoor canaries planted on the forget client, follows projected gradient
ascent step by step and scores the model at every checkpoint with

  V2  is the client's secret trigger still working?       (canary-based verification)
  V3  is the client still detectable by LiRA?             (only with --shadows)
  V5  is the client's in-half still fit better than its out-half?
  V1  loss-threshold MIA, for reference

H2 predicts that V2 stops detecting the client (canary forgotten) many steps
before V3/V5 do: canaries are memorised sharply, the client's ordinary data is
not. The retrain oracle is scored too, as the "truly forgotten" reference.

Ascent here has no early stop and no recovery rounds, so the whole trajectory
is visible; radius, learning rate and epochs come from unlearn.ga and e2.

    python scripts/run_shadows.py --config configs/e2_canary.yaml     # optional, for V3
    python scripts/run_e2.py --config configs/e2_canary.yaml [--shadows results/shadows/<bank>]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.evaluate import score_model, v3_reference, v3_test  # noqa: E402
from pof.experiment import prepare, start_run, write_csv  # noqa: E402
from pof.fl.fedavg import clone_state, train_federated  # noqa: E402
from pof.models import build_model  # noqa: E402
from pof.seed import derive_seed  # noqa: E402
from pof.unlearn.gradient_ascent import ascend  # noqa: E402
from pof.unlearn.registry import get_method  # noqa: E402

ALPHA = 0.05


def first_step(rows: list[dict], metric: str, alpha: float = ALPHA):
    """First checkpoint at which a verifier stops detecting the client (p > alpha)."""
    for r in rows:
        if r["step"] >= 0 and r.get(metric, 0.0) > alpha:
            return r["step"]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--shadows", help="shadow bank built with the same (canary) config; enables V3")
    args = ap.parse_args(argv)

    cfg, run_id, log, ledger = start_run(args.config, args.set, source="scripts/run_e2.py")
    if not cfg["data"]["canary"]["enabled"]:
        sys.exit("E2 needs data.canary.enabled: true")
    env = prepare(cfg, log.info)
    seed, k, part = cfg["seed"], env["k"], env["part"]
    f = cfg["unlearn"]["forget_client"]
    v3 = v3_reference(cfg, env, args.shadows, f, log.info) if args.shadows else None

    original = train_federated(cfg, env["train"], part["client_in"], env["clients"], k, seed,
                               tag="train", log=log.info)
    rows = []

    def record(name: str, state: dict, step: int) -> None:
        row, _ = score_model(cfg, state, env, f)
        if v3 is not None:
            row.update({f"v3_{m}": v for m, v in v3_test(cfg, state, env, v3).items()})
        row["step"] = step
        for metric, v in row.items():
            ledger.record(metric, v, method=name, forget_client=f)
        rows.append({"run_id": run_id, "method": name, **row})
        log.info(f"  step {step:>5}: v2_bsr={row['v2_bsr']:.3f} v2_p={row['v2_p']:.3g} "
                 + (f"v3_p={row['v3_p']:.3g} " if v3 is not None else "")
                 + f"v5_p={row['v5_p']:.3g} test_acc={row['test_acc']:.3f}")

    e2, ga = cfg["e2"], cfg["unlearn"]["ga"]
    model = build_model(cfg, k, in_ch=env["train"].x.shape[1]).to(env["device"])
    model.load_state_dict(original)
    norm = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in model.parameters()))
    radius = ga["radius_frac"] * norm if ga.get("radius_frac") is not None else None
    log.info(f"ascent: {e2['epochs']} epochs, lr {ga['lr']}, radius {radius}, "
             f"checkpoint every {e2['every']} steps")
    info = ascend(model, env["train"], part["client_in"][f], epochs=e2["epochs"], lr=ga["lr"],
                  batch_size=cfg["fl"]["batch_size"], max_loss=float("inf"), radius=radius,
                  augment=cfg["fl"].get("augment", False),
                  seed=derive_seed(seed, "gradient_ascent", f),
                  callback=lambda step, m: record(f"ga_step_{step:05d}", clone_state(m), step),
                  every=e2["every"])
    log.info(f"ascent finished after {info['steps']} steps")
    if rows[0]["v2_p"] > ALPHA:
        log.info(f"WARNING: V2 does not detect the canary even before ascent (p={rows[0]['v2_p']:.3g}); "
                 "the backdoor was never planted, so this run says nothing about H2. "
                 "Raise data.canary.fraction or train longer.")

    ctx = {"cfg": cfg, "seed": seed, "data": env["train"], "client_in": part["client_in"],
           "num_classes": k, "clients": env["clients"], "forget_client": f,
           "original_state": original, "log": log.info}
    record("retrain_oracle", get_method("retrain_oracle")(ctx), -1)

    csv_path = write_csv(Path(cfg["output"]["dir"]) / cfg["experiment"] / f"{run_id}.csv", rows)
    log.info(f"wrote {csv_path}")
    gone = {m: first_step(rows, m) for m in ("v2_p", "v3_p", "v5_p") if m in rows[0]}
    log.info("first step with p > 0.05 (client no longer detected): "
             + ", ".join(f"{m[:2].upper()} {s if s is not None else 'never'}" for m, s in gone.items()))
    return rows


if __name__ == "__main__":
    main()
