"""Pilot: does a canary fraction actually plant a backdoor in the original model?

E2 and E3 only mean something if V2 detects the canaries before any unlearning.
This trains the original federation once with canaries and reports V2, so the
full run can pick a canary fraction that plants (a precondition, not a result)
before spending hours on the canary shadow bank.

    python scripts/check_canary.py --config configs/e2_canary.yaml --set data.canary.fraction=0.1

Prints one JSON line last: {"fraction": ..., "planted": ..., "p": ..., "bsr": ..., "base": ...}.
Exit code 0 if planted (V2 p <= alpha), 3 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.experiment import prepare, start_run  # noqa: E402
from pof.fl.fedavg import train_federated  # noqa: E402
from pof.verify.backdoor import eval_backdoor  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=1e-3,
                    help="strict on purpose: a backdoor barely detected before unlearning "
                         "cannot show anything after it")
    args = ap.parse_args(argv)
    cfg, _, log, _ = start_run(args.config, args.set, source="scripts/check_canary.py")
    if not cfg["data"]["canary"]["enabled"]:
        sys.exit("the config does not enable canaries")
    env = prepare(cfg, log.info)
    original = train_federated(cfg, env["train"], env["part"]["client_in"], env["clients"],
                               env["k"], cfg["seed"], tag="train")
    v2 = eval_backdoor(cfg, original, env["test"], env["test_triggered"], env["v2_idx"], env["k"],
                       env["canary"]["trigger"]["target"])
    out = {"fraction": cfg["data"]["canary"]["fraction"], "planted": v2["p"] <= args.alpha, **v2}
    log.info(f"canary pilot: {out}")
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["planted"] else 3)
