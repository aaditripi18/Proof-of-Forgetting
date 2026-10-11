"""Recompute V5 for an existing E0/E3 run from its saved per-sample outputs.

V5's statistic changed from mean loss to mean logit-scaled confidence (see
pof.evaluate.v5_scores) after the first 65%-accuracy run. Runs made before the
change saved every model's per-sample lsc, so the new V5 can be computed for
them without retraining. Each number is recorded in the ledger under this
script's name and written to results/<experiment>_v5/<run_id>.csv, next to the
original run's CSV rather than over it.

    python scripts/rescore_v5.py --run e0_sanity-20261010-174614-df8519
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.evaluate import v5_scores  # noqa: E402
from pof.experiment import write_csv  # noqa: E402
from pof.ledger import Ledger, read_ledger  # noqa: E402
from pof.seed import derive_seed  # noqa: E402
from pof.stats.hypothesis import permutation_test  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run id of an E0-style run")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    results = Path(args.results)
    npz = np.load(results / "checkpoints" / args.run / "per_sample.npz")
    entries = [e for e in read_ledger(results / "ledger.jsonl") if e["run_id"] == args.run]
    if not entries:
        sys.exit(f"run {args.run} is not in the ledger")
    base = entries[0]
    methods = list(dict.fromkeys(e["method"] for e in entries if "method" in e))
    ledger = Ledger(results / "ledger.jsonl", args.run, base["experiment"] + "_v5",
                    base["config_hash"], base["seed"], source="scripts/rescore_v5.py")
    rows = []
    for m in methods:
        e_in, e_out = {"lsc": npz[f"{m}_in_lsc"]}, {"lsc": npz[f"{m}_out_lsc"]}
        t = permutation_test(v5_scores(e_in), v5_scores(e_out), seed=derive_seed(base["seed"], "v5"))
        # distinct metric names: the run's original v5_p (loss-based) stays in the ledger
        row = {"v5lsc_effect": t["effect"], "v5lsc_p": t["p_value"]}
        for metric, v in row.items():
            ledger.record(metric, v, method=m, rescored_from=args.run)
        rows.append({"run_id": args.run, "method": m, **row})
        print(f"{m:<22} v5lsc_p = {t['p_value']:.4g}")
    out = write_csv(results / (base["experiment"] + "_v5") / f"{args.run}.csv", rows)
    print(json.dumps({"wrote": str(out)}))
    return rows


if __name__ == "__main__":
    main()
