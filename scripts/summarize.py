"""Summarise an experiment across runs (seeds): mean ± std per method and metric.

Every CSV is checked against the ledger first; a CSV with a number the ledger
does not contain is reported and left out of the summary.

    python scripts/summarize.py --experiment e0_sanity
    python scripts/summarize.py --experiment e0_sanity --metrics v1_pooled_auc v5_p
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.ledger import check_csv_against_ledger  # noqa: E402

DEFAULT_METRICS = ["test_acc", "v1_pooled_auc", "v1_matched_auc", "v1_matched_tpr@0.01",
                   "v4_w_rel", "v4_js_forget", "v5_effect", "v5_p"]


def summarize(results_dir: Path, experiment: str, ledger: Path) -> dict:
    """{method: {metric: [values across verified runs]}}, plus the list of problems."""
    values: dict = defaultdict(lambda: defaultdict(list))
    problems, runs = [], 0
    for path in sorted((results_dir / experiment).glob("*.csv")):
        bad = check_csv_against_ledger(path, ledger)
        if bad:
            problems += [f"{path.name}: {b}" for b in bad]
            continue
        runs += 1
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    # empty cell: a column only some methods have (e.g. diagnostics)
                    if k not in ("run_id", "method") and v != "":
                        values[row["method"]][k].append(float(v))
    return {"values": values, "runs": runs, "problems": problems}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--ledger", default=None, help="default: <results>/ledger.jsonl")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    args = ap.parse_args(argv)

    results = Path(args.results)
    s = summarize(results, args.experiment, Path(args.ledger or results / "ledger.jsonl"))
    for p in s["problems"]:
        print("LEDGER MISMATCH:", p)
    print(f"{args.experiment}: {s['runs']} verified run(s)\n")
    width = max([len(m) for m in s["values"]] + [6])
    print(f"{'method':<{width}}  " + "  ".join(f"{m:>22}" for m in args.metrics))
    for method, metrics in s["values"].items():
        cells = []
        for m in args.metrics:
            v = np.array(metrics.get(m, []))
            cells.append(f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f}"
                         if len(v) else "—")
        print(f"{method:<{width}}  " + "  ".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
