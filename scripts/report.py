"""Build docs/RESULTS.md from the result CSVs, so no reported number is typed by hand.

Only CSVs that verify against the ledger are used; each table names the runs it
was built from (run id, seed, config hash, code commit, all read from the
ledger). Archived runs under results/archive are never included.

    python scripts/report.py                 # writes docs/RESULTS.md
    python scripts/report.py --stdout
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.ledger import check_csv_against_ledger, read_ledger  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05


def num(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else None


def fmt(v: float | None, kind: str = "f3") -> str:
    if v is None:
        return "–"
    if kind == "p":  # p-values: enough digits to compare with alpha, never "0"
        return "<0.0001" if v < 1e-4 else f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
    return {"f3": f"{v:.3f}", "f2": f"{v:.2f}", "pct": f"{100 * v:.1f}%", "int": f"{v:.0f}"}[kind]


def load(results: Path, experiment: str, ledger_index: dict) -> list[dict]:
    """Verified runs of one experiment: [{run_id, seed, config_hash, commit, rows}], by seed."""
    runs = []
    for path in sorted((results / experiment).glob("*.csv")):
        if check_csv_against_ledger(path, results / "ledger.jsonl"):
            print(f"skipping {path.name}: does not match the ledger", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        meta = ledger_index[rows[0]["run_id"]]
        runs.append({"run_id": rows[0]["run_id"], "seed": meta["seed"],
                     "config_hash": meta["config_hash"], "commit": meta["git_commit"], "rows": rows})
    return sorted(runs, key=lambda r: (r["seed"], r["run_id"]))


def table(header: list[str], body: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in body])


def provenance(runs: list[dict]) -> str:
    return "Runs: " + "; ".join(f"seed {r['seed']} `{r['run_id']}` (config `{r['config_hash']}`, "
                                f"code `{r['commit']}`)" for r in runs)


def section_e0(runs: list[dict], title: str, canary: bool) -> str:
    cols = [("test_acc", "Test acc.", "pct"), ("v1_matched_auc", "V1 AUC", "f3")]
    if canary:
        cols += [("v2_bsr", "V2 trigger success", "pct"), ("v2_p", "V2 p", "p")]
    cols += [("v3_p", "V3 p", "p"), ("v5_p", "V5 p", "p")]
    if not canary:
        cols += [("v4_js_forget", "V4 JS to oracle", "f3")]
    parts = [f"## {title}", "", provenance(runs), ""]
    for run in runs:
        body = [[r["method"]] + [fmt(num(r, k), kind) for k, _, kind in cols] for r in run["rows"]]
        parts += [f"**Seed {run['seed']}**", "", table(["Method"] + [c[1] for c in cols], body), ""]
        nc = [r for r in run["rows"] if r.get("nc_target", "") != ""]
        if nc:
            r = nc[0]
            parts += [f"Neural Cleanse picked class {fmt(num(r, 'nc_target'), 'int')} "
                      f"({'the' if num(r, 'nc_target_correct') == 1 else 'not the'} client's secret class), "
                      f"anomaly index {fmt(num(r, 'nc_anomaly'), 'f2')}.", ""]
    return "\n".join(parts)


def section_e5(runs: list[dict]) -> str:
    parts = ["## E5: attack strength on the forget client", "", provenance(runs), ""]
    for run in runs:
        body = [[r["method"], fmt(num(r, "auc"), "f3"), fmt(num(r, "tpr@0.01"), "pct"),
                 fmt(num(r, "tpr@0.001"), "pct"), fmt(num(r, "n_pos"), "int"), fmt(num(r, "n_neg"), "int")]
                for r in run["rows"]]
        parts += [f"**Seed {run['seed']}**", "",
                  table(["Attack, level", "AUC", "TPR @ 1% FPR", "TPR @ 0.1% FPR", "Members", "Non-members"],
                        body), ""]
    return "\n".join(parts)


def section_e2(runs: list[dict]) -> str:
    parts = ["## E2: which signal survives gradient ascent longest", "", provenance(runs), "",
             "First checkpoint at which each verifier stops detecting the client (p > 0.05), "
             "and the model's test accuracy there.", ""]
    body = []
    for run in runs:
        traj = sorted((r for r in run["rows"] if num(r, "step") is not None and num(r, "step") >= 0),
                      key=lambda r: num(r, "step"))
        start = traj[0]
        for key, name in (("v5_p", "V5 (in/out test)"), ("v3_p", "V3 (LiRA)"), ("v2_p", "V2 (canary)")):
            if num(start, key) is None:
                continue
            gone = next((r for r in traj if num(r, key) > ALPHA), None)
            body.append([str(run["seed"]), name, fmt(num(start, key), "p"),
                         fmt(num(gone, "step"), "int") if gone else "never",
                         fmt(num(gone, "test_acc"), "pct") if gone else "–"])
        body.append([str(run["seed"]), "(start)", "", "0", fmt(num(start, "test_acc"), "pct")])
    parts += [table(["Seed", "Verifier", "p before ascent", "Lost at step", "Test acc. at that step"], body), ""]
    return "\n".join(parts)


def section_e4(runs: list[dict]) -> str:
    parts = ["## E4: V5 against the client's data size", "", provenance(runs), "",
             "V5 p-value and effect size d when the client holds n samples per half. "
             "`do_nothing` should be detected; `retrain_oracle` should not.", ""]
    body = []
    for run in runs:
        by = {r["method"]: r for r in run["rows"]}
        for n in sorted({int(float(r["n"])) for r in run["rows"]}):
            dn, orc = by.get(f"do_nothing@n={n}"), by.get(f"retrain_oracle@n={n}")
            body.append([str(run["seed"]), str(n),
                         fmt(num(dn, "p"), "p") if dn else "–", fmt(num(dn, "d"), "f3") if dn else "–",
                         fmt(num(orc, "p"), "p") if orc else "–", fmt(num(orc, "d"), "f3") if orc else "–"])
    parts += [table(["Seed", "n per half", "do_nothing p", "do_nothing d", "oracle p", "oracle d"], body), ""]
    return "\n".join(parts)


def build(results: Path) -> str:
    index = {}
    for e in read_ledger(results / "ledger.jsonl"):
        index.setdefault(e["run_id"], e)
    exp = {name: load(results, name, index)
           for name in ("e0_sanity", "e5_lira", "e2_canary", "e3_dishonest", "e4_power")}
    seeds = sorted({r["seed"] for runs in exp.values() for r in runs})
    parts = ["# Results", "",
             "Generated by `scripts/report.py` from the CSVs in `results/` that verify against "
             "`results/ledger.jsonl`. Do not edit by hand.", "",
             f"Seeds present: {', '.join(map(str, seeds)) or 'none'}. "
             "V1 = loss-threshold membership AUC against the client's own held-out half; "
             "V2 = backdoor canary; V3 = client-level LiRA against retrains without the client; "
             "V4 = distance to the retrain oracle; V5 = client-side in/out permutation test. "
             "For V2, V3 and V5 a p-value below 0.05 means the client is still detected.", ""]
    if exp["e0_sanity"]:
        parts.append(section_e0(exp["e0_sanity"], "E0: every method against every verifier", canary=False))
    if exp["e5_lira"]:
        parts.append(section_e5(exp["e5_lira"]))
    if exp["e2_canary"]:
        parts.append(section_e2(exp["e2_canary"]))
    if exp["e3_dishonest"]:
        parts.append(section_e0(exp["e3_dishonest"], "E3: a dishonest server against the canary check",
                                canary=True))
    if exp["e4_power"]:
        parts.append(section_e4(exp["e4_power"]))
    return "\n".join(parts).rstrip() + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "RESULTS.md"))
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args(argv)
    text = build(Path(args.results))
    if args.stdout:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
