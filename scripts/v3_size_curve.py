"""How V3 (client-level LiRA) power depends on how many of the client's samples are audited.

Across seeds 42/43/44, V3's evidence against do_nothing fell as the forget
client got smaller (z 5.5 / 5.1 / 1.8 for 3360 / 2511 / 1855 training samples),
but three points that also differ in seed and partition cannot separate size
from everything else. This script measures the dependence inside each run: it
draws random subsets of m of the forget client's training samples and reruns
the V3 test (lira.fit on the subset, leave-one-out null, Student-t prediction
test) as if the client owned only those samples. No model is retrained, so it
needs no GPU: everything comes from the run's saved per-sample outputs and its
shadow bank.

What it measures is the audit side: how many of its samples a client needs to
audit. A client that *trained* on fewer samples could be memorised more or less
per sample; this script does not model that.

Each number is recorded in the ledger under this script's name and written to
results/<experiment>_v3size/<run_id>.csv.

    python scripts/v3_size_curve.py --run e0_sanity-20261019-121703-94e9dc
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pof.experiment import write_csv  # noqa: E402
from pof.ledger import Ledger, read_ledger  # noqa: E402
from pof.seed import derive_seed  # noqa: E402
from pof.shadows import load_bank  # noqa: E402
from pof.verify import lira  # noqa: E402

SIZES = (50, 100, 250, 500, 1000, 1500)
ALPHA = 0.05


def v3_on(cols, phi, member, target_lsc, out_rows):
    """The V3 test of evaluate.v3_reference/v3_test, restricted to sample columns `cols`."""
    p, m = phi[:, cols], member[:, cols]
    null = lira.loo_scores(p, m).mean(1)[out_rows]
    return lira.client_test(float(lira.score(target_lsc[cols], lira.fit(p, m)).mean()), null)


def bank_of(results: Path, run: str) -> Path:
    log = (results / "logs" / f"{run}.log").read_text(encoding="utf-8")
    hit = re.search(r"V3: bank (\S+)", log)
    if not hit:
        sys.exit(f"run {run} did not use a shadow bank (no V3)")
    return Path(hit.group(1))


def client_start(results: Path, run: str, f: int) -> int:
    """First pool column of client f: the pool concatenates the clients' training
    halves in client order (shadows.pool_of), and the run log lists their sizes."""
    log = (results / "logs" / f"{run}.log").read_text(encoding="utf-8")
    hit = re.search(r"client sizes in/out: (.+)", log)
    if not hit:
        sys.exit(f"run {run}: no client sizes in its log")
    sizes = [int(pair.split("/")[0]) for pair in hit.group(1).split(", ")]
    return sum(sizes[:f])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run id of an E0 run with V3")
    ap.add_argument("--results", default="results")
    ap.add_argument("--reps", type=int, default=200, help="random subsets per size")
    args = ap.parse_args(argv)
    results = Path(args.results)
    entries = [e for e in read_ledger(results / "ledger.jsonl") if e["run_id"] == args.run]
    if not entries:
        sys.exit(f"run {args.run} is not in the ledger")
    base = entries[0]
    npz = np.load(results / "checkpoints" / args.run / "per_sample.npz")
    bank = load_bank(bank_of(results, args.run))
    f = bank["manifest"]["target"]
    methods = list(dict.fromkeys(e["method"] for e in entries if "method" in e))
    n = len(npz[f"{methods[0]}_in_lsc"])
    start = client_start(results, args.run, f)
    phi = bank["lsc"][:, start:start + n]
    member = np.repeat(bank["membership"][:, [f]], n, axis=1)
    out_rows = ~bank["membership"][:, f]

    # alignment check: the full client must reproduce the run's own V3 z
    recorded = {e["method"]: e["value"] for e in entries if e.get("metric") == "v3_z"}
    for m in methods:
        z = v3_on(np.arange(n), phi, member, npz[f"{m}_in_lsc"], out_rows)["z"]
        if m in recorded and not np.isclose(z, recorded[m], rtol=1e-6, atol=1e-9):
            sys.exit(f"{m}: recomputed V3 z {z:.6f} != recorded {recorded[m]:.6f}; columns misaligned")
    print(f"{args.run}: forget client {f} has {n} samples; full-client V3 reproduces the run")

    ledger = Ledger(results / "ledger.jsonl", args.run, base["experiment"] + "_v3size",
                    base["config_hash"], base["seed"], source="scripts/v3_size_curve.py",
                    dataset=base.get("dataset"), forget_size=base.get("forget_size"),
                    rounds=base.get("rounds"))
    rng = np.random.default_rng(derive_seed(base["seed"], "v3_size_curve"))
    sizes = [s for s in SIZES if s < n]
    subsets = {s: [rng.choice(n, s, replace=False) for _ in range(args.reps)] for s in sizes}
    rows = []
    for m in methods:
        target = npz[f"{m}_in_lsc"]
        # one wide row per method: the ledger check keys rows by (run_id, method)
        row = {"v3sz_n_client": n}
        for s in sizes:
            tests = [v3_on(c, phi, member, target, out_rows) for c in subsets[s]]
            zs = np.array([t["z"] for t in tests])
            ps = np.array([t["p"] for t in tests])
            row.update({f"v3sz_detect_{s}": float((ps < ALPHA).mean()),
                        f"v3sz_zmed_{s}": float(np.median(zs)),
                        f"v3sz_zq10_{s}": float(np.quantile(zs, 0.1)),
                        f"v3sz_zq90_{s}": float(np.quantile(zs, 0.9))})
            print(f"{m:<22} m={s:<5} detect={row[f'v3sz_detect_{s}']:.2f}  "
                  f"z median {row[f'v3sz_zmed_{s}']:.2f} "
                  f"[{row[f'v3sz_zq10_{s}']:.2f}, {row[f'v3sz_zq90_{s}']:.2f}]")
        for metric, v in row.items():
            ledger.record(metric, v, method=m, forget_client=f, reps=args.reps, rescored_from=args.run)
        rows.append({"run_id": args.run, "method": m, **row})
    out = write_csv(results / (base["experiment"] + "_v3size") / f"{args.run}.csv", rows)
    print(json.dumps({"wrote": str(out)}))
    return rows


if __name__ == "__main__":
    main()
