"""The ledger: every reported number, with the run that produced it.

One JSON object per line. A result CSV is only allowed to contain numbers that
also appear here (`check_csv_against_ledger`), which is enforced by a test, so a
figure can always be traced back to (run_id, config hash, seed, git commit).
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import time
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit(root: Path = REPO_ROOT) -> str:
    """Commit the code ran from, suffixed '+dirty' if tracked files were modified.
    Untracked files (result CSVs, logs) do not count: they are outputs, not code.

    Outside a git checkout (the Kaggle bundle from scripts/kaggle_run.py) the
    commit is read from the COMMIT file the bundle carries.
    """
    try:
        run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True,
                                        check=True, cwd=root).stdout.strip()
        sha = run("rev-parse", "--short", "HEAD")
        dirty = run("status", "--porcelain", "--untracked-files=no")
        return sha + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        stamp = root / "COMMIT"
        return stamp.read_text(encoding="utf-8").strip() if stamp.exists() else "uncommitted"


def new_run_id(experiment: str) -> str:
    return f"{experiment}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class Ledger:
    def __init__(self, path: str | Path, run_id: str, experiment: str, cfg_hash: str,
                 seed: int, source: str, dataset: str | None = None,
                 forget_size: int | None = None, rounds: int | None = None,
                 model: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base = {"run_id": run_id, "experiment": experiment, "config_hash": cfg_hash,
                     "seed": seed, "git_commit": git_commit(), "source": source}
        if dataset is not None:  # entries written before this key existed are CIFAR-10
            self.base["dataset"] = dataset
        if forget_size is not None:  # only runs that shrank the forget client carry it
            self.base["forget_size"] = forget_size
        if rounds is not None:  # entries written before this key existed trained 30 rounds
            self.base["rounds"] = rounds
        if model is not None:  # entries written before this key existed are ResNet-20
            self.base["model"] = model

    def record(self, metric: str, value: float, **keys) -> dict:
        """Append one number. `keys` identify it within the run (method, verifier, ...)."""
        entry = {**self.base, **keys, "metric": metric, "value": float(value),
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry


def read_ledger(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_csv_against_ledger(csv_path: str | Path, ledger_path: str | Path,
                             key_cols: tuple[str, ...] = ("run_id", "method"),
                             rel_tol: float = 1e-9) -> list[str]:
    """Return a list of problems; empty means every numeric cell is in the ledger.

    Each CSV row is identified by `key_cols`; every other column is a metric
    whose value must match the ledger entry with the same keys and metric name.
    """
    index = {}
    for e in read_ledger(ledger_path):
        index[tuple(str(e.get(k)) for k in key_cols) + (e["metric"],)] = e["value"]

    problems = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            keys = tuple(str(row[k]) for k in key_cols)
            for col, cell in row.items():
                if col in key_cols:
                    continue
                try:
                    value = float(cell)
                except (TypeError, ValueError):
                    continue  # non-numeric annotation column
                logged = index.get(keys + (col,))
                if logged is None:
                    problems.append(f"line {i}: {keys} {col}={cell} has no ledger entry")
                elif not math.isclose(logged, value, rel_tol=rel_tol, abs_tol=1e-12):
                    problems.append(f"line {i}: {keys} {col}={cell} but ledger says {logged}")
    return problems
