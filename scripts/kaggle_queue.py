"""Run several Kaggle sessions one after another, unattended.

Each job is `seed:part[+part...]`. For every job the queue waits until the
notebook is idle, pulls the previous session's results into results/, checks
that enough of the weekly GPU quota is left, and starts the next session. It
stops (leaving later jobs for another day) when the quota is too low or a
session ends in an error, and it rides out network outages by retrying.

    python scripts/kaggle_queue.py 43:clean+power 43:canary 44:clean+power 44:canary

This process must stay alive on this machine: a session already running on
Kaggle finishes regardless, but the next one starts only when the queue sees it
end. Progress is appended to results/kaggle_logs/queue.log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_run  # noqa: E402

# generous per-session budgets in GPU-hours, by part (measured: clean 3.5, power 2, canary ~4.5)
HOURS = {"clean": 4.0, "power": 2.5, "canary": 5.5}
POLL_SECONDS = 300
LOG = kaggle_run.ROOT / "results" / "kaggle_logs" / "queue.log"


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def session_state() -> str:
    """'running', 'complete', 'error', or 'unknown' (e.g. no network)."""
    out = kaggle_run.kaggle("kernels", "status", kaggle_run.kernel_id(), check=False).stdout.upper()
    if "RUNNING" in out or "QUEUED" in out:
        return "running"
    if "COMPLETE" in out:
        return "complete"
    if "ERROR" in out or "CANCEL" in out:
        return "error"
    return "unknown"


def wait_until_idle() -> str:
    while True:
        state = session_state()
        if state in ("complete", "error"):
            return state
        time.sleep(POLL_SECONDS)


def gpu_hours_left() -> float | None:
    r = kaggle_run.kaggle("quota", "--format", "json", check=False)
    try:
        gpu = next(x for x in json.loads(r.stdout) if x["resource"] == "GPU")
        return float(gpu["remaining"].rstrip("h"))
    except (json.JSONDecodeError, StopIteration, KeyError, ValueError):
        return None


def pull() -> None:
    try:
        kaggle_run.pull()
    except SystemExit as e:  # nothing to pull, or a network failure: report, keep going
        say(f"pull: {e}")


def parse_job(text: str) -> tuple[int, tuple[str, ...]]:
    seed, _, parts = text.partition(":")
    parts = tuple(parts.split("+"))
    if not parts or any(p not in HOURS for p in parts):
        raise argparse.ArgumentTypeError(f"bad job '{text}': use seed:part+part with parts in {list(HOURS)}")
    return int(seed), parts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="+", type=parse_job)
    args = ap.parse_args(argv)

    say(f"queue: {len(args.jobs)} job(s): " + ", ".join(f"seed {s} {'+'.join(p)}" for s, p in args.jobs))
    say(f"waiting for the current session (state: {session_state()})")
    state = wait_until_idle()
    say(f"current session ended: {state}")
    pull()
    for i, (seed, parts) in enumerate(args.jobs):
        if state == "error":
            say(f"stopping before seed {seed} {'+'.join(parts)}: the previous session ended in an error")
            break
        need = sum(HOURS[p] for p in parts)
        left = None
        while left is None:  # retry through outages rather than start blind
            left = gpu_hours_left()
            if left is None:
                time.sleep(POLL_SECONDS)
        if left < need:
            rest = ", ".join(f"{s}:{'+'.join(p)}" for s, p in args.jobs[i:])
            say(f"stopping: {left:.1f} GPU-hours left, seed {seed} {'+'.join(parts)} needs ~{need:.1f}. "
                f"Not started: {rest}")
            break
        say(f"starting seed {seed} {'+'.join(parts)} ({left:.1f} GPU-hours left, needs ~{need:.1f})")
        kaggle_run.run("full", [seed], [], parts)
        time.sleep(120)  # let Kaggle register the new session before polling it
        state = wait_until_idle()
        say(f"seed {seed} {'+'.join(parts)} ended: {state}")
        pull()
    else:
        say("queue finished: every job ran")


if __name__ == "__main__":
    main()
