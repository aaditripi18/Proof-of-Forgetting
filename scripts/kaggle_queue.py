"""Run several Kaggle sessions one after another, unattended.

Each job is `seed:part[+part...]`. For every job the queue waits until the
notebook is idle, pulls the previous session's results into results/, checks
that enough of the weekly GPU quota is left, and starts the next session. It
stops (leaving later jobs for another day) when the quota is too low or a
session ends in an error, and it rides out network outages by retrying.

    python scripts/kaggle_queue.py --init 43:clean+power 43:canary 44:clean+power 44:canary
    while true; do python scripts/kaggle_queue.py --step; [ $? -ge 10 ] && break; sleep 300; done

The queue lives in results/kaggle_logs/queue_state.json and each --step is a
short pass, so the loop can be stopped and restarted at any time (a single
long-lived process did not survive in practice). A session already running on
Kaggle finishes regardless, but the next one starts only when a step sees it
end, so the loop must keep running on this machine. Progress is appended to
results/kaggle_logs/queue.log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_run  # noqa: E402

# generous per-session budgets in GPU-hours, by part (clean ~4.2 with the targeted bank,
# power ~2, canary ~5)
HOURS = {"clean": 4.8, "power": 2.5, "canary": 5.8}
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


STATE = kaggle_run.ROOT / "results" / "kaggle_logs" / "queue_state.json"
DONE, FAILED, NO_QUOTA = 10, 11, 12  # step() exit codes that end the loop; 0 = call again later


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")


def label(job) -> str:
    return f"seed {job[0]} {'+'.join(job[1])}"


def step() -> int:
    """One short pass: if the notebook is idle, pull what just finished and start
    the next job. Everything it needs to know is in the state file."""
    st = load_state()
    if time.time() - st.get("started_at", 0) < 240:
        return 0  # Kaggle may still report the previous session for a moment
    state = session_state()
    if state not in ("complete", "error"):
        return 0
    if st.get("active"):
        say(f"{st['active']} ended: {state}")
        pull()
        st["active"] = None
        save_state(st)
        if state == "error":
            say("stopping: that session ended in an error. Not started: "
                + (", ".join(label(j) for j in st["jobs"]) or "nothing"))
            return FAILED
    if not st["jobs"]:
        say("queue finished: every job ran")
        return DONE
    job = st["jobs"][0]
    need = sum(HOURS[p] for p in job[1])
    left = gpu_hours_left()
    if left is None:
        return 0  # no answer (network?): try again next pass rather than start blind
    if left < need:
        say(f"stopping: {left:.1f} GPU-hours left, {label(job)} needs ~{need:.1f}. Not started: "
            + ", ".join(label(j) for j in st["jobs"]))
        return NO_QUOTA
    say(f"starting {label(job)} ({left:.1f} GPU-hours left, needs ~{need:.1f})")
    kaggle_run.run("full", [job[0]], [], tuple(job[1]))
    st.update(active=label(job), jobs=st["jobs"][1:], started_at=time.time())
    save_state(st)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="*", type=parse_job, help="with --init: the jobs to queue")
    ap.add_argument("--init", action="store_true",
                    help="write a new queue; the session running now (if any) is pulled first")
    ap.add_argument("--step", action="store_true", help="do one pass and exit (run this in a loop)")
    args = ap.parse_args(argv)
    if args.init:
        save_state({"jobs": [[s, list(p)] for s, p in args.jobs],
                    "active": "the session already running", "started_at": 0})
        say("queue: " + ", ".join(label(j) for j in args.jobs))
        return 0
    if args.step:
        return step()
    ap.error("use --init jobs... once, then --step repeatedly")


if __name__ == "__main__":
    sys.exit(main())
