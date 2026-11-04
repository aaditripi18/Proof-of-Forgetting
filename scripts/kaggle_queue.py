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
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_run  # noqa: E402

# generous per-session budgets in GPU-hours, by part (clean ~4.2 with the targeted bank,
# power ~2, canary ~5). A part named tune_* is a tune-mode session over configs/<part>.yaml.
HOURS = {"clean": 4.8, "power": 2.5, "canary": 5.8, "canary_bank": 9.5,
         "tune_strong": 2.5, "tune_vgg": 3.0}
POLL_SECONDS = 300
LOG = kaggle_run.ROOT / "results" / "kaggle_logs" / "queue.log"


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def session_state() -> str:
    """'running', 'complete', 'error', 'absent' (no notebook yet: a fresh account),
    or 'unknown' (e.g. no network)."""
    r = kaggle_run.kaggle("kernels", "status", kaggle_run.kernel_id(), check=False)
    out = r.stdout + r.stderr
    if "Cannot access kernel" in out:
        return "absent"
    # read only the status token: a network failure prints e.g. "NameResolutionError",
    # which must not count as a failed session
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    if not m:
        return "unknown"
    status = m.group(1).upper()
    if status in ("RUNNING", "QUEUED"):
        return "running"
    if status == "COMPLETE":
        return "complete"
    if status == "ERROR" or status.startswith("CANCEL"):
        return "error"
    return "unknown"


def gpu_hours_left() -> float | None:
    r = kaggle_run.kaggle("quota", "--format", "json", check=False)
    try:
        # the CLI may print a notice (e.g. "outdated kaggle version") before the JSON
        text = r.stdout[r.stdout.find("["):]
        gpu = next(x for x in json.loads(text) if x["resource"] == "GPU")
        return float(gpu["remaining"].rstrip("h"))
    except (json.JSONDecodeError, StopIteration, KeyError, ValueError):
        return None


PULL_TRIES = 6  # passes (5 min apart) before giving up on a session's download


def pull() -> bool:
    try:
        kaggle_run.pull()
        return True
    except SystemExit as e:  # nothing to pull, or a network failure
        say(f"pull: {e}")
        return False


def parse_job(text: str) -> tuple:
    """`seed:part+part`, optionally `:key=value,key=value` config overrides
    (e.g. `42:clean:unlearn.forget_client=5` forgets another client)."""
    seed, _, rest = text.partition(":")
    parts, _, extra = rest.partition(":")
    parts = tuple(parts.split("+"))
    if not parts or any(p not in HOURS for p in parts):
        raise argparse.ArgumentTypeError(f"bad job '{text}': use seed:part+part[:key=value,...] "
                                         f"with parts in {list(HOURS)}")
    extra = tuple(e for e in extra.split(",") if e)
    if any("=" not in e for e in extra):
        raise argparse.ArgumentTypeError(f"bad override in '{text}': use key=value")
    return (int(seed), parts, extra) if extra else (int(seed), parts)


STATE = kaggle_run.ROOT / "results" / "kaggle_logs" / "queue_state.json"
DONE, FAILED, NO_QUOTA = 10, 11, 12  # step() exit codes that end the loop; 0 = call again later


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")


def label(job) -> str:
    extra = job[2] if len(job) > 2 else ()
    return f"seed {job[0]} {'+'.join(job[1])}" + (f" [{', '.join(extra)}]" if extra else "")


def archive_superseded(spec: dict | None) -> None:
    """Move the just-pulled CSVs of the listed experiments into an archive folder.
    Used once when a session that was already running predates a design change:
    its files must not sit beside those of the runs that replace them."""
    if not spec:
        return
    results = kaggle_run.ROOT / "results"
    for exp in spec["experiments"]:
        dest = results / "archive" / spec["dir"] / exp
        for f in sorted((results / exp).glob("*.csv")):
            dest.mkdir(parents=True, exist_ok=True)
            f.replace(dest / f.name)
            say(f"archived {exp}/{f.name} -> archive/{spec['dir']}")


def step() -> int:
    """One short pass: if the notebook is idle, pull what just finished and start
    the next job. Everything it needs to know is in the state file."""
    st = load_state()
    if time.time() - st.get("started_at", 0) < 240:
        return 0  # Kaggle may still report the previous session for a moment
    state = session_state()
    if state == "absent":  # nothing has ever run on this account: nothing to pull
        st["active"] = None
    elif state not in ("complete", "error"):
        return 0
    if st.get("active"):
        say(f"{st['active']} ended: {state}")
        if not pull() and state == "complete":
            # a complete session always has results: retry rather than start the next
            # session, which would make this one's output no longer the latest
            st["pull_failures"] = st.get("pull_failures", 0) + 1
            if st["pull_failures"] < PULL_TRIES:
                save_state(st)
                return 0
            say(f"giving up on pulling {st['active']}; fetch it later with "
                "`kaggle_run.py pull --version <N>`")
        st.pop("pull_failures", None)
        archive_superseded(st.pop("archive_after_pull", None))
        ended = st.pop("active_job", None)
        if state == "complete" and ended and "canary_bank" in ended[1]:
            # the next canary session reuses this bank: upload it to the account first
            try:
                kaggle_run.push_banks()
                say("uploaded the shadow banks for the next session")
            except SystemExit as e:
                st["active"] = None
                save_state(st)
                say(f"stopping: could not upload the shadow banks ({e}). Not started: "
                    + ", ".join(label(j) for j in st["jobs"]))
                return FAILED
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
    extra = list(job[2]) if len(job) > 2 else []
    if job[1][0].startswith("tune"):  # a tuning session: one variants file, nothing else
        kaggle_run.run("tune", [job[0]], extra, tune_config=f"configs/{job[1][0]}.yaml")
    else:
        kaggle_run.run("full", [job[0]], extra, tuple(job[1]))
    st.update(active=label(job), active_job=job, jobs=st["jobs"][1:], started_at=time.time())
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
        save_state({"jobs": [[j[0], list(j[1]), *([list(j[2])] if len(j) > 2 else [])]
                             for j in args.jobs],
                    "active": "the session already running", "started_at": 0})
        say("queue: " + ", ".join(label(j) for j in args.jobs))
        return 0
    if args.step:
        return step()
    ap.error("use --init jobs... once, then --step repeatedly")


if __name__ == "__main__":
    sys.exit(main())
