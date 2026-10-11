import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import kaggle_run  # noqa: E402

from pof.ledger import git_commit  # noqa: E402


def test_notebook_is_valid_and_carries_parameters():
    nb = kaggle_run.build_notebook("full", [42, 43], ["data.alpha=0.1"], "abc1234")
    json.dumps(nb)  # serialisable
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for src in code:
        ast.parse(re.sub(r"(?m)^(\s*)!.*$", r"\1pass", src))
    assert "MODE = 'full'" in code[0] and "SEEDS = [42, 43]" in code[0]
    assert "EXPECTED_COMMIT = 'abc1234'" in code[0]
    assert len({c["id"] for c in nb["cells"]}) == len(nb["cells"])


def test_merge_results_never_overwrites_and_dedupes_ledger(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "e0").mkdir(parents=True)
    (dst / "e0").mkdir(parents=True)
    (src / "e0" / "run1.csv").write_text("new")
    (src / "e0" / "run2.csv").write_text("fresh")
    (dst / "e0" / "run1.csv").write_text("old")
    (src / "ledger.jsonl").write_text('{"a": 1}\n{"b": 2}\n')
    (dst / "ledger.jsonl").write_text('{"a": 1}\n')
    counts = kaggle_run.merge_results(src, dst)
    assert counts == {"files": 1, "ledger_lines": 1}
    assert (dst / "e0" / "run1.csv").read_text() == "old"
    assert (dst / "e0" / "run2.csv").read_text() == "fresh"
    assert (dst / "ledger.jsonl").read_text().splitlines() == ['{"a": 1}', '{"b": 2}']
    kaggle_run.merge_results(src, dst)  # idempotent
    assert len((dst / "ledger.jsonl").read_text().splitlines()) == 2


def test_commit_stamp_used_outside_git(tmp_path):
    assert git_commit(tmp_path) == "uncommitted"
    (tmp_path / "COMMIT").write_text("abc1234\n")
    assert git_commit(tmp_path) == "abc1234"


def test_queue_parses_jobs_and_reads_states(monkeypatch):
    import argparse
    from types import SimpleNamespace

    import kaggle_queue
    import pytest

    assert kaggle_queue.parse_job("43:clean+power") == (43, ("clean", "power"))
    with pytest.raises(argparse.ArgumentTypeError):
        kaggle_queue.parse_job("43:everything")

    replies = {}
    monkeypatch.setattr(kaggle_queue.kaggle_run, "kaggle",
                        lambda *a, **k: SimpleNamespace(stdout=replies[a[0]], returncode=0))
    for text, want in [('has status "KernelWorkerStatus.RUNNING"', "running"),
                       ('has status "KernelWorkerStatus.COMPLETE"', "complete"),
                       ('has status "KernelWorkerStatus.ERROR"', "error"),
                       ("", "unknown")]:  # e.g. no network
        replies["kernels"] = text
        assert kaggle_queue.session_state() == want
    replies["quota"] = '[{"resource": "GPU", "remaining": "22.63h"}, {"resource": "TPU", "remaining": "20.00h"}]'
    assert kaggle_queue.gpu_hours_left() == 22.63
    replies["quota"] = "network down"
    assert kaggle_queue.gpu_hours_left() is None


def test_queue_steps_through_jobs_and_stops_on_low_quota(monkeypatch, tmp_path):
    import kaggle_queue as q

    monkeypatch.setattr(q, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(q, "LOG", tmp_path / "queue.log")
    world = {"state": "running", "quota": 20.0}
    started, pulled = [], []
    monkeypatch.setattr(q, "session_state", lambda: world["state"])
    monkeypatch.setattr(q, "gpu_hours_left", lambda: world["quota"])
    monkeypatch.setattr(q, "pull", lambda: pulled.append(1))
    monkeypatch.setattr(q.kaggle_run, "run", lambda mode, seeds, extra, parts: started.append((seeds[0], parts)))
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(q.time, "time", lambda: clock["t"])

    assert q.main(["--init", "43:clean+power", "43:canary", "44:canary"]) == 0
    assert q.step() == 0 and started == []            # seed 42's session still running
    world["state"] = "complete"
    assert q.step() == 0                              # pulls it, starts the first job
    assert pulled == [1] and started == [(43, ("clean", "power"))]
    assert q.step() == 0 and len(started) == 1        # just started: stale status is ignored
    clock["t"] += 600
    world["state"] = "running"
    assert q.step() == 0 and len(started) == 1
    world["state"], world["quota"] = "complete", None
    assert q.step() == 0 and len(pulled) == 2 and len(started) == 1   # no quota answer: wait
    world["quota"] = 6.0
    assert q.step() == 0 and started[-1] == (43, ("canary",))
    clock["t"] += 600
    world["quota"] = 3.0                              # not enough for the last job
    assert q.step() == q.NO_QUOTA and len(started) == 2 and len(pulled) == 3
    assert "44" in (tmp_path / "queue.log").read_text()

    # an errored session stops the queue after pulling what it produced
    q.main(["--init", "44:canary"])
    world["state"] = "error"
    assert q.step() == q.FAILED and len(started) == 2
