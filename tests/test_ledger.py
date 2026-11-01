import csv

from pof.ledger import Ledger, check_csv_against_ledger, read_ledger


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_csv_matches_ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl", "r1", "e0", "abc", 42, "test")
    led.record("auc", 0.5061, method="do_nothing")
    led.record("auc", 0.4990, method="retrain_oracle")
    _write_csv(tmp_path / "r.csv", [{"run_id": "r1", "method": "do_nothing", "auc": 0.5061},
                                    {"run_id": "r1", "method": "retrain_oracle", "auc": 0.4990}])
    assert check_csv_against_ledger(tmp_path / "r.csv", tmp_path / "ledger.jsonl") == []
    entry = read_ledger(tmp_path / "ledger.jsonl")[0]
    assert {"run_id", "config_hash", "seed", "git_commit", "source"} <= set(entry)
    assert "dataset" not in entry  # older runs carry no dataset; the report reads them as CIFAR-10


def test_dataset_is_recorded_and_labels_the_report(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import report

    (tmp_path / "e5_lira").mkdir()
    for run, ds in (("r1", None), ("r2", "pathmnist")):
        led = Ledger(tmp_path / "ledger.jsonl", run, "e5_lira", "abc", 42, "test", dataset=ds)
        led.record("auc", 0.6, method="lira_sample")
        _write_csv(tmp_path / "e5_lira" / f"{run}.csv",
                   [{"run_id": run, "method": "lira_sample", "auc": 0.6}])
    assert read_ledger(tmp_path / "ledger.jsonl")[1]["dataset"] == "pathmnist"
    text = report.build(tmp_path)
    assert "Seeds present: CIFAR-10 42; PathMNIST 42" in text
    assert "**Seed 42**" in text and "**Seed 42 (PathMNIST)**" in text


def test_rounds_are_recorded_and_label_longer_runs(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import report

    (tmp_path / "e5_lira").mkdir()
    for run, rounds in (("r1", None), ("r2", 30), ("r3", 70)):
        led = Ledger(tmp_path / "ledger.jsonl", run, "e5_lira", "abc", 42, "test", rounds=rounds)
        led.record("auc", 0.6, method="lira_sample")
        _write_csv(tmp_path / "e5_lira" / f"{run}.csv",
                   [{"run_id": run, "method": "lira_sample", "auc": 0.6}])
    assert "rounds" not in read_ledger(tmp_path / "ledger.jsonl")[0]
    runs = report.load(tmp_path, "e5_lira", {e["run_id"]: e for e in read_ledger(tmp_path / "ledger.jsonl")})
    assert [r["rounds"] for r in runs] == [30, 30, 70]        # unrecorded means the default 30
    assert [r["label"] for r in runs] == ["42", "42", "42 (70 rounds)"]


def test_tampered_or_unlogged_numbers_are_caught(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl", "r1", "e0", "abc", 42, "test")
    led.record("auc", 0.5061, method="do_nothing")
    _write_csv(tmp_path / "r.csv", [{"run_id": "r1", "method": "do_nothing", "auc": 0.51},
                                    {"run_id": "r1", "method": "other_method", "auc": 0.50}])
    problems = check_csv_against_ledger(tmp_path / "r.csv", tmp_path / "ledger.jsonl")
    assert len(problems) == 2
    assert "ledger says" in problems[0] and "no ledger entry" in problems[1]
