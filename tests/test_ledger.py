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


def test_tampered_or_unlogged_numbers_are_caught(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl", "r1", "e0", "abc", 42, "test")
    led.record("auc", 0.5061, method="do_nothing")
    _write_csv(tmp_path / "r.csv", [{"run_id": "r1", "method": "do_nothing", "auc": 0.51},
                                    {"run_id": "r1", "method": "other_method", "auc": 0.50}])
    problems = check_csv_against_ledger(tmp_path / "r.csv", tmp_path / "ledger.jsonl")
    assert len(problems) == 2
    assert "ledger says" in problems[0] and "no ledger entry" in problems[1]
