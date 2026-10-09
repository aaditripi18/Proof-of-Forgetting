import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("torch")

import run_e4  # noqa: E402

from pof.ledger import check_csv_against_ledger  # noqa: E402


def test_shrink_client_only_touches_the_forget_client():
    part = {"client_in": [np.arange(0, 50), np.arange(50, 100)],
            "client_out": [np.arange(100, 150), np.arange(150, 200)]}
    p = run_e4.shrink_client(part, 0, 10, seed=0)
    assert len(p["client_in"][0]) == len(p["client_out"][0]) == 10
    assert np.isin(p["client_in"][0], part["client_in"][0]).all()
    assert np.isin(p["client_out"][0], part["client_out"][0]).all()
    assert np.array_equal(p["client_in"][1], part["client_in"][1])
    assert len(part["client_in"][0]) == 50  # original untouched
    assert np.array_equal(p["client_in"][0], run_e4.shrink_client(part, 0, 10, seed=0)["client_in"][0])
    with pytest.raises(ValueError):
        run_e4.shrink_client(part, 0, 51, seed=0)


def test_e4_end_to_end(tmp_path):
    out = [f"output.dir={tmp_path.as_posix()}", f"output.ledger={(tmp_path / 'ledger.jsonl').as_posix()}"]
    rows = run_e4.main(["--config", str(ROOT / "configs" / "smoke_e4.yaml"), "--set", *out])
    assert [r["method"] for r in rows[:3]] == ["do_nothing@n=10", "finetune_retain@n=10",
                                              "retrain_oracle@n=10"]
    for r in rows:
        ms = [int(k[7:]) for k in r if k.startswith("power_m")]
        assert all(2 * m <= r["n"] for m in ms)  # never the degenerate m = n
        assert r["reject"] == float(r["p"] <= 0.05)
    path = next((tmp_path / "smoke_e4").glob("*.csv"))
    assert check_csv_against_ledger(path, tmp_path / "ledger.jsonl") == []
    oracle_rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
                   if r["method"].startswith("retrain_oracle")]
    assert len(oracle_rows) == 3
