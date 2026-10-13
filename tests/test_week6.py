"""Week 6: the server-held aux split, the dishonest canary_suppressor, and E3."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from pof.unlearn.suppressor import anomaly_index

ROOT = Path(__file__).resolve().parents[1]


def test_anomaly_index_flags_the_small_mask():
    ai = anomaly_index(np.array([100.0, 98.0, 103.0, 101.0, 12.0, 99.0]))
    assert np.argmax(ai) == 4 and ai[4] > 2       # Neural Cleanse's threshold
    assert np.all(ai[[0, 1, 2, 3, 5]] < 2)
    assert np.all(anomaly_index(np.ones(4)) == 0)  # no spread, no anomaly


def test_held_out_splits_are_disjoint_and_aux_is_optional():
    from pof.data.datasets import _split_held_out

    x = np.arange(100)[:, None]
    y = np.arange(100)
    xv, yv, xa, ya, xt, yt = _split_held_out(x, y, val_size=30, aux_size=10, seed=0)
    assert (len(yv), len(ya), len(yt)) == (30, 10, 60)
    assert len(set(yv) | set(ya) | set(yt)) == 100
    # without aux, val is unchanged: aux is carved from what would have been test
    xv0, yv0, _, ya0, _, yt0 = _split_held_out(x, y, val_size=30, aux_size=0, seed=0)
    assert np.array_equal(yv0, yv) and len(ya0) == 0 and len(yt0) == 70


def test_write_csv_unions_columns(tmp_path):
    from pof.experiment import write_csv

    p = write_csv(tmp_path / "r.csv", [{"run_id": "r", "method": "a", "x": 1},
                                       {"run_id": "r", "method": "b", "x": 2, "nc_target": 3}])
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert rows[0]["nc_target"] == "" and rows[1]["nc_target"] == "3"


torch = pytest.importorskip("torch")

from pof.config import load_config, parse_overrides  # noqa: E402
from pof.experiment import prepare  # noqa: E402
from pof.fl.fedavg import train_federated  # noqa: E402
from pof.unlearn.registry import get_method  # noqa: E402
from pof.verify.backdoor import eval_backdoor  # noqa: E402


def test_suppressor_never_reads_client_data():
    """The dishonest server holds only the model and its own data: wiping every
    client's training images must not change what it returns."""
    cfg = load_config(ROOT / "configs" / "smoke_e3.yaml")
    env = prepare(cfg)
    orig = train_federated(cfg, env["train"], env["part"]["client_in"], env["clients"], env["k"], 0)
    ctx = {"cfg": cfg, "seed": 0, "aux": env["aux"], "num_classes": env["k"], "original_state": orig}
    a = get_method("canary_suppressor")(dict(ctx))
    env["train"].x.zero_()
    b = get_method("canary_suppressor")({**ctx, "data": env["train"]})
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.slow
def test_suppressor_finds_the_secret_class_and_weakens_the_backdoor():
    """On a model with a genuinely planted backdoor (low-noise synthetic data,
    ResNet-20), Neural Cleanse must point at the client's secret target class
    without being told it, and erasure must lower the trigger's success rate.

    A toy backdoor only plants with enough training (two local epochs, not the
    smoke default of one), and the test is about the method, so it runs on the
    CPU with exactly the setup verified by hand."""
    cfg = load_config(ROOT / "configs" / "smoke_e3.yaml", parse_overrides([
        "data.synthetic_noise=10", "data.canary.fraction=0.2", "fl.rounds=6", "fl.local_epochs=2",
        "fl.lr=0.02", "model.name=resnet20", "unlearn.suppressor.steps=300", "fl.device=cpu"]))
    env = prepare(cfg)
    orig = train_federated(cfg, env["train"], env["part"]["client_in"], env["clients"], env["k"], 0)
    target = env["canary"]["trigger"]["target"]
    v2 = lambda s: eval_backdoor(cfg, s, env["test"], env["test_triggered"], env["v2_idx"],
                                 env["k"], target)
    before = v2(orig)
    assert before["p"] < 1e-6, "backdoor was not planted; the test setup is broken"
    ctx = {"cfg": cfg, "seed": 0, "aux": env["aux"], "num_classes": env["k"], "original_state": orig}
    after = v2(get_method("canary_suppressor")(ctx))
    assert ctx["diagnostics"]["canary_suppressor"]["nc_target"] == target
    assert after["bsr"] < before["bsr"]


def _out(tmp):
    return [f"output.dir={tmp.as_posix()}", f"output.ledger={(tmp / 'ledger.jsonl').as_posix()}"]


def test_e3_end_to_end(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e0
    import run_shadows

    from pof.ledger import check_csv_against_ledger

    run_shadows.main(["--config", str(ROOT / "configs" / "smoke_e3.yaml"), "--set", *_out(tmp_path)])
    bank = next((tmp_path / "shadows").iterdir())
    run_e0.main(["--config", str(ROOT / "configs" / "smoke_e3.yaml"), "--shadows", str(bank),
                 "--set", *_out(tmp_path)])
    path = next((tmp_path / "smoke_e3").glob("*.csv"))
    assert check_csv_against_ledger(path, tmp_path / "ledger.jsonl") == []
    rows = {r["method"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    assert set(rows) == {"do_nothing", "canary_suppressor", "retrain_oracle", "retrain_oracle_alt"}
    s = rows["canary_suppressor"]
    assert s["nc_target_correct"] in ("0.0", "1.0") and s["v2_p"] and s["v3_p"] and s["v5_p"]
    assert rows["do_nothing"]["nc_target"] == ""


def test_summarize_skips_empty_cells(tmp_path):
    """The first full Kaggle run crashed here: E3 rows only partly carry diagnostics."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import summarize

    from pof.experiment import write_csv
    from pof.ledger import Ledger

    led = Ledger(tmp_path / "ledger.jsonl", "r", "e3", "h", 0, "test")
    rows = [{"run_id": "r", "method": "do_nothing", "v2_p": 0.001},
            {"run_id": "r", "method": "canary_suppressor", "v2_p": 0.4, "nc_target": 3.0}]
    for r in rows:
        for k, v in r.items():
            if k not in ("run_id", "method"):
                led.record(k, v, method=r["method"])
    write_csv(tmp_path / "e3" / "r.csv", rows)
    out = summarize.summarize(tmp_path, "e3", tmp_path / "ledger.jsonl")
    assert out["problems"] == [] and out["runs"] == 1
    assert "nc_target" not in out["values"]["do_nothing"]
    assert out["values"]["canary_suppressor"]["nc_target"] == [3.0]


def test_report_is_built_only_from_verified_runs(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import report
    import run_e0

    run_e0.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--set", *_out(tmp_path),
                 "experiment=e0_sanity"])
    text = report.build(tmp_path)
    assert "## E0" in text and "| do_nothing |" in text and "retrain_oracle_alt" in text
    assert "e0_sanity-" in text and "Seeds present: 42" in text      # provenance from the ledger
    assert "## E3" not in text                                        # no such runs: no section

    # a CSV whose numbers are not in the ledger is left out, not reported
    path = next((tmp_path / "e0_sanity").glob("*.csv"))
    path.write_text(path.read_text().replace("do_nothing,", "do_nothing_edited,", 1))
    assert "## E0" not in report.build(tmp_path)
