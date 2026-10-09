import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from pof.verify.oracle_distance import agreement, js_divergence

ROOT = Path(__file__).resolve().parents[1]


def test_js_identity_symmetry_and_bound():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(10), size=50)
    q = rng.dirichlet(np.ones(10), size=50)
    assert np.allclose(js_divergence(p, p), 0.0, atol=1e-12)
    assert np.allclose(js_divergence(p, q), js_divergence(q, p))
    disjoint = js_divergence(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert disjoint[0] == pytest.approx(math.log(2), rel=1e-6)
    assert (js_divergence(p, q) <= math.log(2)).all()


def test_agreement():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    q = np.array([[0.7, 0.3], [0.9, 0.1], [0.51, 0.49]])
    assert agreement(p, q) == pytest.approx(2 / 3)


def test_weight_distance():
    torch = pytest.importorskip("torch")
    from pof.verify.oracle_distance import weight_distance

    a = {"w": torch.tensor([0.0, 0.0]), "steps": torch.tensor(3)}
    b = {"w": torch.tensor([3.0, 4.0]), "steps": torch.tensor(9)}
    assert weight_distance(a, b) == pytest.approx(5.0)  # integer buffers ignored
    assert weight_distance(a, a) == 0.0


def test_shared_init_tag_gives_identical_init():
    torch = pytest.importorskip("torch")
    from pof.config import load_config
    from pof.data.datasets import load_dataset
    from pof.data.partition import build_partition
    from pof.fl.fedavg import train_federated
    from pof.fl.tensors import TensorData

    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    ds = load_dataset(cfg, seed=0)
    part = build_partition(ds["train_y"], cfg, seed=0)
    data = TensorData(ds["train_x"], ds["train_y"], ds["mean"], ds["std"], torch.device("cpu"))
    args = (cfg, data, part["client_in"], [1, 2, 3], ds["num_classes"], 0)
    a = train_federated(*args, tag="train", rounds=0)
    b = train_federated(*args, tag="train_alt", init_tag="train", rounds=0)
    c = train_federated(*args, tag="train_alt", rounds=0)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_v4_in_e0_rows(tmp_path):
    """do_nothing is exactly as far from the oracle as the original (w_rel = 1),
    the oracle is at distance 0 from itself, and a second retrain is not."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e0

    out = tmp_path / "out"
    run_e0.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--set",
                 f"output.dir={out.as_posix()}", f"output.ledger={(out / 'ledger.jsonl').as_posix()}"])
    with open(next((out / "smoke").glob("*.csv")), encoding="utf-8") as f:
        rows = {r["method"]: {k: v for k, v in r.items()} for r in csv.DictReader(f)}
    assert float(rows["do_nothing"]["v4_w_rel"]) == pytest.approx(1.0)
    assert float(rows["retrain_oracle"]["v4_w_dist"]) == 0.0
    assert float(rows["retrain_oracle"]["v4_agree_test"]) == 1.0
    assert float(rows["retrain_oracle"]["v4_js_forget"]) == pytest.approx(0.0, abs=1e-12)
    assert float(rows["retrain_oracle_alt"]["v4_w_dist"]) > 0.0
