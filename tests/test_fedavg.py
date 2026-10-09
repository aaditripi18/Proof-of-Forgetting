import csv
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pof.config import load_config  # noqa: E402
from pof.data.datasets import load_dataset  # noqa: E402
from pof.data.partition import build_partition  # noqa: E402
from pof.fl.fedavg import fedavg_aggregate, per_sample_eval, train_federated  # noqa: E402
from pof.fl.tensors import TensorData, random_crop_flip  # noqa: E402
from pof.ledger import check_csv_against_ledger  # noqa: E402
from pof.unlearn.registry import available_methods, get_method  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def setup():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    ds = load_dataset(cfg, seed=0)
    part = build_partition(ds["train_y"], cfg, seed=0)
    train = TensorData(ds["train_x"], ds["train_y"], ds["mean"], ds["std"], CPU)
    return cfg, ds, part, train


def test_aggregate_is_weighted_mean():
    a = {"w": torch.tensor([0.0, 0.0]), "n": torch.tensor(7)}
    b = {"w": torch.tensor([3.0, 6.0]), "n": torch.tensor(9)}
    out = fedavg_aggregate([a, b], [1.0, 2.0])
    assert torch.allclose(out["w"], torch.tensor([2.0, 4.0]))
    assert out["n"].item() == 7  # integer buffers copied, not averaged


def test_augmentation_is_seeded_and_shape_preserving():
    x = torch.rand(8, 3, 32, 32)
    a = random_crop_flip(x, torch.Generator().manual_seed(1))
    b = random_crop_flip(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape and torch.equal(a, b)


def test_training_is_bitwise_deterministic(setup):
    cfg, ds, part, train = setup
    clients = list(range(cfg["data"]["num_clients"]))
    s1 = train_federated(cfg, train, part["client_in"], clients, ds["num_classes"], seed=0)
    s2 = train_federated(cfg, train, part["client_in"], clients, ds["num_classes"], seed=0)
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


def test_methods_never_touch_held_out_half(setup):
    """ctx has no client_out key, so any method that tried to read it would raise."""
    cfg, ds, part, train = setup
    clients = list(range(cfg["data"]["num_clients"]))
    original = train_federated(cfg, train, part["client_in"], clients, ds["num_classes"], seed=0)
    ctx = {"cfg": cfg, "seed": 0, "data": train, "client_in": part["client_in"],
           "num_classes": ds["num_classes"], "clients": clients, "forget_client": 0,
           "original_state": original}
    for name in available_methods():
        state = get_method(name)(ctx)
        assert set(state) == set(original)
    same = get_method("do_nothing")(ctx)
    assert all(torch.equal(same[k], original[k]) for k in original)


def test_per_sample_eval_shapes(setup):
    cfg, ds, part, train = setup
    clients = list(range(cfg["data"]["num_clients"]))
    state = train_federated(cfg, train, part["client_in"], clients, ds["num_classes"], seed=0)
    idx = part["client_in"][0]
    e = per_sample_eval(cfg, state, train, idx, ds["num_classes"], batch_size=17)
    assert e["loss"].shape == e["margin"].shape == e["correct"].shape == (len(idx),)
    assert (e["loss"] >= 0).all()
    # margin > 0 exactly when the prediction is correct
    assert np.array_equal(e["margin"] > 0, e["correct"].astype(bool))


def test_e0_end_to_end_and_every_number_is_in_the_ledger(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e0

    out = tmp_path / "out"
    run_e0.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--set",
                 f"output.dir={out.as_posix()}", f"output.ledger={(out / 'ledger.jsonl').as_posix()}"])
    csvs = list((out / "smoke").glob("*.csv"))
    assert len(csvs) == 1
    with open(csvs[0], encoding="utf-8") as f:
        methods = [r["method"] for r in csv.DictReader(f)]
    assert methods == ["do_nothing", "finetune_retain", "retrain_oracle"]
    assert check_csv_against_ledger(csvs[0], out / "ledger.jsonl") == []

    import summarize
    s = summarize.summarize(out, "smoke", out / "ledger.jsonl")
    assert s["runs"] == 1 and s["problems"] == []
    assert set(s["values"]) == {"do_nothing", "finetune_retain", "retrain_oracle"}
