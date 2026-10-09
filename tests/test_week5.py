"""Week 5: canaries and V2, SCRUB, FedEraser-lite, the ascent trajectory, and E2."""
import copy
import csv
import math
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from pof.data.canary import add_canaries, apply_trigger, make_trigger
from pof.verify.backdoor import backdoor_test, binom_sf_half

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- canaries, V2 (numpy)

def test_trigger_stamps_only_its_patch():
    x = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    trig = make_trigger((3, 8, 8), size=3, num_classes=10, seed=0)
    y = apply_trigger(x, trig)
    s, y0, x0 = 3, trig["y0"], trig["x0"]
    assert np.array_equal(y[:, :, y0:y0 + s, x0:x0 + s], np.broadcast_to(trig["pattern"], (2, 3, 3, 3)))
    y[:, :, y0:y0 + s, x0:x0 + s] = 0
    assert not y.any() and not x.any()  # nothing else changed, input untouched


def test_add_canaries_appends_to_forget_client_only():
    rng = np.random.default_rng(0)
    ds = {"train_x": rng.integers(0, 255, (100, 3, 8, 8), dtype=np.uint8),
          "train_y": rng.integers(0, 4, 100), "num_classes": 4}
    part = {"client_in": [np.arange(0, 40), np.arange(40, 80)],
            "client_out": [np.arange(80, 90), np.arange(90, 100)]}
    before = [p.copy() for p in part["client_in"]]
    cfg = {"data": {"canary": {"fraction": 0.25, "size": 2}}, "unlearn": {"forget_client": 0}}
    can = add_canaries(ds, part, cfg, seed=1)
    t = can["trigger"]["target"]
    assert len(can["idx"]) == 10 and np.array_equal(can["idx"], np.arange(100, 110))
    assert (ds["train_y"][can["idx"]] == t).all()
    assert (ds["train_y"][can["source_idx"]] != t).all()          # sources are non-target
    assert np.isin(can["source_idx"], before[0]).all()             # from the client's own data
    assert np.array_equal(part["client_in"][0], np.concatenate([before[0], can["idx"]]))
    assert np.array_equal(part["client_in"][1], before[1])         # other clients untouched
    assert len(ds["train_x"]) == 110


def test_binomial_tail_exact():
    for n, k in product(range(0, 12), range(0, 13)):
        brute = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if k <= n else 0.0
        assert binom_sf_half(k, n) == pytest.approx(brute if k > 0 else 1.0, abs=1e-12)
    assert 0 < binom_sf_half(2600, 5000) < 0.01  # stable for large n


def test_backdoor_test_directions():
    clean = np.array([1, 2, 3, 1, 2, 3] * 50)
    planted = backdoor_test(pred_triggered=np.zeros_like(clean), pred_clean=clean, target=0)
    assert planted["bsr"] == 1.0 and planted["base"] == 0.0 and planted["p"] < 1e-30
    none = backdoor_test(pred_triggered=clean, pred_clean=clean, target=0)
    assert none["effect"] == 0.0 and none["p"] == 1.0


# ---------------------------------------------------------------- methods (torch)

torch = pytest.importorskip("torch")

from pof.config import load_config  # noqa: E402
from pof.experiment import prepare  # noqa: E402
from pof.fl.fedavg import clone_state, train_federated  # noqa: E402
from pof.models import build_model  # noqa: E402
from pof.unlearn.gradient_ascent import ascend  # noqa: E402
from pof.unlearn.registry import get_method  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    env = prepare(cfg)
    norms = {}
    original = train_federated(cfg, env["train"], env["part"]["client_in"], env["clients"],
                               env["k"], 0, update_norms=norms)
    return {"cfg": cfg, "seed": 0, "data": env["train"], "client_in": env["part"]["client_in"],
            "num_classes": env["k"], "clients": env["clients"], "forget_client": 0,
            "original_state": original, "update_norms": norms}


def _equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_update_norms_recorded_per_round_client_layer(ctx):
    norms = ctx["update_norms"]
    assert sorted(norms) == list(range(ctx["cfg"]["fl"]["rounds"]))
    assert sorted(norms[0]) == ctx["clients"]
    float_keys = {k for k, v in ctx["original_state"].items() if v.is_floating_point()}
    assert set(norms[0][0]) == float_keys
    assert all(v >= 0 for v in norms[0][0].values())


def test_federaser_with_zero_norms_returns_the_initialisation(ctx):
    zero = {t: {c: {k: 0.0 for k in layer} for c, layer in per_c.items()}
            for t, per_c in ctx["update_norms"].items()}
    out = get_method("federaser_lite")({**ctx, "update_norms": zero})
    init = train_federated(ctx["cfg"], ctx["data"], ctx["client_in"], [], ctx["num_classes"],
                           0, tag="train", rounds=0)
    assert _equal(out, init)


def test_federaser_needs_norms(ctx):
    with pytest.raises(ValueError):
        get_method("federaser_lite")({**ctx, "update_norms": None})


def test_methods_are_deterministic(ctx):
    for name in ("scrub", "federaser_lite"):
        assert _equal(get_method(name)(ctx), get_method(name)(ctx))


def test_scrub_without_max_steps_never_reads_forget_data(ctx):
    cfg = copy.deepcopy(ctx["cfg"])
    cfg["unlearn"]["scrub"].update(rounds=1, max_rounds=0)
    a = get_method("scrub")({**ctx, "cfg": cfg})
    data = copy.copy(ctx["data"])
    x = data.x.clone()
    x[torch.from_numpy(ctx["client_in"][0])] = 0  # wipe the forget client's images
    data.x = x
    b = get_method("scrub")({**ctx, "cfg": cfg, "data": data})
    assert _equal(a, b)


def test_ascent_callback_does_not_change_the_trajectory(ctx):
    def run(callback):
        m = build_model(ctx["cfg"], ctx["num_classes"]).to(ctx["data"].device)
        m.load_state_dict(ctx["original_state"])
        info = ascend(m, ctx["data"], ctx["client_in"][0], epochs=2, lr=0.05, batch_size=16,
                      max_loss=float("inf"), radius=None, augment=True, seed=3,
                      callback=callback, every=4 if callback else 0)
        return clone_state(m), info

    seen = []
    with_cb, info = run(lambda step, m: (seen.append(step), build_model(ctx["cfg"], 4)))
    without, _ = run(None)
    assert _equal(with_cb, without)
    assert seen[0] == 0 and seen[-1] == info["steps"]
    assert all(s % 4 == 0 for s in seen[:-1]) and seen == sorted(set(seen))


# ---------------------------------------------------------------- end to end

def _out(tmp):
    return [f"output.dir={tmp.as_posix()}", f"output.ledger={(tmp / 'ledger.jsonl').as_posix()}"]


def test_e2_and_canary_e0_end_to_end(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e0
    import run_e2
    import run_shadows

    from pof.ledger import check_csv_against_ledger

    cfg = str(ROOT / "configs" / "smoke_e2.yaml")
    run_shadows.main(["--config", cfg, "--set", *_out(tmp_path)])
    bank = next((tmp_path / "shadows").iterdir())
    rows = run_e2.main(["--config", cfg, "--shadows", str(bank), "--set", *_out(tmp_path)])
    steps = [r["step"] for r in rows]
    assert steps[0] == 0 and steps[-1] == -1 and steps[:-1] == sorted(steps[:-1])
    assert {"v2_p", "v3_p", "v5_p"} <= set(rows[0])
    e2_csv = next((tmp_path / "smoke_e2").glob("*.csv"))
    assert check_csv_against_ledger(e2_csv, tmp_path / "ledger.jsonl") == []

    # E0 with canaries: every method gets V2 and, against the canary bank, V3
    run_e0.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--shadows", str(bank),
                 "--set", *_out(tmp_path), "data.canary.enabled=true",
                 "lira.num_shadow=8", "lira.num_out=2"])
    e0_csv = next((tmp_path / "smoke").glob("*.csv"))
    assert check_csv_against_ledger(e0_csv, tmp_path / "ledger.jsonl") == []
    with open(e0_csv, encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    assert all(row["v2_p"] != "" and row["v3_p"] != "" for row in r)


def test_without_canaries_the_bank_of_a_canary_setup_is_refused(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e0
    import run_shadows

    run_shadows.main(["--config", str(ROOT / "configs" / "smoke_e2.yaml"), "--set", *_out(tmp_path)])
    bank = next((tmp_path / "shadows").iterdir())
    with pytest.raises(RuntimeError):
        run_e0.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--shadows", str(bank),
                     "--set", *_out(tmp_path)])


def test_cifar_mirror_is_used_when_valid_and_skipped_when_not(monkeypatch):
    import torchvision.datasets as tvd

    from pof.data import datasets

    calls = []

    class FakeCIFAR10:
        def __init__(self, root, train, download):
            calls.append((root, download))
            if root == "bad-mirror":
                raise RuntimeError("Dataset not found or corrupted.")
            n = 5 if train else 3
            self.data = np.zeros((n, 32, 32, 3), dtype=np.uint8)
            self.targets = [0] * n

    monkeypatch.setattr(tvd, "CIFAR10", FakeCIFAR10)
    monkeypatch.setenv("POF_DATA_MIRROR", "good-mirror")
    out = datasets._cifar10("data")
    assert calls == [("good-mirror", False), ("good-mirror", False)]
    assert out[0].shape == (5, 3, 32, 32)

    calls.clear()
    monkeypatch.setenv("POF_DATA_MIRROR", "bad-mirror")
    datasets._cifar10("data")
    assert calls == [("bad-mirror", False), ("data", True), ("data", True)]
