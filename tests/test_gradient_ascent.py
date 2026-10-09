import copy
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pof.config import load_config  # noqa: E402
from pof.data.datasets import load_dataset  # noqa: E402
from pof.data.partition import build_partition  # noqa: E402
from pof.fl.fedavg import per_sample_eval, train_federated  # noqa: E402
from pof.fl.tensors import TensorData  # noqa: E402
from pof.models import build_model  # noqa: E402
from pof.unlearn.gradient_ascent import ascend, project_to_ball  # noqa: E402
from pof.unlearn.registry import get_method  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def ctx():
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    ds = load_dataset(cfg, seed=0)
    part = build_partition(ds["train_y"], cfg, seed=0)
    train = TensorData(ds["train_x"], ds["train_y"], ds["mean"], ds["std"], CPU)
    clients = list(range(cfg["data"]["num_clients"]))
    original = train_federated(cfg, train, part["client_in"], clients, ds["num_classes"], seed=0)
    return {"cfg": cfg, "seed": 0, "data": train, "client_in": part["client_in"],
            "num_classes": ds["num_classes"], "clients": clients, "forget_client": 0,
            "original_state": original}


def _with_ga(ctx, **ga):
    c = dict(ctx)
    c["cfg"] = copy.deepcopy(ctx["cfg"])
    c["cfg"]["unlearn"]["ga"].update(ga)
    return c


def _forget_loss(ctx, state):
    return per_sample_eval(ctx["cfg"], state, ctx["data"], ctx["client_in"][0],
                           ctx["num_classes"])["loss"].mean()


def test_projection_keeps_params_in_ball():
    center = [torch.zeros(3), torch.zeros(2)]
    params = [torch.tensor([3.0, 0.0, 0.0]), torch.tensor([0.0, 4.0])]
    before = project_to_ball(params, center, radius=1.0)
    assert before == pytest.approx(5.0)
    after = math.sqrt(sum(float(p.pow(2).sum()) for p in params))
    assert after == pytest.approx(1.0)
    # direction is preserved
    assert torch.allclose(params[0], torch.tensor([0.6, 0.0, 0.0]))


def test_ascent_raises_forget_loss_without_recovery(ctx):
    c = _with_ga(ctx, recovery_rounds=0, radius_frac=None, max_loss=100.0, epochs=2, lr=0.05)
    state = get_method("gradient_ascent")(c)
    assert _forget_loss(ctx, state) > _forget_loss(ctx, ctx["original_state"])


def test_ascent_stays_within_radius(ctx):
    c = _with_ga(ctx, recovery_rounds=0, radius_frac=0.01, max_loss=100.0, epochs=2, lr=0.5)
    state = get_method("gradient_ascent")(c)
    orig = ctx["original_state"]
    keys = [k for k, v in orig.items() if v.is_floating_point()]
    dist = math.sqrt(sum(float((state[k] - orig[k]).pow(2).sum()) for k in keys))
    norm = math.sqrt(sum(float(orig[k].pow(2).sum()) for k in keys))
    assert dist <= 0.01 * norm * (1 + 1e-5)


def test_early_stop_at_max_loss(ctx):
    model = build_model(ctx["cfg"], ctx["num_classes"])
    model.load_state_dict(ctx["original_state"])
    info = ascend(model, ctx["data"], ctx["client_in"][0], epochs=5, lr=0.05, batch_size=32,
                  max_loss=0.0, radius=None, augment=False, seed=0)
    assert info == {"steps": 0, "stop": "max_loss", "last_loss": pytest.approx(info["last_loss"])}
    # the model must be untouched when ascent stops before its first step
    assert all(torch.equal(model.state_dict()[k], v) for k, v in ctx["original_state"].items())


def test_only_forget_client_data_is_ascended(ctx):
    """With recovery off, ascent sees only client 0's in-half: scrambling every
    other client's labels must not change the result."""
    c = _with_ga(ctx, recovery_rounds=0, epochs=1)
    a = get_method("gradient_ascent")(c)
    scrambled = dict(c)
    data = copy.copy(ctx["data"])
    y = data.y.clone()
    others = np.concatenate([ctx["client_in"][i] for i in ctx["clients"][1:]])
    y[torch.from_numpy(others)] = (y[torch.from_numpy(others)] + 1) % ctx["num_classes"]
    data.y = y
    scrambled["data"] = data
    b = get_method("gradient_ascent")(scrambled)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_ascent_is_deterministic(ctx):
    a = get_method("gradient_ascent")(ctx)
    b = get_method("gradient_ascent")(ctx)
    assert all(torch.equal(a[k], b[k]) for k in a)
