from pathlib import Path

from pof.config import config_hash, deep_merge, load_config, parse_overrides

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_deep_merge_overrides_leaves_and_keeps_siblings():
    out = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20}})
    assert out == {"a": {"x": 1, "y": 20}, "b": 3}


def test_base_inheritance_chain():
    smoke = load_config(CONFIGS / "smoke.yaml")
    assert smoke["experiment"] == "smoke"                  # own key
    assert smoke["methods"][0] == "do_nothing"             # from e0_sanity.yaml
    assert smoke["fl"]["momentum"] == 0.9                  # from default.yaml
    assert smoke["fl"]["rounds"] == 2                      # overridden
    assert "base" not in smoke


def test_hash_is_order_independent_and_sensitive():
    a = {"x": 1, "y": {"p": 2, "q": 3}}
    b = {"y": {"q": 3, "p": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({**a, "x": 2})


def test_parse_overrides_types():
    o = parse_overrides(["fl.rounds=5", "data.alpha=0.1", "seed=43", "data.name=cifar10"])
    assert o == {"fl": {"rounds": 5}, "data": {"alpha": 0.1, "name": "cifar10"}, "seed": 43}
