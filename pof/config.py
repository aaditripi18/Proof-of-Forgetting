"""YAML configs with single inheritance (`base:`) and a stable content hash.

The hash covers the fully merged config, so two runs share a hash exactly when
they were asked to do the same thing. It is recorded next to every number.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`; override wins."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a config, resolving `base:` relative to the file, then apply overrides."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_merge(load_config(path.parent / base), cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(cfg: dict) -> str:
    """Short, order-independent hash of a merged config."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def parse_overrides(pairs: list[str]) -> dict:
    """['fl.rounds=5', 'data.alpha=0.1'] -> nested dict, values parsed as YAML."""
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return out
