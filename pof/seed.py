"""Seeding and derived seeds.

Every random draw in a run is keyed by (base seed, purpose, ...) through
`derive_seed`, so adding a new random step never shifts the draws of existing ones.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed from any sequence of hashable parts."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
