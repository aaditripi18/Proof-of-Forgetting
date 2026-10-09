"""Client partitioning, the in/out split behind V5, and shadow-model client subsets.

All functions are pure numpy and deterministic in their `seed`.

Index vocabulary used throughout the project:
  client_in[c]   training-set indices client c trains on (its "members")
  client_out[c]  client c's held-out indices: same distribution, never trained on
                 by anyone, known only to the client. They are V5's reference set.
"""
from __future__ import annotations

import numpy as np


def dirichlet_partition(labels: np.ndarray, num_clients: int, alpha: float, seed: int,
                        min_size: int = 1, max_tries: int = 1000) -> list[np.ndarray]:
    """Label-skewed partition: for each class, split its indices across clients
    with proportions ~ Dir(alpha). Resamples until every client has >= min_size.

    Returns sorted index arrays that are disjoint and cover every index.
    """
    labels = np.asarray(labels)
    if num_clients * min_size > len(labels):
        raise ValueError("min_size * num_clients exceeds the dataset size")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(max_tries):
        parts: list[list[int]] = [[] for _ in range(num_clients)]
        for k in classes:
            idx = np.flatnonzero(labels == k)
            rng.shuffle(idx)
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for c, chunk in enumerate(np.split(idx, cuts)):
                parts[c].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"no partition with min_size={min_size} after {max_tries} tries; "
                       "raise alpha or lower min_size")


def in_out_split(indices: np.ndarray, holdout_frac: float, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split one client's indices into (in, out). Stratification is not
    needed: the split is uniform, so in and out are exchangeable by construction,
    which is exactly what makes the V5 permutation test valid under the null."""
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(indices))
    n_out = int(round(len(perm) * holdout_frac))
    return np.sort(perm[n_out:]), np.sort(perm[:n_out])


def build_partition(labels: np.ndarray, cfg: dict, seed: int) -> dict:
    """The full partition used by every experiment for a given (data cfg, seed)."""
    from pof.seed import derive_seed

    d = cfg["data"]
    clients = dirichlet_partition(labels, d["num_clients"], d["alpha"],
                                  seed=derive_seed(seed, "dirichlet"),
                                  min_size=d["min_client_size"])
    client_in, client_out = [], []
    for c, idx in enumerate(clients):
        i, o = in_out_split(idx, d["holdout_frac"], seed=derive_seed(seed, "in_out", c))
        client_in.append(i)
        client_out.append(o)
    return {"client_in": client_in, "client_out": client_out}


def balanced_client_subsets(num_clients: int, num_shadow: int, seed: int) -> np.ndarray:
    """Membership matrix M[s, c] for shadow federations (LiRA, week 4).

    Every client is IN exactly num_shadow/2 shadow runs and every run contains
    exactly num_clients/2 clients, so each client gets equally many IN and OUT
    reference models. Built by stacking random column permutations of a
    half-ones row pattern, then fixing column balance by pairing complements.
    """
    if num_shadow % 2 or num_clients % 2:
        raise ValueError("num_shadow and num_clients must both be even")
    rng = np.random.default_rng(seed)
    half = np.zeros(num_clients, dtype=bool)
    half[: num_clients // 2] = True
    rows = []
    for _ in range(num_shadow // 2):
        r = rng.permutation(half)
        rows.extend([r, ~r])  # a row and its complement: column sums stay balanced
    m = np.array(rows)
    return m[rng.permutation(num_shadow)]
