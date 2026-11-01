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
    if d.get("forget_size") is not None:  # optional key, absent from default.yaml
        f = cfg["unlearn"]["forget_client"]
        client_in[f], client_out[f] = shrink_client(client_in[f], client_out[f], d["forget_size"],
                                                    seed=derive_seed(seed, "forget_size", f))
    return {"client_in": client_in, "client_out": client_out}


def shrink_client(c_in: np.ndarray, c_out: np.ndarray, size: int, seed: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Keep a uniform random `size` of a client's in half, and the same share of its
    out half, so the in/out halves stay exchangeable. The rest is set aside: nobody
    trains on it and no verifier reads it, so only the client's contribution changes
    and every other client stays exactly as it was."""
    if not 0 < size <= len(c_in):
        raise ValueError(f"forget_size {size} must be in (0, {len(c_in)}]: the forget "
                         "client cannot be grown, only shrunk; forget a larger client")
    rng = np.random.default_rng(seed)
    n_out = min(len(c_out), int(round(size * len(c_out) / len(c_in))))
    return (np.sort(rng.choice(c_in, size, replace=False)),
            np.sort(rng.choice(c_out, n_out, replace=False)))


def balanced_client_subsets(num_clients: int, num_shadow: int, seed: int,
                            num_out: int | None = None) -> np.ndarray:
    """Membership matrix M[s, c] for shadow federations (LiRA).

    Every shadow leaves out exactly `num_out` clients (default: half), and every
    client is left out of exactly num_shadow * num_out / num_clients shadows.

    `num_out` matters: LiRA is calibrated only if shadows look like the models
    under test. Those are federations of all clients or all but one, so shadows
    that drop half the clients are smaller, less confident models, and any
    full-size model looks "member-like" against them. Leaving out few clients
    keeps the shadows close to the targets' size.

    Construction: concatenate random permutations of the clients and cut the
    sequence into consecutive groups of `num_out`; with num_clients % num_out == 0
    no group straddles two permutations, so no group repeats a client.
    """
    num_out = num_clients // 2 if num_out is None else num_out
    if not 0 < num_out < num_clients or num_clients % num_out:
        raise ValueError("num_out must divide num_clients and be in (0, num_clients)")
    if (num_shadow * num_out) % num_clients:
        raise ValueError(f"num_shadow * num_out must be a multiple of num_clients "
                         f"({num_shadow} * {num_out} vs {num_clients})")
    rng = np.random.default_rng(seed)
    seq = np.concatenate([rng.permutation(num_clients)
                          for _ in range(num_shadow * num_out // num_clients)])
    m = np.ones((num_shadow, num_clients), dtype=bool)
    for s, group in enumerate(seq.reshape(num_shadow, num_out)):
        m[s, group] = False
    return m


def targeted_client_subsets(num_clients: int, num_shadow: int, target: int) -> np.ndarray:
    """Membership matrix for shadows built around one client.

    The first half of the shadows are federations of every client except
    `target` (they are retrains without it: the same kind of model as the
    retrain oracle, differing only in training randomness); the second half are
    full federations (the same kind as the original model). Only the target
    client has OUT shadows, so only its samples can be scored.

    This is the design to use when one client is being forgotten: the null a
    verifier compares against is then made of genuine retrains, with no
    difference in federation size to explain away.
    """
    if num_shadow % 2 or num_shadow < 4:
        raise ValueError("num_shadow must be even and at least 4")
    if not 0 <= target < num_clients:
        raise ValueError("target is not a client")
    m = np.ones((num_shadow, num_clients), dtype=bool)
    m[: num_shadow // 2, target] = False
    return m
