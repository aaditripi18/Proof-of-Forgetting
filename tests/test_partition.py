import numpy as np
import pytest

from pof.data.partition import (balanced_client_subsets, build_partition,
                                dirichlet_partition, in_out_split)

LABELS = np.random.default_rng(0).integers(0, 10, size=5000)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 100.0])
def test_dirichlet_is_disjoint_and_covers(alpha):
    parts = dirichlet_partition(LABELS, 10, alpha, seed=1, min_size=10)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(LABELS)
    assert len(np.unique(allidx)) == len(LABELS)
    assert min(len(p) for p in parts) >= 10


def test_dirichlet_is_deterministic_and_seed_sensitive():
    a = dirichlet_partition(LABELS, 10, 0.5, seed=1)
    b = dirichlet_partition(LABELS, 10, 0.5, seed=1)
    c = dirichlet_partition(LABELS, 10, 0.5, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def test_low_alpha_is_more_skewed_than_high_alpha():
    def mean_label_entropy(parts):
        ent = []
        for p in parts:
            q = np.bincount(LABELS[p], minlength=10) / len(p)
            q = q[q > 0]
            ent.append(-(q * np.log(q)).sum())
        return np.mean(ent)
    skewed = dirichlet_partition(LABELS, 10, 0.1, seed=3, min_size=10)
    iid = dirichlet_partition(LABELS, 10, 100.0, seed=3, min_size=10)
    assert mean_label_entropy(skewed) < mean_label_entropy(iid) - 0.5


def test_in_out_split_disjoint_and_sized():
    idx = np.arange(100, 301)
    i, o = in_out_split(idx, 0.5, seed=0)
    assert len(np.intersect1d(i, o)) == 0
    assert np.array_equal(np.sort(np.concatenate([i, o])), idx)
    assert abs(len(o) - 100) <= 1


def test_out_halves_never_overlap_any_in_half():
    cfg = {"data": {"num_clients": 10, "alpha": 0.5, "min_client_size": 20, "holdout_frac": 0.5}}
    part = build_partition(LABELS, cfg, seed=7)
    all_in = np.concatenate(part["client_in"])
    all_out = np.concatenate(part["client_out"])
    assert len(np.intersect1d(all_in, all_out)) == 0
    assert len(np.unique(all_in)) == len(all_in)


@pytest.mark.parametrize("num_clients,num_shadow", [(10, 32), (4, 8), (10, 64)])
def test_shadow_subsets_balanced(num_clients, num_shadow):
    m = balanced_client_subsets(num_clients, num_shadow, seed=0)
    assert m.shape == (num_shadow, num_clients)
    assert (m.sum(axis=0) == num_shadow // 2).all()   # every client IN half the runs
    assert (m.sum(axis=1) == num_clients // 2).all()  # every run has half the clients
