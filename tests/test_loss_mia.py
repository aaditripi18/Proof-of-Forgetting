import numpy as np
import pytest

from pof.verify.loss_mia import auc, loss_mia, tpr_at_fpr


def test_auc_extremes_and_ties():
    assert auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auc([1, 1, 1], [1, 1]) == 0.5


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    m, n = rng.integers(0, 5, 40), rng.integers(0, 5, 30)  # many ties on purpose
    pairwise = np.mean([(a > b) + 0.5 * (a == b) for a in m for b in n])
    assert auc(m, n) == pytest.approx(pairwise)


def test_auc_is_half_for_same_distribution():
    rng = np.random.default_rng(1)
    assert auc(rng.normal(size=20000), rng.normal(size=20000)) == pytest.approx(0.5, abs=0.01)


def test_tpr_at_fpr_respects_the_fpr_budget():
    rng = np.random.default_rng(2)
    nonmem = rng.normal(size=1000)
    # a threshold just above the top non-member: FPR exactly 0
    assert tpr_at_fpr([10.0, -10.0], nonmem, 0.0) == 0.5
    # ties with the threshold never count for the attacker
    assert tpr_at_fpr([nonmem.max()], nonmem, 0.0) == 0.0


def test_auc_half_can_hide_low_fpr_leakage():
    """The point of E5 in one test: 2% of members are confidently identifiable
    while the AUC is indistinguishable from chance."""
    rng = np.random.default_rng(3)
    nonmem = rng.normal(size=20000)
    mem = rng.normal(size=20000)
    mem[:400] = 10.0  # a few memorised outliers
    assert auc(mem, nonmem) == pytest.approx(0.5, abs=0.02)
    assert tpr_at_fpr(mem, nonmem, 0.001) >= 0.02


def test_loss_mia_uses_negative_loss():
    out = loss_mia(member_losses=[0.1, 0.2], nonmember_losses=[1.0, 2.0])
    assert out["auc"] == 1.0
    assert set(out) == {"auc", "tpr@0.01", "tpr@0.001"}
