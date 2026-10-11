import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from pof.verify import lira
from pof.verify.loss_mia import auc

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_bank(S=40, N=2000, signal=1.0, seed=0):
    """phi = per-sample difficulty + signal * member + noise. Difficulty varies a
    lot between samples, which is exactly what a global threshold cannot handle."""
    rng = np.random.default_rng(seed)
    # balanced like a real bank: every sample IN exactly half the shadows
    member = np.array([rng.permutation(S) < S // 2 for _ in range(N)]).T
    difficulty = rng.normal(0, 3.0, size=N)
    phi = difficulty + signal * member + rng.normal(0, 0.5, size=(S, N))
    return phi, member


def test_gaussian_logpdf_matches_closed_form():
    v = lira.gaussian_logpdf(np.array([1.0]), np.array([0.0]), 4.0)[0]
    assert v == pytest.approx(-0.5 * math.log(2 * math.pi * 4.0) - 1 / 8)


def test_fit_recovers_shift():
    phi, member = _synthetic_bank(signal=1.0)
    st = lira.fit(phi, member)
    assert np.mean(st["mu_in"] - st["mu_out"]) == pytest.approx(1.0, abs=0.05)
    assert st["var_in"] == pytest.approx(0.25, rel=0.1)  # fixed variance = noise var


def test_fit_needs_two_per_side():
    phi = np.zeros((3, 5))
    member = np.array([[1] * 5, [0] * 5, [0] * 5], dtype=bool)
    with pytest.raises(ValueError):
        lira.fit(phi, member)


def test_per_sample_calibration_beats_global_threshold():
    """LiRA's whole advantage: per-sample reference distributions."""
    phi, member = _synthetic_bank(signal=1.0)
    lira_auc = auc(*_split(lira.loo_scores(phi, member), member))
    global_auc = auc(*_split(phi, member))
    assert lira_auc > 0.85
    assert global_auc < 0.65


def test_leave_one_out_prevents_self_leakage():
    """With no membership signal, attacking a shadow with statistics that include
    itself still 'detects' membership; leave-one-out does not."""
    phi, member = _synthetic_bank(signal=0.0, S=8)
    loo_auc = auc(*_split(lira.loo_scores(phi, member), member))
    stats = lira.fit(phi, member)
    leaky = np.stack([lira.score(phi[s], stats) for s in range(len(phi))])
    assert loo_auc == pytest.approx(0.5, abs=0.03)
    assert auc(*_split(leaky, member)) > 0.55


def test_client_scores_and_test():
    s = np.array([[1.0, 3.0, 5.0, 7.0]])
    owner = np.array([0, 0, 1, 1])
    assert np.allclose(lira.client_scores(s, owner, 2), [2.0, 6.0])
    t = lira.client_test(10.0, np.array([0.0, 1.0, -1.0, 0.5]))
    assert t["p_emp"] == pytest.approx(1 / 5)
    # 11.6 sd from a null of only four points: strong, but nowhere near the 1e-30
    # a normal table would claim
    assert t["z"] > 5 and 1e-4 < t["p"] < 0.01 and t["n_null"] == 4


def test_student_t_tail_matches_tables():
    assert lira.student_t_sf(0.0, 9) == pytest.approx(0.5, abs=1e-9)
    assert lira.student_t_sf(2.262157, 9) == pytest.approx(0.025, abs=1e-5)
    assert lira.student_t_sf(3.182446, 3) == pytest.approx(0.025, abs=1e-5)
    assert lira.student_t_sf(-2.262157, 9) == pytest.approx(0.975, abs=1e-5)
    assert lira.student_t_sf(1.644854, 100000) == pytest.approx(0.05, abs=1e-4)  # -> normal


def test_client_test_is_calibrated_with_a_small_null():
    """A genuinely forgotten model is one more draw from the null population: its
    p-value must be uniform even though the null has only ten members. A z-score
    read off a normal table fails this (too many small p-values)."""
    rng = np.random.default_rng(0)
    ps, naive = [], []
    for _ in range(4000):
        draws = rng.normal(3.0, 2.0, size=11)
        t = lira.client_test(draws[0], draws[1:])
        ps.append(t["p"])
        naive.append(lira.normal_sf(t["z"]))
    ps, naive = np.array(ps), np.array(naive)
    assert np.mean(ps < 0.05) == pytest.approx(0.05, abs=0.012)
    assert np.mean(ps < 0.01) == pytest.approx(0.01, abs=0.006)
    assert np.mean(naive < 0.01) > 0.02   # the uncorrected version over-rejects


def _split(scores, member):
    return scores[member], scores[~member]


# ---------------------------------------------------------------- shadow bank

torch_needed = pytest.mark.skipif(not __import__("importlib").util.find_spec("torch"),
                                  reason="needs torch")


def _cfg_overrides(out):
    return [f"output.dir={out.as_posix()}", f"output.ledger={(out / 'ledger.jsonl').as_posix()}"]


@torch_needed
def test_bank_is_shardable_resumable_and_checked(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_shadows

    cfg = str(ROOT / "configs" / "smoke_lira.yaml")
    a = run_shadows.main(["--config", cfg, "--shard", "0/2", "--set", *_cfg_overrides(tmp_path)])
    b = run_shadows.main(["--config", cfg, "--shard", "1/2", "--set", *_cfg_overrides(tmp_path)])
    c = run_shadows.main(["--config", cfg, "--set", *_cfg_overrides(tmp_path)])
    assert a["trained"] + b["trained"] == 8 and c == {"trained": 0, "skipped": 8}

    from pof.config import load_config, parse_overrides
    from pof.experiment import prepare
    from pof.shadows import bank_dir, check_compatible, load_bank

    conf = load_config(cfg, parse_overrides(_cfg_overrides(tmp_path)))
    bank = load_bank(bank_dir(conf))
    assert bank["lsc"].shape[0] == 8 and bank["membership"].shape == (8, 4)
    check_compatible(bank, conf, prepare(conf)["part"])

    other = load_config(cfg, parse_overrides(_cfg_overrides(tmp_path) + ["seed=1"]))
    with pytest.raises(RuntimeError):
        check_compatible(bank, other, prepare(other)["part"])

    # a tampered manifest makes the bank refuse to train into it
    mpath = bank_dir(conf) / "manifest.json"
    m = json.loads(mpath.read_text())
    m["membership"][0] = m["membership"][1]
    mpath.write_text(json.dumps(m))
    with pytest.raises(RuntimeError):
        run_shadows.main(["--config", cfg, "--set", *_cfg_overrides(tmp_path)])


@torch_needed
def test_e5_and_v3_end_to_end(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_e0
    import run_e5
    import run_shadows

    from pof.ledger import check_csv_against_ledger

    run_shadows.main(["--config", str(ROOT / "configs" / "smoke_lira.yaml"),
                      "--set", *_cfg_overrides(tmp_path)])
    rows = run_e5.main(["--config", str(ROOT / "configs" / "smoke_lira.yaml"),
                        "--set", *_cfg_overrides(tmp_path)])
    assert [r["method"] for r in rows] == ["loss_sample", "loss_client", "lira_sample", "lira_client"]
    e5_csv = next((tmp_path / "smoke_e5").glob("*.csv"))
    assert check_csv_against_ledger(e5_csv, tmp_path / "ledger.jsonl") == []

    bank = next((tmp_path / "shadows").iterdir())
    run_e0.main(["--config", str(ROOT / "configs" / "smoke.yaml"), "--shadows", str(bank),
                 "--set", *_cfg_overrides(tmp_path)])
    e0_csv = next((tmp_path / "smoke").glob("*.csv"))
    assert check_csv_against_ledger(e0_csv, tmp_path / "ledger.jsonl") == []
    with open(e0_csv, encoding="utf-8") as f:
        r = {row["method"]: row for row in csv.DictReader(f)}
    for m in r.values():
        assert 0.0 < float(m["v3_p_emp"]) <= 1.0
    # the model that trained on the client scores as more member-like than the oracle
    assert float(r["do_nothing"]["v3_score"]) > float(r["retrain_oracle"]["v3_score"])
