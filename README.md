# Proof of Forgetting

**When a federated client asks to be forgotten, can it prove that it was?**

Federated unlearning methods are judged by tests that ask whether the forgotten
client is still detectable in the model. This project audits those tests rather
than the methods: it measures which tests would certify a server that did no
unlearning at all, whether a dishonest server can pass them without forgetting,
and how much data a client needs before forgetting can be verified at all.

> Status: week 1 of 10. Infrastructure and E0 are in place; no results are
> reported yet. Every number this README eventually shows will name the run it
> came from in [`results/ledger.jsonl`](results/).

## Questions

| | Question |
|---|---|
| RQ1 | Do standard checks (loss-threshold MIA AUC, backdoor canaries) pass a server that did nothing? |
| RQ2 | Can a dishonest server pass canary-based verification while the client's real influence stays? |
| RQ3 | What is the smallest client dataset for which "not forgotten" is statistically detectable? |

## How it works

Each client splits its own data uniformly at random into an **in** half, which
it trains on, and an **out** half, which it keeps private. Unlearning methods
never see the out half. After unlearning, the client compares the model's losses
on the two halves with an exact permutation test (verifier V5): a server cannot
single out these "canaries", because they are the client's ordinary data.

| Verifier | Run by | What it measures |
|---|---|---|
| V1 | anyone | loss-threshold membership inference (AUC and TPR at low FPR) |
| V2 | client | backdoor canary success rate (Sommer et al.) — week 5 |
| V3 | auditor | client-level LiRA over shadow federations — week 4 |
| V4 | server | distance to the retrain oracle — week 3 |
| V5 | client | in/out permutation test on the client's own data |

Every verifier is run against a `do_nothing` method. A verifier that certifies
it as "forgotten" cannot tell forgetting from not forgetting.

## Run it

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_e0.py --config configs/smoke.yaml       # ~1 min on CPU
.venv/Scripts/python scripts/run_e0.py --config configs/e0_sanity.yaml   # CIFAR-10, GPU
```

On a free Kaggle GPU, use [notebooks/kaggle_e0.ipynb](notebooks/kaggle_e0.ipynb)
(setup steps are in its first cell). Summarise runs across seeds with
`python scripts/summarize.py --experiment e0_sanity`.

Override any config key from the command line:
`--set seed=43 data.alpha=0.1 fl.rounds=50`.

## Reproducibility

- Every random draw is keyed by `(seed, purpose, round, client)`, and training is
  bitwise deterministic on a fixed device (tested).
- Every reported number is written to `results/ledger.jsonl` with its run id,
  config hash, seed and git commit **before** it is written to a CSV. A test
  fails if a CSV contains a number the ledger does not.
- The models use GroupNorm, not BatchNorm, so a model is a pure function of its
  parameters and no membership signal leaks through running statistics.

## Layout

```
configs/      default.yaml, e0_sanity.yaml, smoke.yaml
pof/data/     datasets (CIFAR-10, PathMNIST, synthetic), Dirichlet + in/out partition
pof/fl/       FedAvg, seeded augmentation, per-sample evaluation
pof/unlearn/  method registry: do_nothing, retrain_oracle, finetune_retain
pof/verify/   V1 loss MIA, AUC, TPR at fixed FPR
pof/stats/    permutation test, bootstrap CIs
pof/ledger.py results ledger and CSV check
scripts/      one script per experiment
tests/        unit tests, determinism, end-to-end smoke run
```

See [PLAN.md](PLAN.md) for the full research plan and timeline.
