# Research plan

Working title: **Proof of Forgetting: what federated unlearning verification can and cannot certify**

## Motivation

A loss-threshold membership-inference test (members vs a pooled validation set)
is the common way to "verify" federated unlearning. In a previous project, that
test scored the *original, pre-unlearning* model at AUC ≈ 0.5, the same as the
retrain oracle. A test that cannot separate the model before unlearning from the
gold standard after it would also certify a server that never unlearned.
E0 reproduces this on public data.

Likely causes, each isolated by an experiment:
1. the attack is weak and average-case (E5: AUC vs TPR at low FPR);
2. the non-member set is the pooled population while the client is non-IID, so
   client-vs-population shift is mixed into the membership signal (E6);
3. AUC averages away leakage concentrated in a few samples (E5).

## Hypotheses (fixed before running)

- **H1** `do_nothing` passes V1 (loss-threshold MIA AUC).
- **H2** Backdoor canaries are memorised sharply and forgotten first: a few steps
  of gradient ascent remove the canary while LiRA still detects the client.
- **H3** A dishonest server can suppress canaries generically (Neural Cleanse +
  trigger erasure) without knowing the client's secret and without forgetting.
- **H4** V5 cannot be passed without forgetting, but has a minimum client data
  size below which it has no power.

## Setup

| | |
|---|---|
| Datasets | CIFAR-10; PathMNIST 64×64 (medical, multi-class) |
| Model | ResNet-20 with GroupNorm |
| Federation | 10 clients, Dirichlet α ∈ {0.1, 0.5, 1.0}, FedAvg, 30 rounds, full participation |
| Client data | uniformly split into in (trained) and out (private reference) halves |
| Seeds | 3 for tables, 5 for the headline figure |
| Shadow federations (LiRA) | 40 runs, each leaving out 2 of 10 clients, so shadows match the size of the models under test; each client OUT of exactly 8 |

## Methods (under test)

`do_nothing` · `retrain_oracle` · `finetune_retain` · `gradient_ascent` · `scrub` ·
`federaser_lite` · `canary_suppressor` (dishonest) · private methods via
`--external-methods` (never committed).

## Verifiers (audited)

V1 loss MIA · V2 backdoor canary · V3 client-level LiRA (TPR @ 1%, 0.1% FPR) ·
V4 distance to oracle · V5 in/out permutation test.

## Experiments

| ID | Experiment | Output |
|---|---|---|
| E0 | V1 on do_nothing vs oracle, pooled vs matched non-members | the sanity finding |
| E1 | methods × verifiers × α | main table |
| E2 | canary vs real influence across gradient-ascent steps | two curves (H2) |
| E3 | canary_suppressor vs every verifier | which tests a dishonest server passes (H3) |
| E4 | V5 power vs client size n ∈ {25…1000} × α × method | **headline: the verifiability limit** (H4) |
| E5 | AUC vs TPR at low FPR on the same models | why AUC hides leakage |
| E6 | pooled vs client-matched non-member reference | the distribution confound |
| E7 | replication on PathMNIST | generalisation |

## Timeline

| Week | Work | Done when |
|---|---|---|
| 1 | ✅ skeleton, partition, FedAvg, ledger, determinism, E0 script, tests | smoke run passes, ledger check passes |
| 2 | ✅ gradient_ascent (projected, early-stopped, optional recovery); run E0 on CIFAR-10 (3 seeds) | E0 numbers in ledger |
| 3 | ✅ V4 + noise floor (retrain_oracle_alt); write the E0 finding as one paragraph (needs Kaggle results) | finding drafted |
| 4 | ✅ shadow bank (resumable, 2-GPU), client-level LiRA (V3), E5, Kaggle notebook | LiRA passes toy test |
| 5 | backdoor canaries (V2), SCRUB, FedEraser-lite; E2 | canary curve |
| 6 | canary_suppressor (Neural Cleanse); E3 | dishonest-server table |
| 7 | V5 power analysis machinery; calibration on oracle | uniform p-values on real models |
| 8 | E4, E6 | headline figure |
| 9 | E7, multi-seed, final tables | every number in ledger |
| 10 | demo, project page, explainer, paper draft | live |

## Risks

| Risk | Fallback |
|---|---|
| LiRA compute | 20 shadows (still 4 OUT per client), or 5 clients |
| dishonest server fails | report it; "canaries are harder to fake than expected" is a result |
| V5 low power everywhere | that is the limit result |
| GPU budget | one α + PathMNIST, 3 seeds |

## Reading

- Carlini et al. 2022, *Membership Inference Attacks From First Principles* (LiRA)
- Sommer et al. 2022, *Athena: Probabilistic Verification of Machine Unlearning*
- Kurmanji et al. 2023, *Towards Unbounded Machine Unlearning* (SCRUB)
- Liu et al. 2021, *FedEraser*
- Wang et al. 2019, *Neural Cleanse*
- Thudi et al. 2022, *On the Necessity of Auditable Algorithmic Definitions for Machine Unlearning*

## Confidentiality

This repository is public-safe: public datasets and published methods only.
Private methods are loaded at runtime from a path outside the repo
(`private/` is git-ignored) and never committed.
