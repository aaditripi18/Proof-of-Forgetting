# Findings

**Status: complete for three seeds (42, 43, 44).** Every number here is in
[RESULTS.md](RESULTS.md), which `scripts/report.py` generates from runs whose
CSVs match `results/ledger.jsonl` exactly. Read these as findings for one
setup, not as general laws (see the limits at the end).

Setup: CIFAR-10, ResNet-20 with GroupNorm, 10 clients (Dirichlet α = 0.5),
FedAvg for 30 rounds of 5 local epochs; the federated models reach 61–67% test
accuracy (the same code reaches 84.9% centrally). Client 0 asks to be
forgotten; it holds 3360, 2511 and 1855 training samples on seeds 42, 43 and 44.

## The question

When a client asks a federated server to forget it, the server usually shows
that a membership-inference test can no longer find the client. We ask what
such tests can and cannot certify, by running each of them against a server
that did nothing at all and against genuine retrains without the client.

## Results

### 1. The standard check cannot tell "did nothing" from "retrained" (3/3 seeds)

V1, the loss-threshold membership AUC on matched in/out samples, reads between
0.489 and 0.522 for all seven models on all three seeds: the server that did
nothing (0.522, 0.510, 0.495), every unlearning method, and both retrains. A
server could return the original model and report "AUC ≈ 0.5, client forgotten".

### 2. LiRA separates them, and never accuses a genuine retrain

V3, client-level LiRA judged against ten retrains without the client (Student-t
prediction test):

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| `do_nothing` | z 5.5, p 0.0003 | z 5.1, p 0.0004 | z 1.8, p 0.061 |
| `retrain_oracle` | p 0.62 | p 0.52 | p 0.59 |
| `retrain_oracle_alt` | p 0.24 | p 0.78 | p 0.76 |

Across E0 and E3, the two retrains were tested 12 times and V3 flagged none of
them at α = 0.05. It detected `do_nothing` clearly on two seeds and only
marginally on the third.

### 3. LiRA's evidence saturates at a few hundred samples

Why was seed 44 weak? Its client is the smallest, so the obvious guess is that
fewer samples means less evidence. `scripts/v3_size_curve.py` tests that guess
without retraining: it reruns V3 on random subsets of m of the client's samples,
200 subsets per size.

| samples audited (m) | 50 | 250 | 500 | 1500 |
|---|---|---|---|---|
| seed 42, median z | 3.3 | 4.8 | 5.2 | 5.5 |
| seed 43, median z | 2.1 | 3.9 | 4.6 | 4.9 |
| seed 44, median z | 1.1 | 1.6 | 1.8 | 1.8 |

The evidence levels off by 250–500 samples, and at the same m = 1500 seed 44
stays at z 1.8 while the others are near 5. So the number of samples is not
why seed 44 is weak; past a few hundred samples, what limits detection is how
strongly that federation memorised the client, which varied about threefold
between seeds. The retrains were flagged in at most 6.5% of subsets at any m (5% is
expected by chance at α = 0.05), and in none once m ≥ 1000.

This covers the audit side only. It does not model a client that trained on
fewer samples, which could be memorised more or less per sample.

### 4. Fine-tuning on the remaining clients is unreliable as forgetting

`finetune_retain` keeps its accuracy and was detected by V3 on seed 42
(p 0.005) but not on 43 or 44 (p 0.88, 0.17). Gradient ascent with recovery
rounds was not detected on any seed, at a cost of under 3 points of accuracy. SCRUB
and FedEraser-lite were not detected either, but FedEraser-lite loses 14–22
points; a model that "forgets" by getting worse at everything is not the oracle.

### 5. The backdoor canary is the last signal to go, not the first (3/3 seeds)

We expected a client's backdoor canary to be forgotten before its ordinary
influence (H2). Following gradient ascent step by step shows the opposite order
on every seed. Step at which each verifier first stops detecting the client:

| | V5 (client's own data) | V3 (LiRA) | V2 (canary) |
|---|---|---|---|
| seed 42 | 20 | 28 | 44 (accuracy at chance) |
| seed 43 | 0 (never detected) | 8 | 12 |
| seed 44 | 0 (never detected) | 20 | 28 |

Gradient ascent cannot remove the backdoor before it has removed every other
trace of the client, and on seed 42 not before destroying the model.

### 6. A dishonest server beat the canary once; LiRA caught it every time

`canary_suppressor` never unlearns. It reconstructs triggers with Neural Cleanse
on its own data and tries to erase the backdoor (H3). It picked the wrong
trigger class on all three seeds, yet:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| V2 (canary) still detects the client | yes, p < 0.0001 | barely, p 0.0003 | **no, p 0.97** |
| V3 (LiRA) still detects the client | z 16.3 | z 13.0 | z 2.5, p 0.021 |

On seed 44 the canary check would have certified a server that forgot nothing.
LiRA was not fooled on any seed. The canary was also weakest on seed 44 to begin
with (trigger success 9% on `do_nothing`, against 90% on seed 42), which is
likely why suppression worked there.

### 7. A client on its own can verify very little

V5 needs nothing but the client's own data: it compares the model's confidence
on the half the client trained with and the half it held back. It detected
`do_nothing` only on seed 42 (p 0.0002; 0.10 and 0.59 on 43 and 44). In E4 it
detected `do_nothing` only at the largest size tested (2500 samples per half),
on seed 42 only.

## What this adds up to

- The check most often reported for unlearning, loss-threshold membership AUC,
  cannot tell a server that forgot from one that did nothing.
- Client-level LiRA can, with no false alarms on genuine retrains, but it needs
  models retrained without the client, which only an auditor with compute can
  produce; and its strength depends on how much the model memorised the client,
  not on how many samples are audited past a few hundred.
- Backdoor canaries outlast every other signal under gradient ascent, but a
  server that never unlearned got past them once in three seeds.
- A client cannot verify forgetting from its own data alone at the client sizes
  and model quality studied here.

## Design changes made after seeing data

Each was made for a stated reason and is recorded in the commit history.
Seeds 43 and 44 ran after all of them, so they test whether the changes hold.

| Change | Why | Evidence it rests on |
|---|---|---|
| 5 local epochs instead of 2 | 2 gave ~43% test accuracy and almost no memorisation | tuning run, seed 42 |
| V5 compares logit-scaled confidence, not loss | mean loss is heavy-tailed and flagged a retrain oracle whose accuracy on both halves was identical | seed 42 only; chosen post hoc |
| V3's null is ten retrains without the client, with a Student-t prediction test | with a 4-shadow null of smaller federations, a second retrain oracle scored z = 4.3 | seed 42; held on 43 and 44 (no false alarms) |
| SCRUB max-steps capped; FedEraser replays every round | uncapped SCRUB fell to chance; FedEraser to 23% | first full run |

V5 was chosen post hoc on seed 42 and then failed to detect `do_nothing` on
seeds 43 and 44, so its seed-42 result should not be read as evidence for it.

## Limits

- One dataset, one architecture, one non-IID level, one forgotten client per
  seed, three seeds, 61–67% test accuracy.
- The canary result is for one attacker (an adaptive Neural Cleanse variant).
- V3 assumes the model under test resembles its reference models. For a model
  that has collapsed (accuracy at chance) its score is extreme and means little;
  read V3 together with test accuracy.
- V4 (distance to the oracle in weight space) is not informative here: two
  retrains differ from each other about as much as the original differs from
  either. Prediction-space distance is reported instead.
