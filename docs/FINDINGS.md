# Findings

**Status: complete for six seeds (42–47)**, plus two client swaps (seed 42
forgetting client 5, seed 45 forgetting client 8). E4 covers seeds 42–44 only.
Every number here is in [RESULTS.md](RESULTS.md), which `scripts/report.py`
generates from runs whose CSVs match `results/ledger.jsonl` exactly. Read these
as findings for one setup, not as general laws (see the limits at the end).

Setup: CIFAR-10, ResNet-20 with GroupNorm, 10 clients (Dirichlet α = 0.5),
FedAvg for 30 rounds of 5 local epochs; the federated models reach 61–67% test
accuracy (the same code reaches 84.9% centrally). Client 0 asks to be
forgotten; it holds 3360, 2511, 1855, 1670, 4593 and 3667 training samples on
seeds 42 to 47.

Seeds 42–44 were the original study. Seeds 45–47 ran later, on the same code,
and several results below changed when they were added; each such change is
stated where it happens.

## The question

When a client asks a federated server to forget it, the server usually shows
that a membership-inference test can no longer find the client. We ask what
such tests can and cannot certify, by running each of them against a server
that did nothing at all and against genuine retrains without the client.

## Results

### 1. The standard check cannot tell "did nothing" from "retrained" (6/6 seeds)

V1, the loss-threshold membership AUC on matched in/out samples, reads between
0.489 and 0.526 for all seven models on all six seeds: the server that did
nothing (0.522, 0.510, 0.495, 0.509, 0.520, 0.526), every unlearning method,
and both retrains (0.489–0.504). A server could return the original model and
report "AUC ≈ 0.5, client forgotten".

### 2. LiRA separates them when the client is large, and never accuses a genuine retrain

V3, client-level LiRA judged against ten retrains without the client (Student-t
prediction test):

| | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | seed 47 |
|---|---|---|---|---|---|---|
| client size | 3360 | 2511 | 1855 | 1670 | 4593 | 3667 |
| `do_nothing` | z 5.5, p 0.0003 | z 5.1, p 0.0004 | z 1.8, p 0.061 | z 0.2, p 0.43 | z 5.3, p 0.0004 | z 6.6, p < 0.0001 |
| `retrain_oracle` | p 0.62 | p 0.52 | p 0.59 | p 0.79 | p 0.67 | p 0.23 |
| `retrain_oracle_alt` | p 0.24 | p 0.78 | p 0.76 | p 0.62 | p 0.67 | p 0.95 |

Across E0 and E3, the two retrains were tested 24 times (and 4 more times in the
client swaps), and V3 flagged none of them at α = 0.05. It detected
`do_nothing` clearly on four seeds, marginally on seed 44, and not at all on
seed 45. With three seeds this read as "2 of 3"; with six, the misses are the
two seeds with the smallest clients (see finding 3).

### 3. LiRA's evidence depends on how much the client contributed, not on how much is audited

`scripts/v3_size_curve.py` reruns V3 on random subsets of m of the client's
samples, 200 subsets per size, without retraining:

| samples audited (m) | 50 | 250 | 500 | 1500 |
|---|---|---|---|---|
| seed 42, median z | 3.3 | 4.8 | 5.2 | 5.5 |
| seed 43, median z | 2.1 | 3.9 | 4.6 | 4.9 |
| seed 44, median z | 1.1 | 1.6 | 1.8 | 1.8 |
| seed 45, median z | 0.2 | 0.2 | 0.2 | 0.2 |
| seed 46, median z | 4.5 | 5.0 | 5.1 | 5.2 |
| seed 47, median z | 4.7 | 6.0 | 6.4 | 6.5 |

On every seed the evidence levels off by 250–500 samples, so auditing more of
the client's samples does not rescue a weak result. The retrains were flagged
in at most 6.5% of subsets at any m (5% is expected by chance at α = 0.05), and
in none once m ≥ 1000.

What does decide it is how much data the client trained with. Across all eight
runs (six seeds and two client swaps), every forgotten client with at least
2511 training samples was detected (5/5, z 4.4–6.6, p ≤ 0.0011), and every
client with at most 1855 was not (0/3, z 0.2–1.9, p ≥ 0.050). The client swaps
show that this is the client and not the federation: in the same seed-45
federation, client 0 (1670 samples) gives z 0.2 and client 8 (3345) gives
z 4.4; in seed 42, client 0 (3360) gives z 5.5 and client 5 (1515) gives
z 1.9 ([figure](figures/client_size.svg)).

With three seeds we wrote that "the number of samples is not why seed 44 is
weak" and blamed differences between federations. That was right about the
audit side and wrong about the cause: seed 44's client is simply small. Eight
points and two swaps are still few, so the threshold (somewhere between about
1900 and 2500 samples here) should not be read as a constant.

### 4. Fine-tuning on the remaining clients is unreliable as forgetting

`finetune_retain` keeps its accuracy and was detected by V3 on seeds 42 and 47
(p 0.005, 0.0008) but not on the other four. Gradient ascent with recovery
rounds costs under 4 points of accuracy and was detected on seed 47 only
(p 0.003). SCRUB was detected on seeds 46 and 47 (p 0.03, 0.02).
FedEraser-lite was never detected, but loses 11–22 points; a model that
"forgets" by getting worse at everything is not the oracle.

With three seeds, gradient ascent and SCRUB looked undetected everywhere. On
seed 47, where the model memorised its client most strongly (z 6.6), every
method that kept its accuracy was still detected.

### 5. The backdoor canary usually outlasts the other signals, but not always (4/6 seeds)

We expected a client's backdoor canary to be forgotten before its ordinary
influence (H2). Following gradient ascent step by step, step at which each
verifier first stops detecting the client:

| | V5 (client's own data) | V3 (LiRA) | V2 (canary) |
|---|---|---|---|
| seed 42 | 20 | 28 | 44 (accuracy at chance) |
| seed 43 | 0 (never detected) | 8 | 12 |
| seed 44 | 0 (never detected) | 20 | 28 |
| seed 45 | 4 | 4 | 24 |
| seed 46 | 24 | never | 44 (accuracy at chance) |
| seed 47 | 28 | 28 | **24** |

On seeds 42–45 the canary went last, the opposite of H2. On seed 46 LiRA
outlasted it, but only into a model at chance accuracy, where V3 means little
(see limits). On seed 47 the canary went first, as H2 predicted, at 58%
accuracy. With three seeds this was "3/3, the canary is last"; with six it is
the usual order, not a rule.

### 6. A dishonest server beat the canary twice and LiRA once

`canary_suppressor` never unlearns. It reconstructs triggers with Neural Cleanse
on its own data and tries to erase the backdoor (H3).

| | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | seed 47 |
|---|---|---|---|---|---|---|
| Neural Cleanse found the client's class | no | no | no | no | **yes** | no |
| V2 (canary) still detects the client | p < 0.0001 | p 0.0003 | **no, p 0.97** | p < 0.0001 | **no, p 1.00** | p 0.032 |
| V3 (LiRA) still detects the client | z 16.3 | z 13.0 | z 2.5, p 0.021 | z 2.6, p 0.018 | **no, z 1.1, p 0.15** | z 11.4 |
| V5 (client's own data) | p 0.048 | p 0.27 | p 0.30 | p 0.14 | p 0.0003 | p 0.013 |

The canary check would have certified a server that forgot nothing on seeds 44
and 46. On seed 44 the canary was weak to begin with (trigger success 9% on
`do_nothing`, against 90% on seed 42). On seed 46, the one seed where Neural
Cleanse found the client's actual target class, the suppressor also got past
LiRA, at a cost of 7 points of accuracy; only the client's own in/out test
still caught it. With three seeds this finding read "LiRA caught it every
time"; that no longer holds.

The canary check also has false alarms: it flagged a genuine retrain 3 times
in 12 (both retrains on seed 45, one on seed 47), against none in 6 on seeds
42–44.

### 7. A client can verify on its own data only if it contributed a lot of it

V5 needs nothing but the client's own data: it compares the model's confidence
on the half the client trained with and the half it held back. It detected
`do_nothing` on seeds 42, 46 and 47 (p 0.0002, 0.0002, < 0.0001) and not on
43, 44 or 45 (0.10, 0.59, 0.10). Including the client swaps, it detected every
forgotten client with at least 3345 training samples (4/4) and none with 2511
or fewer (0/4). It never flagged a retrain. In E4 (seeds 42–44 only) it
detected `do_nothing` only at the largest size tested (2500 samples per half),
on seed 42 only.

## What this adds up to

- The check most often reported for unlearning, loss-threshold membership AUC,
  cannot tell a server that forgot from one that did nothing (6/6 seeds).
- Client-level LiRA can, with no false alarms in 28 tests on genuine retrains,
  but only for clients that contributed enough data (here, about 2500 samples
  or more), and it needs models retrained without the client, which only an
  auditor with compute can produce. Auditing more samples does not help past a
  few hundred.
- Backdoor canaries usually outlast other signals under gradient ascent, but
  not always; a server that never unlearned got past them on 2 of 6 seeds, and
  they wrongly accused a genuine retrain 3 times in 12.
- A server that found the right trigger class got past both the canary and
  LiRA once. No single check caught the dishonest server on every seed.
- A client can check forgetting from its own data alone only if it contributed
  a lot of data (here, about 3300 samples or more).

## Design changes made after seeing data

Each was made for a stated reason and is recorded in the commit history.
Seeds 43–47 ran after all of them, so they test whether the changes hold.

| Change | Why | Evidence it rests on |
|---|---|---|
| 5 local epochs instead of 2 | 2 gave ~43% test accuracy and almost no memorisation | tuning run, seed 42 |
| V5 compares logit-scaled confidence, not loss | mean loss is heavy-tailed and flagged a retrain oracle whose accuracy on both halves was identical | seed 42 only; chosen post hoc |
| V3's null is ten retrains without the client, with a Student-t prediction test | with a 4-shadow null of smaller federations, a second retrain oracle scored z = 4.3 | seed 42; held on 43–47 (no false alarms) |
| SCRUB max-steps capped; FedEraser replays every round | uncapped SCRUB fell to chance; FedEraser to 23% | first full run |

V5 was chosen post hoc on seed 42. It then failed to detect `do_nothing` on
seeds 43, 44 and 45 and succeeded on 46 and 47, the seeds with the largest
clients, so read it as a check that works only for large clients.

## Limits

- One dataset, one architecture, one non-IID level, 61–67% test accuracy.
- One forgotten client per seed plus two client swaps: eight forgotten clients
  in all, which is few for the client-size thresholds in findings 3 and 7.
- Six seeds for E0, E2, E3 and E5; three for E4.
- The canary result is for one attacker (an adaptive Neural Cleanse variant).
- V3 assumes the model under test resembles its reference models. For a model
  that has collapsed (accuracy at chance) its score is extreme and means little;
  read V3 together with test accuracy.
- V4 (distance to the oracle in weight space) is not informative here: two
  retrains differ from each other about as much as the original differs from
  either. Prediction-space distance is reported instead.
