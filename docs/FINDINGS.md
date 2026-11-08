# Findings

**Status: complete.** CIFAR-10 with ResNet-20 on six seeds (42–47) plus two
client swaps; a controlled client-size experiment (the same client cut to
500–3000 samples, three seeds); three longer-trained federations (70 rounds,
76–79% accuracy); a second architecture, VGG-11, on three seeds; and a second
dataset, PathMNIST, on six seeds. Every number here is in
[RESULTS.md](RESULTS.md), which `scripts/report.py` generates from runs whose
CSVs match `results/ledger.jsonl` exactly. Read these as findings for two small
image models and one non-IID level, not as general laws (see the limits at the
end).

Setup: 10 clients (Dirichlet α = 0.5), FedAvg with 5 local epochs. The main
model is ResNet-20 with GroupNorm; the main CIFAR-10 runs train 30 rounds and
reach 61–67% test accuracy (the same code reaches 84.9% centrally); the
longer-trained runs train 70 rounds. VGG-11 is a slim VGG (no skip
connections, GroupNorm, 32–256 channels) trained 30 rounds at learning rate
0.02, chosen by `configs/tune_vgg.yaml`; it reaches 74–77%. Client 0 asks to be
forgotten; on CIFAR-10 it holds 3360, 2511, 1855, 1670, 4593 and 3667 training
samples on seeds 42 to 47. PathMNIST (colon pathology, 9 classes, 28×28) uses
50,000 training images drawn from its training split, ResNet-20 and the main
federation settings, and reaches 79–82%; its forgotten clients hold 1900, 1111,
3423, 793, 1115 and 3672 samples on seeds 42 to 47.

Seeds 42–44 were the original study. Everything else ran later on the same
code, and several results changed as runs were added; each such change is
stated where it happens.

## The question

When a client asks a federated server to forget it, the server usually shows
that a membership-inference test can no longer find the client. We ask what
such tests can and cannot certify, by running each of them against a server
that did nothing at all and against genuine retrains without the client.

## Results

### 1. The standard check cannot tell "did nothing" from "retrained" (46 of 46 runs)

V1, the loss-threshold membership AUC on matched in/out samples, reads between
0.489 and 0.526 for all seven models on all six CIFAR-10 seeds: the server that
did nothing (0.522, 0.510, 0.495, 0.509, 0.520, 0.526), every unlearning
method, and both retrains (0.489–0.504). The same holds everywhere else: on
PathMNIST (0.496–0.517 for `do_nothing`), at 70 rounds (0.523–0.552) and on
VGG-11 (0.525–0.539). Across all 46 E0 and E3 runs, every model lies between
0.476 and 0.552. A server could return the original model and report
"AUC ≈ 0.5, client forgotten".

### 2. LiRA separates them when the client is large, and rarely accuses a genuine retrain

V3, client-level LiRA judged against ten retrains without the client (Student-t
prediction test), on the main CIFAR-10 runs:

| | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | seed 47 |
|---|---|---|---|---|---|---|
| client size | 3360 | 2511 | 1855 | 1670 | 4593 | 3667 |
| `do_nothing` | z 5.5, p 0.0003 | z 5.1, p 0.0004 | z 1.8, p 0.061 | z 0.2, p 0.43 | z 5.3, p 0.0004 | z 6.6, p < 0.0001 |
| `retrain_oracle` | p 0.62 | p 0.52 | p 0.59 | p 0.79 | p 0.67 | p 0.23 |
| `retrain_oracle_alt` | p 0.24 | p 0.78 | p 0.76 | p 0.62 | p 0.67 | p 0.95 |

Across every E0 and E3 run (both datasets, both architectures, all client sizes,
both training lengths: 46 runs) the two retrains were tested 92 times, and V3
flagged two of them at α = 0.05, both on PathMNIST (seed 45 in E0, p 0.030;
seed 44 in E3, p 0.017): 2.2%, below the 5% expected by chance. On CIFAR-10 it
flagged none in 64.

### 3. What decides LiRA's verdict is how much data the client trained with

**Auditing more samples does not help.** `scripts/v3_size_curve.py` reruns V3
on random subsets of m of the client's samples, 200 subsets per size, without
retraining:

| samples audited (m) | 50 | 250 | 500 | 1500 |
|---|---|---|---|---|
| seed 42, median z | 3.3 | 4.8 | 5.2 | 5.5 |
| seed 43, median z | 2.1 | 3.9 | 4.6 | 4.9 |
| seed 44, median z | 1.1 | 1.6 | 1.8 | 1.8 |
| seed 45, median z | 0.2 | 0.2 | 0.2 | 0.2 |
| seed 46, median z | 4.5 | 5.0 | 5.1 | 5.2 |
| seed 47, median z | 4.7 | 6.0 | 6.4 | 6.5 |

On every seed the evidence levels off by 250–500 samples. The retrains were
flagged in at most 6.5% of subsets at any m, and in none once m ≥ 1000.

**Cutting the same client shows that its training size is the cause.** In the
controlled experiment, `data.forget_size` keeps the federation, the seed and
every other client fixed, and trains with only a random subset of the forgotten
client's data (its held-out half is cut in proportion):

| client keeps | 500 | 1000 | 2000 | 3000 | all |
|---|---|---|---|---|---|
| seed 42 (all = 3360) | z 1.4 | z 0.6 | **z 2.3, p 0.028** | **z 5.8** | **z 5.5** |
| seed 46 (all = 4593) | z 0.1 | z 0.7 | z 1.7, p 0.065 | **z 2.8, p 0.012** | **z 5.3** |
| seed 47 (all = 3667) | z −0.6 | z 0.0 | **z 3.6, p 0.004** | **z 4.3** | **z 6.6** |

Bold = detected at α = 0.05 ([figure](figures/controlled_size.svg)). No client
cut to 1000 samples or fewer was detected (0/6); at 2000, two of three were; at
3000 or more, all were (6/6). The retrains were never flagged in these runs.

The natural clients agree on CIFAR-10. Across the eight main runs (six seeds
and two client swaps), every forgotten client with at least 2511 training
samples was detected (5/5, z 4.4–6.6) and every client with at most 1855 was
not (0/3, z 0.2–1.9); swapping which client is forgotten in the same federation
flips the verdict ([figure](figures/client_size.svg)).

PathMNIST is weaker. None of its four small clients (793–1900 samples) was
detected, but of its two large ones only seed 44's was (3423 samples: z 3.3,
p 0.006); seed 47's (3672 samples) was not (z 1.5, p 0.094). So on PathMNIST a
large client is necessary but not sufficient. Across both datasets, no client
with fewer than 2000 training samples was detected by ResNet-20 at 30 rounds
(0/13: three natural CIFAR-10 clients, six cuts, four PathMNIST clients).

One caveat on the cuts: removing up to 4000 samples also shrinks the
federation's training data, and seed 46's accuracy fell from 61% to 54–57%. The
other two seeds lost about 3 points or less.

### 4. A better-trained model is easier to audit, whatever the architecture

The main runs reach 61–67%. Tuning (`configs/tune_strong.yaml`, seed 42) showed
the gap is training budget, not capacity: 90 rounds reached 79.4%, while
doubling ResNet-20's width did nothing (65.3% at 30 rounds). A full run at 90
rounds would exceed Kaggle's 12-hour session limit, so three seeds were rerun
at 70 rounds; and VGG-11, a different architecture family, reaches 74–77% in
the original 30 rounds:

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| client size | 3360 | 2511 | 1855 |
| ResNet-20, 30 rounds: accuracy, V3 z, V5 p | 64.9%, **5.5**, **0.0002** | 63.9%, **5.1**, 0.10 | 67.1%, 1.8, 0.59 |
| ResNet-20, 70 rounds: accuracy, V3 z, V5 p | 77.5%, **8.6**, **0.0001** | 76.3%, **6.5**, **0.0001** | 78.7%, **3.2**, **0.0006** |
| VGG-11, 30 rounds: accuracy, V3 z, V5 p | 74.0%, **7.9**, **0.0001** | 74.5%, **9.1**, **0.0002** | 76.5%, **3.9**, **0.002** |

Bold = detected at α = 0.05. Seed 44's 1855-sample client, too small to detect
with the main model, is detected by both better-trained models, by LiRA and by
the client's own test. Better-trained models memorise more (in tuning, the gap
between the client's training accuracy and test accuracy grew from 4.5 points
at 30 rounds to 12 at 90), which is what both checks detect. No retrain was
flagged in these six runs (V3 p ≥ 0.30, V5 p ≥ 0.11), and V1 stayed blind
(0.523–0.552). So the size threshold in finding 3 is not a constant: it falls
as the model learns more, on either architecture, and part of what looked like
"small clients cannot be audited" was an undertrained model.

### 5. Fine-tuning on the remaining clients is unreliable as forgetting

`finetune_retain` keeps its accuracy and was detected by V3 on seeds 42 and 47
(p 0.005, 0.0008) but not on the other four. Gradient ascent with recovery
rounds costs under 4 points of accuracy and was detected on seed 47 only
(p 0.003). SCRUB was detected on seeds 46 and 47 (p 0.03, 0.02).
FedEraser-lite was never detected, but loses 11–22 points; a model that
"forgets" by getting worse at everything is not the oracle.

With three seeds, gradient ascent and SCRUB looked undetected everywhere. On
seed 47, where the model memorised its client most strongly (z 6.6), every
method that kept its accuracy was still detected.

### 6. The backdoor canary usually outlasts the other signals (10 of 14 runs)

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
| seed 42, 70 rounds | 32 | 32 | 48 (accuracy at chance) |
| seed 43, 70 rounds | 12 | 16 | 24 |
| seed 44, 70 rounds | 28 | 36 | 40 (accuracy 10%) |
| PathMNIST 42 | 0 (never detected) | never | **24** |
| PathMNIST 43 | 0 (never detected) | never | 40 (accuracy 14%) |
| PathMNIST 44 | 0 (never detected) | 32 | 60 (accuracy 12%) |
| PathMNIST 46 | 0 (never detected) | 0 (never detected) | never |
| PathMNIST 47 | 0 (never detected) | 48 | 52 (accuracy 9%) |

The canary went last on 10 of 14 runs, the opposite of H2. On CIFAR-10 seed 46
and PathMNIST seed 43 LiRA outlasted it, but only into a model near chance
accuracy, where V3 means little (see limits). Only on CIFAR-10 seed 47 and
PathMNIST seed 42 did the canary go first, as H2 predicted, while the model was
still accurate (58% and 72%). On PathMNIST seed 45 the canary could not be
planted at all: with triggered copies of 10% and of 25% of the client's data,
the trigger sent inputs to the target class no more often than they went there
without it (1.0% against 2.1%, and 2.2% against 2.2%), so E2 and E3 did not
run for that seed (pilot logs `results/logs/e2_canary-20261104-085014-b32caa.log`
and `e2_canary-20261104-090738-851c1f.log`).

### 7. A dishonest server beat each check at least once; none caught it everywhere

`canary_suppressor` never unlearns. It reconstructs triggers with Neural Cleanse
on its own data and tries to erase the backdoor (H3).

| | NC found the class | V2 (canary) | V3 (LiRA) | V5 (own data) |
|---|---|---|---|---|
| seed 42 | no | p < 0.0001 | p < 0.0001 | p 0.048 |
| seed 43 | no | p 0.0003 | p < 0.0001 | p 0.27 |
| seed 44 | no | **fooled, p 0.97** | p 0.021 | p 0.30 |
| seed 45 | no | p < 0.0001 | p 0.018 | p 0.14 |
| seed 46 | **yes** | **fooled, p 1.00** | **fooled, p 0.15** | p 0.0003 |
| seed 47 | no | p 0.032 | p < 0.0001 | p 0.013 |
| seed 42, 70 rounds | no | p < 0.0001 | p < 0.0001 | p 0.0001 |
| seed 43, 70 rounds | no | **fooled, p 0.79** | p 0.0016 | p 0.045 |
| seed 44, 70 rounds | **yes** | p < 0.0001 | p < 0.0001 | p 0.055 |
| PathMNIST 42 | no | p < 0.0001 † | **fooled, p 0.59** | p 0.15 |
| PathMNIST 43 | no | p < 0.0001 † | **fooled, p 0.16** | p 0.37 |
| PathMNIST 44 | no | p < 0.0001 † | p < 0.0001 | p 0.038 |
| PathMNIST 46 | no | p < 0.0001 † | p 0.0022 | p 0.66 |
| PathMNIST 47 | **yes** | p < 0.0001 † | **fooled, p 0.45** | p 0.72 |

† On PathMNIST the canary check also fires on genuine retrains (see below), so
here it does not distinguish the dishonest server from an honest one.

The canary check would have certified a server that forgot nothing on three
CIFAR-10 runs. On seed 44 the canary was weak to begin with (trigger success
9% on `do_nothing`, against 90% on seed 42). On seed 46, the one 30-round run
where Neural Cleanse found the client's target class, the suppressor also got
past LiRA, at a cost of 7 points of accuracy. At 70 rounds LiRA caught the
suppressor on all three seeds, including the one where Neural Cleanse found the
class; the better-trained model memorised the client too strongly to hide. On
PathMNIST the suppressor got past LiRA on three of five seeds while keeping its
accuracy (80.2–83.6%, against 80.0–81.4% for `do_nothing`); on those three, only
the canary still fired, and the canary fires on honest servers too.

The canary check is not safe to trust when it fires. It flagged a genuine
retrain 3 times in 18 on CIFAR-10 (all at 30 rounds) and 8 times in 10 on
PathMNIST (11/28 in all), so on PathMNIST it flags the honest server about as
often as the dishonest one.

### 8. A client can verify on its own data only if it contributed a lot of it

V5 needs nothing but the client's own data: it compares the model's confidence
on the half the client trained with and the half it held back. On the main
CIFAR-10 runs it detected `do_nothing` on seeds 42, 46 and 47 (p 0.0002, 0.0002,
< 0.0001) and not on 43, 44 or 45 (0.10, 0.59, 0.10). Including the client
swaps, it detected every forgotten client with at least 3345 training samples
(4/4) and none with 2511 or fewer (0/4). In the controlled cuts it did better:
it detected every client cut to 2000 or more (6/6) and one of six cut to 1000
or fewer. It detected all three clients at 70 rounds and all three on VGG-11,
and none of the six on PathMNIST. So size matters, but the dataset and how well
the model is trained matter too.

V5 flagged a genuine retrain once in 64 E0 tests (PathMNIST seed 42, p 0.011).
In E4, which subsamples the client's halves to a fixed size, it detected
`do_nothing` at some sizes on seeds 42, 45 and 47 and at none on 43, 44 and 46,
and flagged the retrain twice in 28 tests: at the sizes tested, it is not a
reliable check.

## What this adds up to

- The check most often reported for unlearning, loss-threshold membership AUC,
  cannot tell a server that forgot from one that did nothing: on both datasets,
  both architectures and both training lengths (46 of 46 runs).
- Client-level LiRA can, with 2 false alarms in 92 tests on genuine retrains,
  but only for clients that contributed enough data. Cutting one client while
  holding everything else fixed shows the size is the cause: no client cut to
  1000 samples or fewer was detected, every client with 3000 or more was. The
  threshold is not fixed: better-trained models (ResNet-20 at 70 rounds, or
  VGG-11) made a 1855-sample client detectable. On PathMNIST even a large client
  was sometimes missed. LiRA needs models retrained without the client, which
  only an auditor with compute can produce, and auditing more samples does not
  help past a few hundred.
- Backdoor canaries usually outlast other signals under gradient ascent, but
  they could not be planted at all for one client, a server that never
  unlearned got past them on 3 of 14 runs, and they accused a genuine retrain
  11 times in 28.
- A dishonest server that suppresses the canary got past LiRA on 4 of 14 runs,
  and never at 70 rounds. No single check caught it on every run.
- A client can check forgetting from its own data alone only for large clients,
  well-trained models, or both.

## Design changes made after seeing data

Each was made for a stated reason and is recorded in the commit history.
Seeds 43–47 and every later run ran after all of them, so they test whether
the changes hold.

| Change | Why | Evidence it rests on |
|---|---|---|
| 5 local epochs instead of 2 | 2 gave ~43% test accuracy and almost no memorisation | tuning run, seed 42 |
| V5 compares logit-scaled confidence, not loss | mean loss is heavy-tailed and flagged a retrain oracle whose accuracy on both halves was identical | seed 42 only; chosen post hoc |
| V3's null is ten retrains without the client, with a Student-t prediction test | with a 4-shadow null of smaller federations, a second retrain oracle scored z = 4.3 | seed 42; held on every later run (2 false alarms in 92) |
| SCRUB max-steps capped; FedEraser replays every round | uncapped SCRUB fell to chance; FedEraser to 23% | first full run |
| 70 rounds for the longer-trained runs | 90 rounds reached 79.4%, but a full run would exceed Kaggle's 12-hour limit | tuning run, seed 42 (`configs/tune_strong.yaml`) |
| VGG-11 at width 32 and learning rate 0.02 | the best of four settings at 30 rounds (74.0%, against 64.2–71.4%) | tuning run, seed 42 (`configs/tune_vgg.yaml`) |

V5 was chosen post hoc on seed 42. It then failed to detect `do_nothing` on
seeds 43, 44 and 45 and on every PathMNIST seed, so read it as a check that
works only for large clients or well-trained models.

## Limits

- Two small image models (ResNet-20 and a slim VGG-11) and one non-IID level
  (α = 0.5); two datasets.
- Most results are at 61–67% accuracy on CIFAR-10. The 70-round runs (E0, E2, E3,
  E5) and VGG-11 (E0, E5) show the main conclusions hold at 74–79%; neither ran
  E4.
- Client-size evidence: eight natural forgotten clients and twelve controlled
  cuts on CIFAR-10, six clients on PathMNIST. The thresholds (about 2000–2500
  samples for LiRA with ResNet-20 at 30 rounds here) depend on the setup and
  should not be read as constants.
- Six seeds for E0, E2, E3, E4 and E5 on CIFAR-10; six for E0 and E5 and five
  for E2 and E3 on PathMNIST (the canary could not be planted on seed 45).
- The canary result is for one attacker (an adaptive Neural Cleanse variant).
- V3 assumes the model under test resembles its reference models. For a model
  that has collapsed (accuracy at chance) its score is extreme and means little;
  read V3 together with test accuracy.
- V4 (distance to the oracle in weight space) is not informative here: two
  retrains differ from each other about as much as the original differs from
  either. Prediction-space distance is reported instead.
