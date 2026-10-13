# Findings so far

**Status: preliminary.** Seed 42 is complete under the final design. Seed 43 has
run only E4 and the verifier columns that do not depend on LiRA; seed 44 has not
run. Nothing here is a conclusion until all three seeds are in. Every number is
in [RESULTS.md](RESULTS.md), which is generated from ledger-verified runs, with
one exception: seed 43's V1 and V5 values for `do_nothing` (0.510 and p = 0.105)
come from its archived run `e0_sanity-20261011-175032-a5e72b`
(`results/archive/2026-10-11_balanced-null`), whose V3 column predates the
current design and is not used. That run will be replaced when seed 43 is rerun.

Setup: CIFAR-10, ResNet-20 with GroupNorm, 10 clients (Dirichlet α = 0.5),
FedAvg for 30 rounds of 5 local epochs; the federated models reach about 65%
test accuracy (the same code reaches 84.9% centrally). One client asks to be
forgotten.

## The question

When a client asks a federated server to forget it, the server usually shows
that a membership-inference test can no longer find the client. We ask what
such tests can and cannot certify, by running each of them against a server
that did nothing at all and against a genuine retrain without the client.

## What seed 42 shows

### 1. The standard check does not separate "did nothing" from "retrained"

V1, the loss-threshold membership AUC, reads between 0.50 and 0.52 for every
model: the server that did nothing (0.522), every unlearning method, and both
retrains (0.501, 0.497). Seed 43 agrees (0.510 for `do_nothing`). A server could
return the original model and report "AUC ≈ 0.5, client forgotten".

The same models are separated by V3, client-level LiRA judged against ten
retrains without the client: `do_nothing` is detected (p = 0.0003), both
retrains are not (p = 0.62, 0.24).

### 2. Fine-tuning on the remaining clients does not forget

`finetune_retain` keeps its accuracy and is still detected by V3 (p = 0.005).
Gradient ascent with recovery rounds is not detected (p = 0.50), at a cost of
about two points of accuracy. SCRUB and FedEraser-lite are not detected either,
but lose 9 and 22 points of accuracy; a model that forgets by getting worse at
everything is a different thing from the oracle, which V4 also shows.

### 3. The backdoor canary is the last signal to go, not the first

We expected a client's backdoor canary to be forgotten before its ordinary
influence (H2). Following gradient ascent step by step shows the opposite order:
the client-side test loses the client first (step 20), LiRA next (step 28), and
the canary last (step 44), by which point the model's accuracy is at chance.
Gradient ascent cannot remove the backdoor without destroying the model.

### 4. A dishonest server did not get past the canary check

`canary_suppressor` never unlearns; it reconstructs triggers with Neural Cleanse
on its own data and tries to erase the backdoor (H3). It picked the wrong class,
and although the trigger's success rate fell from 90% to 26%, plain fine-tuning
produces the same drop (24%), and the canary test still detected the client
overwhelmingly. Canary verification was harder to cheat than we assumed, at
least for this attacker.

### 5. A client on its own can verify very little

V5 needs nothing but the client's own data: it compares the model's confidence
on the half the client trained with and the half it held back. On seed 42 it
detects `do_nothing` at full client size but only at the largest size tested in
E4 (2500 samples per half). On seed 43 it does not detect `do_nothing` at all
(p = 0.105; E4: no detection at any size). The verifier that works, V3, needs
models retrained without the client, which a client cannot produce.

## What this adds up to (if it holds)

- The check most often reported for unlearning, loss-threshold membership AUC,
  cannot tell a server that forgot from one that did nothing.
- Checks that can tell them apart exist, but each has a price: LiRA needs
  retrains without the client; canaries need the client to poison its own data
  in advance and detect the backdoor rather than the client's ordinary data.
- A client cannot verify forgetting from its own data alone at the client sizes
  and model quality studied here.

## Design changes made after seeing data

Each was made for a stated reason and is recorded in the commit history; the
later seeds are the test of whether they hold.

| Change | Why | Evidence it rests on |
|---|---|---|
| 5 local epochs instead of 2 | 2 gave ~43% test accuracy and almost no memorisation | tuning run, seed 42 |
| V5 compares logit-scaled confidence, not loss | mean loss is heavy-tailed and flagged a retrain oracle whose accuracy on both halves was identical | seed 42 only; chosen post hoc |
| V3's null is ten retrains without the client, with a Student-t prediction test | with a 4-shadow null of smaller federations, a second retrain oracle scored z = 4.3 | seed 42; calibration of the t-test is unit-tested |
| SCRUB max-steps capped; FedEraser replays every round | uncapped SCRUB fell to chance; FedEraser to 23% | first full run |

## What is not yet established

- Anything across seeds: V3's results exist for seed 42 only.
- Whether V3 keeps leaving retrains alone on seeds 43 and 44, which it was not
  tuned on. This is the main thing the remaining runs test.
- Generality: one dataset, one architecture, one non-IID level, one forget
  client per seed, 65% test accuracy.
- The canary result for stronger attackers than our Neural Cleanse variant.

## Known limitations of the verifiers

- V3 assumes the model under test resembles its reference models. For a model
  that has collapsed (accuracy at chance) its score is extreme and means little;
  read V3 together with test accuracy.
- V4 (distance to the oracle in weight space) is not informative here: two
  retrains differ from each other about as much as the original differs from
  either. Prediction-space distance is reported instead.
