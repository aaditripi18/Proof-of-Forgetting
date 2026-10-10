# Archived: first full run (seed 42), 2 local epochs per round

These results come from the first configuration: 30 FedAvg rounds of 2 local
epochs, 40 shadow federations, 7 E4 client sizes. The federated models reached
only ~43% CIFAR-10 test accuracy (the same code reaches 84.9% centrally; see
results/tune/), so they memorise little and every membership signal is weak.
SCRUB collapsed to chance and FedEraser-lite fell to 23% in that run; both were
fixed afterwards (the second e0_sanity CSV here already has the fixes).

Kept for the record and still verifiable against results/ledger.jsonl, but not
used for findings: the runs used from 2026-10-10 on train with 5 local epochs.
