# Archived: seed 42 with 5 local epochs, V5 still loss-based

First run of the 65%-accuracy configuration (E0, E5, E4 for seed 42). V5 here
compares mean cross-entropy loss, which flagged the retrain oracle (p = 0.03
in E0, 0.0007 on one E4 subset) because of a few very-high-loss images in the
held-out half; accuracy on the two halves was identical. V5 was then switched
to the logit-scaled confidence (pof.evaluate.v5_scores). e0_sanity_v5/ holds
the new V5 recomputed from this run's saved per-sample outputs.

Superseded by the rerun of seed 42 with the new V5, so that every seed reports
the same statistic. Still verifiable against results/ledger.jsonl.
