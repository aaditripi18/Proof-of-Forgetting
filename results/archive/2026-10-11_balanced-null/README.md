# Archived: seed 42, V3 judged against a 4-shadow null

E0, E5, E2 and E3 for seed 42 with 20 *balanced* shadow federations (each
leaving out 2 of 10 clients). V3's null for the forget client then rested on
only 4 shadows, each one client smaller than the retrain oracle, and its
p-value came from a normal table. In the canary setup the second retrain oracle
scored z = 4.3 against that null: a false alarm.

Everything not involving V3 in these files is unaffected (V1, V2, V4, V5, the
E2 trajectory's V2/V5 columns). Superseded by reruns with the targeted design:
ten retrains without the client as the null, and a Student-t prediction test.
Still verifiable against results/ledger.jsonl. E4 for seed 42 stays in
results/e4_power (it does not use V3).
