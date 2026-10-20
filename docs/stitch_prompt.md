# Stitch prompts for the Proof of Forgetting project page

Use Stitch in Web mode. Paste prompt 1 and generate. Then paste prompts 2–5 one
at a time as refinements of the same design. Export the HTML/Tailwind code at
the end and upload it to the session. Numbers get wired to `site/data.json`
(built from the ledger-verified results) afterwards, so don't worry if Stitch
rounds or tweaks a figure; any number or claim Stitch invents is removed.

---

## Prompt 1: the page (paste first)

A one-page website for "Proof of Forgetting", a research project that audits how AI systems prove they have deleted someone's data. In federated learning, many devices train one shared model; when one of them asks to be forgotten, the server claims it "unlearned" them and shows a test as proof. This project shows that the most common test would also pass a server that did nothing.

Feel: an editorial research report crossed with a lab notebook. Calm, precise, slightly forensic. Think Stripe Press typography meets a data-journalism investigation. No startup clichés, no gradients, no 3D blobs, no stock imagery, no brains or glowing circuits.

Look: cool off-white background (#F8F9FA), ink-black text, one signal colour, coral #E8590C, used only where something is "still remembered". A secondary muted teal #0C8599 for "genuinely forgotten". Inter for prose, JetBrains Mono for numbers, method names and code. Thin hairline rules, lots of whitespace, 1100px content width, a dark mode.

Hero, the centrepiece: on the left, a large headline "Can a server prove it forgot you?" and a short line under it: "Federated unlearning is judged by tests that ask whether a client can still be detected. I ran those tests against a server that did nothing at all." Two buttons: "See what I found" and "Reproduce it". On the right, a custom illustration built from UI elements: two identical-looking model cards side by side, one labelled "Retrained without you" (teal tick) and one labelled "Did nothing" (coral dot). Under both, the same grey stamp reading "AUC 0.50 · FORGOTTEN". Below the pair, a thin mono line: "The usual check gives both the same verdict."

Below the hero, a slim row of three stats separated by hairlines: "3 seeds · 7 unlearning methods · 5 verifiers", "0.49–0.52 AUC for every model, including do-nothing", "0 false alarms on 12 genuine retrains".

Then sections, each with a small mono section number (01, 02, …) and a short title: The problem · How the audit works · The usual check is blind · What does work · A dishonest server · Which signal survives longest · How much data an audit needs · What I learned · Reproduce · Limitations. Sticky top bar with the "proof-of-forgetting" wordmark in mono and links Findings, Method, Explorer, GitHub.

---

## Prompt 2: the evidence sections (paste second)

Design "The usual check is blind" and "What does work" as one paired section: two side-by-side dot-plot panels with a shared row axis. Rows (mono labels): do_nothing, finetune_retain, gradient_ascent, scrub, federaser_lite, retrain_oracle, retrain_oracle_alt. Each row has three small dots (one per seed: circle, square, triangle). Left panel titled "V1 · membership AUC (the usual check)", x-axis 0.45 to 0.55 with a dashed line at 0.50; every dot sits between 0.49 and 0.52, all bunched on the line. Right panel titled "V3 · client-level LiRA", x-axis −9 to 7, with a soft coral band to the right of 1.9 labelled "still detected". The do_nothing dots sit at 5.5, 5.1 and 1.8; the retrain rows sit near 0. Above the panels a one-line takeaway in large text: "Same models. One test can't tell them apart; the other can."

Design "How the audit works" as a horizontal flow of five small cards, one per verifier, each with a mono chip, who runs it, and one line: V1 · anyone · "Does the model's loss give the client away?"; V2 · client · "Does my secret backdoor trigger still work?"; V3 · auditor · "Does the model look like ones trained with me, or without me?"; V4 · server · "How far is the model from a true retrain?"; V5 · client · "Is the model more confident on the data I trained with than the data I held back?". Under it, a line of seven method chips, with "do_nothing" outlined in coral and "retrain_oracle" outlined in teal, and "canary_suppressor" with a small mask icon labelled "dishonest".

---

## Prompt 3: the adversary and the race (paste third)

Design "A dishonest server" as a three-column comparison, one column per seed (42, 43, 44). Each column has two rows with a verdict pill: "Canary check" and "LiRA". Seed 42: Canary "Caught" (coral pill), LiRA "Caught". Seed 43: Canary "Caught, barely", LiRA "Caught". Seed 44: Canary "Fooled" (grey pill with a small warning icon), LiRA "Caught". A one-line caption above: "A server that never unlearned erased the backdoor trigger on one seed and would have been certified. LiRA caught it on all three."

Design "Which signal survives longest" as three small line charts side by side (one per seed), x-axis "gradient-ascent step" 0 to 64, y-axis "evidence the client is still in", three thin lines: V5 client's own data (blue), V3 LiRA (coral), V2 backdoor canary (teal), with a dashed threshold line. In every panel the blue line falls first, the coral next, and the teal last. Above: "We expected the canary to go first. It went last, every time."

Design "How much data an audit needs" as one line chart: x-axis "samples audited" (50, 100, 250, 500, 1000, 1500, log scale), y-axis "evidence (z)", three lines that rise and then flatten: seed 42 from 3.3 to 5.5, seed 43 from 2.1 to 4.9, seed 44 from 1.1 to 1.8, with a horizontal threshold at 1.9. Caption: "Evidence levels off by ~500 samples. Past that, what matters is how much the model memorised you, not how much you check."

---

## Prompt 4: findings, trust and footer (paste fourth)

Design "What I learned" as a 2×2 grid of editorial cards, each with a bold headline and two sentences:
"The usual check is blind." Membership AUC reads about 0.50 for every model, including a server that did nothing. A server could return the original model and call it forgotten.
"Strong checks have a price." LiRA works with no false alarms, but needs an auditor who can train models without the client.
"Canaries can be cheated." They outlast every other signal, but a server that never unlearned got past them once.
"A client can't check alone." Using only its own data, a client detected the do-nothing server on one seed in three.

Design a "Built to be checked" strip: three large mono figures with captions: "Every number traced to an append-only ledger", "94 tests, including bitwise-deterministic training", "Design changes made after seeing data: listed, with why".

Design "Reproduce" as a dark terminal-style code block with a copy button, containing:
pip install -r requirements.txt
python scripts/run_e0.py --config configs/smoke.yaml
python scripts/report.py
python scripts/figures.py
with a caption "Experiments ran on free Kaggle T4 GPUs · CIFAR-10 · ResNet-20 · 10 clients".

Design "Limitations" as a quiet, muted list of five short lines, deliberately understated.

Footer: the mono wordmark, the line "Audit the proof, not just the method.", and links GitHub, Findings, Results.

---

## Prompt 5: the explorer page (a second screen, same design system)

Add a second page in the same style: "Be the auditor". A visitor picks a model and sees what each test would say about it. Everything comes from saved results; nothing is computed live.

Top: title "Be the auditor" and a line: "Pick what the server did. See which tests would certify it as forgotten."

Controls row: a segmented control for the seed (42, 43, 44), and a list of seven selectable method cards (do_nothing, finetune_retain, gradient_ascent, scrub, federaser_lite, retrain_oracle, canary_suppressor), each with a one-line description and its test accuracy in mono.

Result panel: a large verdict line, e.g. "The usual check says: forgotten. LiRA says: still in." Below it, five verifier rows (V1–V5), each with the verifier name, the number in large mono (AUC or p-value), and a pill "Certifies forgotten" (teal) or "Still detected" (coral). When the selected method is do_nothing and a row says "Certifies forgotten", that row gets a small coral note: "This test would certify a server that did nothing."

A second panel, "Watch gradient ascent forget": a slider from step 0 to 64. As it moves, three horizontal bars (V5, V3, V2) shrink to show the evidence left, each turning grey with the label "lost" when it crosses the threshold, plus the model's test accuracy in mono that falls as the slider moves right.

Keep the same typography, coral/teal accents, hairline rules and dark mode as the main page. On phones the method cards become a dropdown and the panels stack.
