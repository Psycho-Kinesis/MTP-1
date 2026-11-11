# H2: generalization to unseen load shapes (executed)

9 November 2026 · Pre-registration: [`H2_PREREGISTRATION.md`](H2_PREREGISTRATION.md), written before any H2 test design existed (sha256 `5b8e0ef2…`, asserted). Results: `results/h2_loads/`. Presentation figure: `figures/h2_load_families_summary.png` (`experiments/plot_h2.py`, drawn from the saved manifest).
Command: `python experiments/run_h2.py --output <new dir>` (about 4 min). An identical rerun reproduces every number and the test set bit for bit.

## Question
Every model up to E5 was trained and tested on one load family: smooth signed mixtures of the 25 modes m, n ≤ 5. Does the confirmed E5 pipeline stay accurate for **load shapes it has never seen**, including loads with content **outside its 5×5 modal basis**, without new high-fidelity designs?

## Why modal transfer functions help
The pipeline learns, for each mode, how the plate responds to a unit load in that mode. For linear plates the response to *any* load is the sum of these per-mode responses. So any load whose content lies in the trained modes is handled without retraining. Content in **untrained modes** is the only real gap.

Plates filter loads strongly (response ∝ load/k⁴). On training designs, even a narrow patch had only 0.8–2.7% of its response outside 5×5 and ≤ 0.04% outside 15×15. So extending the basis to 15×15 should close the gap. The **same 10 labelled designs** can be reused for the extra modes, because one high-fidelity solution of a design contains all its modes.

## Protocol, as pre-registered
1. **Refit the frozen E5 arms** and assert that every arm, budget and seed reproduces the E5 development validation metrics (147 runs).
2. **Extend** the label-free base (P-ext), the correction (C) and data only (D-ext) to 15×15 modes, with every hyperparameter **frozen from E5** and the **same labelled designs**. Assert that their 5×5 parts reproduce the frozen arms exactly, on all test designs.
3. **Only then generate 200 fresh designs** (pilot-1.1, seed 20261231), disjoint from the pilot, all collocation designs and the E4/E5 fresh sets.
   - Each design gets four loads: L1 training-type mixture (control), L2 uniform pressure, L3 wide Gaussian patch (σ 0.10–0.20), and L4 narrow Gaussian patch (σ 0.04–0.07).
   - The truth is TSDT with modes m, n ≤ 25.
4. **Metric:** the exact continuous relative L2 of the five scaled fields, computed from modal coefficients by orthogonality, so missing modes count fully. Evaluated once.

## Pre-registered questions (primary arm C, 10 labels): all confirmed

| ID | Question | Result | Verdict |
|---|---|---|---|
| Q1 | Control: C meets the targets on L1 | median 0.065%, p95 0.37% | **Confirmed** |
| Q2 | C meets median ≤ 1% and p95 ≤ 3% on **each** new family | L2 0.061 / 0.33; L3 0.062 / 0.34; **L4 0.070 / 0.40** | **Confirmed** |
| Q3 | C beats data only (D-ext) on each new family | +18.8, +19.0, +19.1 points; better on 200/200 designs each | **Confirmed** |
| Q4 | C beats physics only (P-ext) on each new family | +0.20 [+0.14, +0.36], +0.24 [+0.17, +0.40], +0.31 [+0.22, +0.53] points; better on 179, 181 and 190 of 200 | **Confirmed** |
| Q5 | Extending the basis matters on the narrow patch | A − C = +1.35 [+1.26, +1.49] points; better on **200/200** designs | **Confirmed** |

## All arms (median % / p95 %, 10 labels unless stated; mean over 3 seeds)

| Arm | L1 mixture | L2 uniform | L3 wide patch | L4 narrow patch |
|---|---|---|---|---|
| Physics only P-ext (0 labels) | 0.39 / 5.43 | 0.31 / 4.18 | 0.33 / 5.47 | 0.41 / 6.68 |
| Data only D-ext | 16.6 / 81.9 | 15.7 / 78.9 | 16.1 / 78.2 | 16.8 / 79.1 |
| A: frozen E5 pipeline (5×5) | 0.065 / 0.37 | 0.24 / 0.47 | 0.12 / 0.43 | 1.47 / **3.14** |
| B: A + physics only outside 5×5 | 0.065 / 0.37 | 0.073 / 0.44 | 0.065 / 0.35 | 0.13 / 1.63 |
| **C: E5 pipeline extended to 15×15** | **0.065 / 0.37** | **0.061 / 0.33** | **0.062 / 0.34** | **0.070 / 0.40** |
| *Reference: TSDT truncated to 5×5* | *0* | *0.22 / 0.38* | *0.071 / 0.31* | *1.43 / 3.09* |
| *Reference: exact FSDT* | *0.39 / 5.44* | *0.29 / 4.19* | *0.33 / 5.48* | *0.38 / 6.68* |

With 40 and 140 labels, C improves to 0.033–0.041% median and 0.14–0.18% p95 on every family. Data only (D-ext) needs 140 labels to reach 0.06–0.07% median and 0.82–0.87% p95. The frozen 5×5 data-only model stays at the truncation floor on L4 even with 140 labels (1.47% / 3.26%).

## What H2 establishes
1. **The pipeline generalizes to unseen load shapes.** Uniform pressure and Gaussian patches give essentially the same accuracy as the training-type loads (median about 0.06–0.07%, p95 0.33–0.40%), with the same 10 high-fidelity designs.
2. **The only failure mode is untrained modes, and it is predictable.** The frozen 5×5 pipeline (A) sits exactly at its truncation floor: 1.47% vs 1.43% median and 3.14% vs 3.09% p95 on the narrow patch. Its error is therefore truncation, not learning. It narrowly misses the 3% tail target on L4 only.
3. **The fix needs no new high-fidelity designs.** Extending the label-free base and the correction to 15×15 modes, using the extra modes already present in the 10 labelled solutions, removes the floor (L4: 0.070% / 0.40%).
4. **Correcting the extra modes matters too.** B, which uses physics only outside 5×5, reaches 1.63% p95 on L4, against 0.40% for C. The FSDT→TSDT correction is needed in the higher modes as well, especially for thick plates.
5. **Narrow loads on thick plates.** For 87 of the 200 L4 designs the patch is narrower than one plate thickness, where plate theories themselves are questionable. C still matches TSDT to a median of 0.15% and a p95 of 0.65%, while exact FSDT differs from TSDT by 2.1% (median) there. The worst design (2.0%) is a very thick, narrow plate (a/h 6.5, b/a 0.57). This shows agreement with the TSDT reference, **not** physical accuracy: no plate theory is reliable for loads narrower than the thickness.

## Caveats
- **TSDT is the reference throughout;** it is a plate theory, not 3D elasticity. For loads narrower than the plate thickness both theories are outside their domain of validity.
- **All four families are smooth or moderately concentrated pressures.** Point loads, line loads and loads with discontinuities would need many more modes, and the per-mode error budget would need re-checking.
- **The extension to 15×15 reuses E5's hyperparameters unchanged.** That is conservative (nothing was tuned for the new modes), and it worked; tuning might help further but was deliberately not done.

## Status
| Completed and verified | Not done |
|---|---|
| H2: four load families, 200 fresh designs, Q1–Q5 confirmed; extension to 15×15 asserted identical in the 5×5 part; deterministic rerun | Point or line loads; 3D/FE reference; other boundary conditions |
| `experiments/plot_h2.py` presentation figure | Inverse design demonstration; thesis write-up |

## Recommended next steps
1. **Inverse design demonstration:** minimum mass under deflection and frequency limits, using the extended pipeline, verified with the TSDT solver.
2. **MTP report and presentation:** E2 → E5 + H2 now form a complete, pre-registered argument.
