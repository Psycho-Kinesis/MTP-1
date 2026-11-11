# E5 pre-registration: confirmation of the improved base model on a new fresh test set

Written 9 November 2026, **after** E5 development (`results/e5_base_model`, `results/e5_base_model_v2`; pilot train/validation only) and **before** any design of the new fresh set exists. `experiments/run_e5_confirm.py` asserts the sha256 of this file.

## Why a confirmation is needed
In E5 development, the pre-declared rule v1 (choose the label-free base by its own score against TSDT) selected the old base P0. That rule rewards a base whose approximation error happens to cancel the FSDT→TSDT bias. Rule v2 (choose the base by the downstream M-S score at N = 10, 20, 40) was declared **after** seeing that outcome, and selected P2 (classical-plate prefactor, linear b/a). Because rule v2 was chosen with knowledge of rule v1's outcome, and validation was used for selection, the large validation gains must be confirmed on designs that played no role in any choice. The E4 fresh set is spent; a new one is used.

## Frozen (no changes allowed after this file)
- **Arms:** refitted exactly as in `run_e5.py` (`fit_all`), from the pilot only, and asserted to reproduce `results/e5_base_model_v2` validation metrics.

| Arm | What it is |
|---|---|
| P0 | old label-free base (= E3c/E4 physics only) |
| **P2** | new label-free base: FSDT energy, classical-plate prefactor, linear b/a coordinates |
| D | data only, random TSDT labels |
| M-S@P0 | old pipeline (= E3c/E4 primary M-S) |
| **M-S@P2** | **new primary pipeline:** P2 + multiplicative (k h)² correction, shear-guided labels |
| M-R@P2 | P2 + the same correction, random labels |

- **Shared settings:** label budgets N ∈ {10, 20, 40, 70, 140}; seeds 11/22/33; collocation as before. The target is TSDT; the metric is the per-design relative L2 over the five scaled fields (w-only also reported).

## New fresh test set
- **200 designs**, generator **pilot-1.1**, seed **20261215**. Never used before: the pilot used 20261109/20261110, E4 used 20261201, collocation used 1011/1022/1033, and label subsets used 11/22/33.
- Full pilot domain, smooth-mixture loads, TSDT labels (FSDT computed only as the model-form reference).
- **All 200 are test designs**, asserted disjoint from the pilot designs, all collocation designs and the 200 E4 fresh designs.

## Confirmatory questions (primary arm M-S@P2)
Statistics as in E4: seed-averaged per-design errors for paired tests; the mean over seeds of per-seed median and p95 for targets; 95% percentile bootstrap, 10,000 resamples.

| ID | Question | Confirmed if |
|---|---|---|
| C1 | M-S@P2 beats its own physics-only base P2 | CI of the median paired difference (P2 − M-S@P2) above 0 at each of N = 10, 20, 40 |
| C2 | M-S@P2 beats data only D | CI of (D − M-S@P2) above 0 at each of N = 10, 20, 40 |
| C3 | Median target with 10 labels | median ≤ 1% at N = 10 |
| C4 | Tail target with 10 labels | p95 ≤ 3% at N = 10 |
| C5 | Tail target **robustly**: the E4 weakness | p95 ≤ 3% at **every** N ∈ {10, 20, 40, 70}, **and** the upper end of the bootstrap CI of the seed-averaged p95 at N = 10 is ≤ 3% |
| C6 | The new base improves the pipeline | CI of (M-S@P0 − M-S@P2) above 0 at each of N = 10, 20, 40 |

**Secondary (descriptive):** all arms at all N; P2 vs P0 accuracy against exact FSDT; M-S@P2 vs D at N = 140; the worst designs; w-only errors; comparison with the E4 results.

## Reporting commitments
- All C1–C6 outcomes are reported, confirmed or not. No re-runs with changed settings on this fresh set; any change needs another new fresh set.
- The interpretation will state that the FSDT→TSDT gap is unusually well structured, H_TSDT ≈ H_FSDT·(1 + c (k h)²), and that a real high-fidelity gap (3D FE, experiments) may not be.
