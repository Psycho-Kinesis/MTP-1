# E4 pre-registration: confirmation on a fresh, locked test set

Written 9 November 2026, **before** any fresh design or label exists. `experiments/run_e4.py` records the sha256 of this file in its manifest. If this file is edited after the run, the mismatch is visible.

## Purpose
Every result so far was evaluated on the pilot's 30-design test partition, which earlier experiments had already viewed (scalar baseline, E2, E3a, E3b, E3c). E4 re-evaluates the **frozen** E3c configurations on designs that no model, choice or person has seen. The aim is thesis-grade numbers, and to settle whether the 3% tail target holds; E3c met it on validation but narrowly missed it on the pilot test set.

## What is frozen (no changes allowed after this file)
- **Models:** the E3c arms P, D, G-R, A-R, A-S, M-R and M-S, for label budgets N ∈ {10, 20, 40, 70, 140} and seeds {11, 22, 33}.
  - They are refitted **exactly as in E3c**: pilot training labels, FSDT collocation designs (scrambled Halton, seeds 1011/1022/1033, M = 560), the same grids, the tail-aware selection on the **pilot validation** set, and the per-field ridge.
  - The script asserts that every refitted arm reproduces E3c's validation metrics. Nothing is fitted, selected or tuned on the fresh designs.
- **Target model:** Reddy TSDT (`fgplate/tsdt.py`). The physics loss is FSDT. The metric is the per-design relative L2 over the five scaled generalized fields (w-only also reported).

## Fresh test set
- **200 designs** from generator **pilot-1.1** (platform-stable IDs, parameters quantized to 10 significant digits), **seed 20261201**. This seed has never been used: the pilot used 20261109 (and 20261110 for its split), collocation used 1011/1022/1033, and label subsets used 11/22/33.
- **Domain as before:** g ∈ [0, 8], a/h ∈ [5, 100], b/a ∈ [0.5, 2], transformed Latin hypercube. **All 200 are test designs**; the generator's own split column is ignored.
- **Loads:** the smooth-mixture family of the pilot (25 signed sine modes, unit norm), seeded by (seed, sample index).
- **Labels:** TSDT transfer functions for the 25 modes. The FSDT transfer is computed only as the model-form reference.
- **Asserted:** every fresh design differs from all 200 pilot designs and all collocation designs (minimum coordinate distance > 1e-6).

## Pre-registered confirmatory questions (primary arm M-S)
Statistics follow E3c: per-design errors are averaged over the 3 seeds for paired tests; target checks use the mean over seeds of per-seed median and p95. 95% percentile-bootstrap intervals use 10,000 resamples over designs.

| ID | Question | Confirmed if |
|---|---|---|
| C1 | M-S beats physics only P | the CI of the median paired difference (P − M-S) lies above 0 at **each** of N = 10, 20, 40 |
| C2 | M-S beats data only D | the CI of the median paired difference (D − M-S) lies above 0 at **each** of N = 10, 20, 40 |
| C3 | M-S meets the median target with 10 labels | mean-over-seeds median ≤ 1% at N = 10 |
| C4 | M-S meets the tail target with 10 labels | mean-over-seeds p95 ≤ 3% at N = 10 (bootstrap CI of the seed-averaged p95 also reported) |

**Secondary (descriptive, no confirmation claim):** all arms at all N; the smallest N meeting both targets per arm; the form contrast (G-R vs M-S); error against slenderness; the pilot-test vs fresh-test comparison; w-only errors.

## Reporting commitments
- Every C1–C4 outcome is reported whether or not it is confirmed. A failed C4 is reported as a failure, not re-tuned.
- **No re-runs with altered settings on the fresh set.** Any later change creates a new experiment and requires another fresh test set.
- The fresh designs, loads and labels are saved, with checksums, in the E4 output so the evaluation can be audited.
