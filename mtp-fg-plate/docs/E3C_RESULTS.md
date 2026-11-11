# E3c: physics-structured correction and physics-guided label placement (executed)

9 November 2026 · Same data, split, seeds, budgets, collocation, model class and tail-aware selection as E3b. Target: Reddy TSDT. Physics loss: FSDT. Results: `results/e3c_structured/`.
Command: `python experiments/run_e3c.py --output <new dir>`. The run takes about 50 s on one CPU.

## Why E3c
E3b showed that a few random TSDT labels could not remove FSDT's bias (Q1 failed at N ≤ 40). The mechanism: a **global** polynomial correction fixed thick plates but added error to slender plates, where FSDT and TSDT already agree. E3c tests two physics-motivated fixes. Both were fixed **before** any E3c fit, using diagnostics on the 140 training designs only:

| Diagnostic (training designs only) | Result | Consequence |
|---|---|---|
| Relative FSDT→TSDT transfer discrepancy vs (k h)², with k² = π²(m² + n²(a/b)²) | log–log correlation **0.990**; discrepancy/(k h)² lies in **0.06–0.16** (5–95%) across all design-modes; with (h/a)² alone the spread is 25× wider | Correction proportional to (k h)²; the multiplicative form H_P·(k h)²·p(x) needs a nearly constant p |
| FSDT shear-compliance share ψ of the (1,1) mode (label-free) vs the five-field gap | Spearman **0.981**; the top 20% of designs by ψ hold 96% of the squared gap | Spend labels where ψ is large |

## Set-up
- **Base model:** the physics-only arm P (FSDT energy, 560 unlabelled designs), exactly as in E3b.
- **Six correction variants on top of P (a 3 × 2 design):**

| | Random labels (R) | Shear-guided labels (S) |
|---|---|---|
| **G**: global polynomial (= E3b Hc) | G-R | G-S |
| **A**: (k h)² × polynomial (vanishes for thin plates) | A-R | A-S |
| **M**: H_P × (k h)² × polynomial (relative correction) | M-R | **M-S (primary)** |

- **Placement S** samples training designs without replacement with probability ∝ ψ (seed-dependent and nested). Placement R uses the same random subsets as E2/E3a/E3b.
- **Selection:** correction degree 0–4 and ridge, by the tail-aware validation score of E3b.
- **Primary hypotheses (pre-registered):**
  - **Q1:** M-S beats physics only at N = 10, 20 and 40.
  - **Q2:** M-S beats data only (random labels) at the same budgets.
  - Both use the paired per-design bootstrap test of E3b. The other variants and the factor contrasts are secondary.
- **Consistency check (asserted in the script):** P, D and G-R reproduce E3b's P, D and Hc **exactly** (all 33 comparisons pass).

## Results

### Primary hypotheses: both supported

| N | Q1: P − M-S (points) [95% CI], designs better | Q2: D − M-S (points) [95% CI], designs better |
|---|---|---|
| 10 | +0.17 [+0.05, +0.53], 22/30 | +17.1 [+12.2, +20.6], 30/30 |
| 20 | +0.23 [+0.04, +0.55], 22/30 | +7.08 [+4.91, +8.29], 30/30 |
| 40 | +0.23 [+0.04, +0.55], 22/30 | +1.70 [+1.03, +2.12], 29/30 |
| 70 | +0.26 [+0.12, +0.62], 24/30 | +0.08 [+0.01, +0.16], 20/30 |
| 140 | +0.27 [+0.07, +0.84], 21/30 | −0.30 (data only better), 3/30 |

### Error vs TSDT (median % / p95 %, mean over 3 seeds)

| Arm | N = 10 | 20 | 40 | 70 | 140 | Smallest N meeting target (validation) |
|---|---|---|---|---|---|---|
| Physics only P (0 labels) | 0.98 / 4.55 | | | | | not met |
| Data only D | 18.1 / 91.2 | 7.63 / 44.3 | 1.60 / 11.5 | 0.41 / 3.79 | **0.045 / 1.01** | 140 |
| G-R (E3b Hc) | 0.78 / 3.81 | 0.80 / 3.94 | 0.57 / 3.78 | 0.47 / 3.06 | 0.37 / 1.75 | 140 |
| G-S | 0.95 / 4.58 | 1.42 / 6.66 | 0.93 / 5.01 | 0.36 / 3.11 | 0.37 / 1.75 | 140 |
| A-R | 0.67 / 5.61 | 0.55 / 3.55 | 0.43 / 2.72 | 0.44 / 2.25 | 0.44 / 2.08 | 40 |
| A-S | 0.57 / 2.61 | 0.43 / 2.28 | 0.43 / 2.18 | 0.43 / 2.17 | 0.44 / 2.08 | **10** |
| M-R | 0.48 / 2.32 | 0.50 / 2.23 | 0.49 / 2.15 | 0.46 / 2.12 | 0.35 / 1.80 | **10** |
| **M-S (primary)** | **0.50 / 2.22** | **0.47 / 2.05** | **0.42 / 2.15** | **0.34 / 1.99** | 0.35 / 1.80 | **10** |

Rows above are **validation**. The **single test evaluation** is below:

| Arm | N = 10 | 20 | 40 | 70 | 140 |
|---|---|---|---|---|---|
| Physics only P | 1.17 / 5.18 | | | | |
| Data only D | 18.6 / 77.0 | 6.37 / 29.5 | 2.18 / 21.3 | 0.40 / 3.31 | 0.10 / 0.82 |
| G-R (E3b Hc) | 0.74 / 4.16 | 0.71 / 4.24 | 0.53 / 6.34 | 0.47 / 3.74 | 0.32 / 2.50 |
| M-R | 0.56 / 3.18 | 0.53 / 3.19 | 0.41 / 3.17 | 0.43 / 3.22 | 0.35 / 3.44 |
| **M-S** | **0.48 / 3.27** | **0.43 / 3.98** | **0.36 / 3.19** | **0.33 / 3.13** | 0.35 / 3.44 |
| A-S | 0.49 / 3.18 | 0.39 / 3.25 | 0.40 / 3.24 | 0.37 / 3.15 | 0.35 / 3.14 |

### Secondary contrasts (validation, paired)
- **The correction form is the decisive factor.** M beats G significantly at N = 10, 20, 40 and 70 with guided labels (by 0.50, 1.02, 0.59 and 0.02 points), and at N = 10–40 with random labels. A beats G at N = 10–40 with guided labels and N = 20–70 with random labels.
- **Guided placement adds almost nothing once the correction is structured.** For A and M the difference between S and R is at most 0.05 points and significant only once (M at N = 70). For the global correction, guided placement **significantly hurts** at N = 10, 20 and 40: by 0.29, 0.84 and 0.38 points, with 95% intervals entirely below zero. Labels concentrated on thick plates make a global polynomial extrapolate badly to slender ones.

## What this shows
1. **Cheap imperfect physics plus a handful of high-fidelity labels beats either alone, *if the correction respects the physics*.** With 10 TSDT labels, M-S has a validation median of 0.50% and a p95 of 2.22%:
   - better than physics only (0.98% / 4.55%) on 22 of 30 designs, with a CI above 0;
   - far better than data only (18.1% / 91.2%);
   - data only needs 140 labels to reach the same target.
2. **What made the difference is knowing *how* the model-form error behaves.** FSDT and TSDT differ only through transverse shear, so the discrepancy scales with (k h)² and vanishes for thin plates. Building that into the correction lets a degree-1 polynomial (selected at N = 20 and 40) correct the bias from very few labels, without disturbing slender plates. This is visible per design in `e3c_error_vs_thickness.png`.
3. **The gain over physics only is mainly in the tails.** The median improves by about 0.2 points; the 95th percentile improves from about 4.6% to about 2.1% (validation).

## Caveats (state them in the thesis)
- **The test tails sit just above the target.** On the single test evaluation, A-S, M-R and M-S have medians of 0.33–0.56% but 95th percentiles of **3.13–3.98%**, not ≤ 3%.
  - Part of the gap is selection optimism: validation chose the hyperparameters, so validation errors are biased low.
  - The physics-only arm itself also does worse in the test tail (p95 5.2% vs 4.6% on validation), although the exact-FSDT gap does not (4.9% vs 5.7%).
  - With 30 test designs, the p95 rests on the second-worst design.
  - Report it as: "meets the median target; relative to physics only it roughly halves the tail error on validation (p95 4.6% → 2.1–2.2%) and cuts it by about 25–40% on test (5.2% → 3.1–4.0%); the 3% p95 target is met on validation but narrowly missed on test".
- **The effect over physics only is significant but modest in the median** (+0.17 to +0.27 points); the larger benefit is in the tails.
- **Data only overtakes everything at N = 140** (validation 0.045% / 1.01%; test 0.10% / 0.82%). With enough high-fidelity labels, physics is no longer needed for this smooth 3-parameter problem.
- **TSDT stands in for high fidelity.** The (k h)² structure is specific to FSDT vs TSDT, where the theories differ only in shear. A different fidelity gap (3D elasticity, FE with other boundary conditions) may need a different structural prior. The method, *use the physics to decide what form the correction takes*, is what transfers.
- **The (k h)² form was chosen after E3b's result**, using training-only diagnostics, and was pre-registered before any E3c fit. The test set had been viewed in earlier experiments (pilot test set).

## Status
| Completed and verified | Not done |
|---|---|
| `assemble_data(..., prefactor)` and `predict_scaled` (1 new unit test; 30 in total) | Locked fresh test set for the final thesis numbers |
| E3c 3×2 design, Q1/Q2 primary + secondary contrasts, single test evaluation; P, D and G-R reproduce E3b exactly | Held-out load families (H2), g ∈ (8, 10] extrapolation |
| | Clamped family, FEM verification, inverse design |

## Recommended next steps
1. **Lock the result with a fresh test set (E4, the confirmation run).** Generate new designs with generator pilot-1.1 (new seed, about 100 designs), never used for any choice. Rerun only the frozen configurations (P, D, M-S, M-R; hyperparameter grids as here, selection on the existing validation set) and report test metrics once. This gives thesis-grade numbers and settles whether the p95 target holds.
2. **Then broaden:** the held-out Gaussian load family (H2), and the inverse design demonstration (minimum mass subject to deflection/frequency limits, using the M-S surrogate and verified by the solver).
