# H2 pre-registration: generalization to unseen load shapes

Written 9 November 2026, after feasibility checks on **pilot training designs only** and **before** any H2 test design exists. `experiments/run_h2.py` asserts the sha256 of this file.

## Question (PID hypothesis H2)
Every model so far was trained and tested on one load family: smooth signed mixtures of the 25 sine modes (m, n ≤ 5). Does the confirmed E5 pipeline keep its accuracy for load shapes it never saw, including loads with content outside its 5×5 modal basis, without new high-fidelity designs?

## Facts used to design the test (training designs only)
- **The plate response is a strong low-pass filter** (∝ load/k⁴). On three training designs (a/h = 5, 15, 100), the share of the TSDT response norm outside the 5×5 modes was:
  - uniform pressure: 0.14–0.35%;
  - wide Gaussian patch (σ = 0.15): ≤ 0.08%;
  - narrow patch (σ = 0.05): 0.8–2.7%.
  - Outside 15×15 it was ≤ 0.04% for all three, and outside 25×25 ≤ 0.012%.
- So a **25×25 TSDT reference** is used as the truth, and **15×15** as the extended basis.

## Test designs and load families
- **200 fresh designs**, generator pilot-1.1, seed **20261231**. Never used: pilot 20261109/10, E4 20261201, E5 20261215, collocation 1011/1022/1033, label subsets 11/22/33. Asserted disjoint from the pilot designs, all collocation designs and the E4 and E5 fresh sets.
- **Each design gets four loads** (load shapes are defined in normalized coordinates ξ = x/a, η = y/b):

| Family | Definition | Role |
|---|---|---|
| L1 smooth mixture | as in training (25 signed modes, unit norm) | in-distribution control |
| L2 uniform pressure | q = 1 Pa everywhere (one-sided; all odd modes) | new shape, nearly in-basis |
| L3 wide Gaussian patch | q = exp(−[(ξ−ξ₀)² + (η−η₀)²]/(2σ²)), σ ~ U[0.10, 0.20], (ξ₀, η₀) ~ U[0.25, 0.75]² | new shape, in-basis |
| L4 narrow Gaussian patch | same form, σ ~ U[0.04, 0.07], (ξ₀, η₀) ~ U[0.2, 0.8]² | new shape, partly **out of basis** |

- **Patch randomness:** parameters are drawn from `default_rng([20261231, family, index])`.
- **Load coefficients:** the patch is separable, Q_mn = 4·Iₘ(ξ₀, σ)·Iₙ(η₀, σ), with Gauss–Legendre quadrature (600 points). Uniform load: Q_mn = 16/(π² m n) for odd m, n.

## Metric
The exact continuous relative L2 norm over the plate of the five scaled generalized fields, computed from modal coefficients by orthogonality. Every basis function has the same norm, so missing modes count fully as error. The truth is TSDT with modes m, n ≤ 25. This replaces the 41×41 grid norm of earlier experiments, which cannot resolve modes beyond about 20.

## Arms
All arms use pilot training/collocation data only, with hyperparameters **frozen from E5** (`run_e5.fit_all`, asserted to reproduce `results/e5_base_model_v2`). No selection is done on any H2 data.

| Arm | Description |
|---|---|
| A | frozen E5 pipeline M-S@P2, 5×5 modes only (as confirmed in E5) |
| P-ext | label-free P2 base fitted for all 15×15 modes (same degree and ridge per seed as E5's P2) |
| B | A for the 5×5 modes + P-ext for the remaining modes (no extra use of labels) |
| **C (primary)** | **M-S@P2 extended to 15×15 modes.** P-ext base + multiplicative (k h)² correction for every mode, fitted on the **same** N shear-guided labelled designs as E5 (their TSDT solutions include all modes; no additional high-fidelity designs), with the correction degree and ridge frozen per (seed, N) from E5 |
| D | frozen E5 data only, 5×5 |
| D-ext | data only on the same random labelled designs, 15×15 modes, hyperparameters frozen from E5 |

- **References:** TSDT truncated to 5×5 (the floor for A and D); TSDT truncated to 15×15 (the floor for the extended arms); exact FSDT (25×25).
- **Asserted:** C, B and D-ext restricted to the 5×5 modes reproduce A, A and D exactly. Each mode is fitted independently.
- **Budgets and seeds:** N = 10 (primary), 40 and 140; seeds 11/22/33.

## Pre-registered questions (primary arm C, N = 10)
Targets: median ≤ 1%, p95 ≤ 3% (mean over seeds of per-seed values). Paired tests use the 95% bootstrap CI of the median per-design difference of seed-averaged errors.

| ID | Question | Confirmed if |
|---|---|---|
| Q1 | Control: C meets both targets on L1 | median ≤ 1% and p95 ≤ 3% on L1 |
| Q2 | C meets both targets on **each** new family L2, L3, L4 | median ≤ 1% and p95 ≤ 3% for each |
| Q3 | C beats data only (D-ext) on each of L2–L4 | CI of (D-ext − C) above 0 for each family |
| Q4 | C beats physics only (P-ext) on each of L2–L4 | CI of (P-ext − C) above 0 for each family |
| Q5 | Extending the basis matters for out-of-basis loads | CI of (A − C) above 0 on L4 |

**Secondary (descriptive):** all arms, families and budgets; errors against the truncation floors; error against slenderness and against patch width measured in plate thicknesses (σ·a/h). Plate theories are questionable when a load is narrower than about one thickness; such cases are reported, not excluded.

## Reporting commitments
All Q1–Q5 outcomes are reported, confirmed or not. No re-runs with changed settings on this test set.
