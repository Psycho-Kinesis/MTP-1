# E3a: physics-informed vs data-only learning at matched capacity (executed)

9 November 2026 · Uses `results/pilot_200` as delivered (read-only; checksum asserted on load). Results: `results/e3a_transfer/`.
Command: `python experiments/run_e3a.py --output <new dir>`. The run takes about 90 s on one CPU.

## Question
Does adding the governing physics reduce the number of labelled designs needed to learn the full five-field response (PID hypothesis H1), **when the physics-informed and data-only models share exactly the same model class**?

## Set-up
- **Response representation.** The response to the 25-mode smooth load mixture is written through the analytic Navier basis. Each of the 125 dimensionless transfer functions H̃_mn,c(g, a/h, b/a) is a **tensor-Legendre polynomial of total degree d** in the three design coordinates. The coordinates are mapped from the fixed pilot domain box to [−1, 1]³, so the mapping uses no data statistics (`fgplate/transfer.py`).
- **Three training arms, same model class:**

| Arm | Loss | Uses |
|---|---|---|
| D, data only | standardized least squares on labels | N labelled training designs (10–140; same nested subsets as E2) |
| P, physics only | FSDT total potential energy ½H̃ᵀK̂H̃ − f̂·H̃ per mode, averaged over designs | M **unlabelled** collocation designs (20–560): mode matrices only, **no solve, no labels** |
| Hy, hybrid | data loss + weight × energy | N labels + 560 collocation designs |

- **Linear solves.** Every loss is quadratic in the polynomial coefficients, so each fit is one linear solve per mode. The ridge is relative to each mode's Hessian scale.
- **Collocation designs.** Scrambled Halton points in the transformed domain, quantized as in pilot-1.1 and nested in M. They are asserted never to coincide with a validation or test design.
- **Selection and testing.** Degree (2–6), ridge and physics weight are chosen per arm, budget and seed on **validation only**, by median and then p95. Seeds are 11, 22 and 33. The test partition was evaluated **once**, after everything was frozen.
- **Rules fixed before the run** (stored in the manifest):
  - the target is median ≤ 1% and p95 ≤ 3%;
  - H1 holds at budget N if the 95% bootstrap interval of the median per-design paired difference (data-only − hybrid) lies above 0, checked at N = 10, 20 and 40.
- **Implementation checks** (`validation/test_transfer.py`):
  - the mode matrices reproduce the solver to 1e-9;
  - physics-only training recovers the exact answer **with no labels** whenever the model class contains it;
  - the energy minimizer is invariant to positive design weights;
  - data-only degree 4 reproduces E2's B3 exactly (0.444% / 3.62%).

## Results

### Labels needed to reach the target
Validation values are the mean over 3 seeds; test results come from the single final evaluation.

| Arm | Labels | Validation median % | Validation p95 % | Test median % | Test p95 % |
|---|---|---|---|---|---|
| Data only | 10 | 8.61 | 116 | 10.5 | 82.7 |
| Data only | 20 | 4.58 | 47.8 | 5.37 | 35.4 |
| Data only | 40 | 1.28 | 14.2 | 2.17 | 23.6 |
| Data only | 70 | 0.36 | 4.91 | 0.46 | 5.09 |
| Data only | **140** | **0.080** | **2.35** | 0.089 | 1.94 |
| Hybrid | **10** | **0.16** | **0.82** | 0.15 | 1.58 |
| Hybrid | 20 | 0.16 | 1.05 | 0.18 | 1.38 |
| Hybrid | 40 | 0.17 | 1.49 | 0.21 | 2.59 |
| Hybrid | 70 | 0.13 | 2.19 | 0.19 | **4.32** |
| Hybrid | 140 | 0.080 | 2.35 | 0.089 | 1.94 |
| Physics only (M = 140) | **0** | **0.12** | **0.98** | 0.15 | 1.55 |
| Physics only (M = 280) | 0 | 0.14 | 0.75 | 0.16 | 1.28 |
| Physics only (M = 560) | 0 | 0.14 | 0.83 | 0.17 | 1.27 |

**Smallest budget that meets the target:** data only needs **140 labels**; hybrid needs **10 labels**; physics only needs **0 labels** with M ≥ 140 collocation designs (at M = 70 its p95 is 3.12%).

### H1 (label efficiency): supported
Paired over the 30 validation designs, averaged over seeds:

| N | Median (data − hybrid), percentage points | 95% interval | Designs where hybrid is better |
|---|---|---|---|
| 10 | 8.04 | 6.42 – 19.6 | 30 / 30 |
| 20 | 4.54 | 3.91 – 5.99 | 30 / 30 |
| 40 | 1.41 | 1.01 – 1.56 | 30 / 30 |
| 70 | 0.27 | 0.15 – 0.41 | 29 / 30 |
| 140 | ≈ 0 | −4.5e-6 – 3.7e-7 | 12 / 30 |

### Capacity is what limits accuracy
Best ridge per degree; median / p95 on validation.

| Degree (coefficients per function) | 2 (10) | 3 (20) | 4 (35) | 5 (56) | 6 (84) |
|---|---|---|---|---|---|
| Physics only, M = 560 | 6.21 / 47.4 | 2.28 / 14.0 | 1.36 / 4.64 | 0.37 / 2.43 | **0.14 / 0.83** |
| Data only, N = 140 | 3.31 / 30.7 | 1.52 / 9.57 | 0.44 / 3.62 | 0.27 / 1.27 | **0.08 / 2.35** |

The physics-only error stops improving beyond M ≈ 140–280 collocation designs, but keeps falling with degree all the way to 6, which is the grid maximum. So with physics, the remaining error is set by the **model class**, not by the amount of information. Data-only at N = 140 can only use high degrees because it has enough labels. At degree 6 it fits the median better but has a worse tail (p95 2.35% vs 1.27% at degree 5).

## Interpretation and caveats (read before citing)
1. **The physics here is the same model that produced the labels.** The labels are FSDT solutions, and the physics loss is the FSDT energy. The physics term therefore contains *all* the information the labels contain, so label-free success is expected in principle. E3a shows this is also achieved **in practice at matched capacity**: 0 labels reach the target that data-only needs 140 labels for. It does not show that FSDT is physically accurate. Model-form error relative to 3D elasticity or higher-order theories (~1% at high g) is outside this experiment.
2. **There is no practical cost saving on this family.** A label costs one 5×5 solve per mode and a collocation design costs one 5×5 assembly per mode; both take microseconds. H1 is supported **methodologically**. Its practical value appears only where labels are expensive (FE, 3D, experiments) and the physics loss is cheap.
3. **The hybrid tail grows as labels increase.** Selection favours the median, so as labels grow it picks smaller physics weights, and the tails move back towards the data-only values. The test p95 at N = 70 (4.32%) exceeds the 3% target. If tails matter more, select by p95 or fix a high physics weight; that choice must be made before the next run.
4. **The test set is the pilot test set** (viewed earlier for the scalar and E2 results). One seed-group of 30 designs: the p95 values rest on one or two designs.

## Status
| Completed and verified | Not yet done |
|---|---|
| Transfer model with data / physics / hybrid losses (`fgplate/transfer.py`, 5 tests) | Neural (non-polynomial) model class; degree > 6 |
| E3a learning curves, H1 test, capacity study, single test evaluation | Labels from a higher-fidelity model (model-form mismatch) |
| | Held-out load families (Gaussian / out-of-basis), g ∈ (8, 10] extrapolation |
| | Clamped family, FEM verification, inverse design |

## Recommended next experiment: E3b, physics with model-form error
E3a's limitation is that the physics and the labels agree perfectly. E3b should break that agreement, which is exactly the situation in which a hybrid is useful in practice:
- **Labels** from a higher-fidelity plate theory: Reddy TSDT, already implemented symbolically in `validation/ritz_crosscheck.py`. Derive its per-mode Navier matrices once.
- **Physics loss** from FSDT, as here.
- **Question:** how many high-fidelity labels does the hybrid need to remove the FSDT-to-TSDT model-form error (about 1% at high g, comparable to the target)? Compare against data-only on the TSDT labels and against physics-only, which is biased.
- **Same machinery,** still CPU and linear algebra. Fix the tail-aware selection rule (point 3) **before** running.

**In parallel:** raise model capacity (degree 7–8 or a small network) for the physics-only arm. Also prepare the held-out Gaussian load family (generator pilot-1.1) to test H2.
