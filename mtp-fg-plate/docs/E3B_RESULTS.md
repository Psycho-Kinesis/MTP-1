# E3b: imperfect physics + higher-fidelity labels (executed)

9 November 2026 · Designs, mixture loads and the 140/30/30 split come from `results/pilot_200` (read-only; checksum asserted). Results: `results/e3b_model_form/`.
Command: `python experiments/run_e3b.py --output <new dir>`. The run takes about 3 min on one CPU.

## Question
In practice the physics we can afford is imperfect. When the physics loss (FSDT) **disagrees** with the model that produced the labels, what do labels and physics each contribute?

## Set-up
- **Target ("truth"): Reddy's third-order theory (TSDT)** for the same 200 designs and loads. It stands in for an expensive higher-fidelity model; it is itself a plate theory, not 3D elasticity.
- **The new TSDT solver** (`fgplate/tsdt.py`) uses the same five generalized fields and Navier basis as FSDT. It is checked in three ways:
  - with the third-order term off, it reproduces the FSDT matrices to 3e-16;
  - it matches an independent symbolic Ritz TSDT derivation to 9e-16 (now part of `validate.py --full`, 25/25);
  - it reproduces published higher-order deflections (0.7573 exactly; 0.8815 / 1.0087 against SSDT 0.8819 / 1.0089).
- **Physics loss: FSDT energy**, which carries a **model-form error** relative to TSDT.
- **Size of that error, measured on training designs only before the rules were fixed:** five-field median 0.36%, p95 5.6%, max 18%. It is concentrated in **thick plates** (a/h < 10: median 4.4%; a/h ≥ 25: 0.08%) and in the **rotation fields** (w alone: median 0.05%, p95 2.3%).
- **Arms:** all use the same model class (tensor-Legendre polynomials, degrees 2–7) with a per-field relative ridge.

| Arm | Trained on |
|---|---|
| P, physics only | FSDT energy at 560 unlabelled collocation designs. No training labels. |
| D, data only | TSDT labels at N designs (same nested subsets as E2 and E3a) |
| Hy, hybrid | TSDT labels + weighted FSDT energy, with shared coefficients |
| Hc, physics + correction | P's prediction plus a polynomial correction (degree 0–4) fitted to TSDT residuals at the N labelled designs |
| Reference | exact FSDT per design: the pure model-form gap, with no learning |

- **Selection.** Selection is tail-aware, fixed before the run: minimize max(median / 1%, p95 / 3%) on validation. This replaces E3a's median-first rule. **All arms, including P, use the 30 validation TSDT labels for hyperparameter selection**; "labels" in the tables means training labels.
- **Pre-declared questions:**
  - **Q1 (bias correction):** does arm X beat physics only P at N = 10, 20 and 40?
  - **Q2 (label efficiency):** does arm X beat data only D at N = 10, 20 and 40?
  - Both use paired per-design differences on the 30 validation designs, averaged over seeds; "supported" means the 95% bootstrap interval of the median difference lies above 0.
  - **Q3:** the smallest label budget meeting the target for each arm.
- **Testing.** The test set was evaluated once, after all choices were frozen.

## Results (median % / p95 % vs TSDT, five fields)

| Arm | Labels | Validation | Test |
|---|---|---|---|
| Exact FSDT (reference, no learning) | – | 0.38 / 5.71 | 0.36 / 4.91 |
| Physics only P | 0 | 0.98 / 4.55 | 1.17 / 5.18 |
| Data only D | 10 | 18.1 / 91.2 | 18.6 / 77.0 |
| | 20 | 7.63 / 44.3 | 6.37 / 29.5 |
| | 40 | 1.60 / 11.5 | 2.18 / 21.3 |
| | 70 | 0.41 / 3.79 | 0.40 / 3.31 |
| | 140 | **0.045 / 1.01** | **0.10 / 0.82** |
| Hybrid Hy | 10 | 1.23 / 7.44 | 1.36 / 6.76 |
| | 20 | 1.46 / 8.37 | 1.58 / 12.3 |
| | 40 | 1.23 / 8.93 | 1.76 / 19.6 |
| | 70 | 0.50 / 3.33 | 0.36 / 3.01 |
| | 140 | 0.046 / 1.01 | 0.10 / 0.83 |
| Physics + correction Hc | 10 | **0.78 / 3.81** | **0.82 / 4.23** |
| | 20 | 0.80 / 3.94 | 0.66 / 3.93 |
| | 40 | 0.57 / 3.78 | 0.57 / 6.11 |
| | 70 | 0.47 / 3.06 | 0.51 / 3.50 |
| | 140 | 0.37 / 1.75 | 0.33 / 2.42 |

### Pre-declared answers

| Question | Hybrid Hy | Physics + correction Hc |
|---|---|---|
| **Q2: beats data only at N = 10/20/40?** | **Supported.** Median differences +15.3 / +5.0 / +0.41 points; hybrid better on 30/30, 29/30 and 29/30 designs | **Supported.** +16.8 / +6.2 / +1.45 points; 30/30, 30/30 and 28/30 designs |
| **Q1: beats physics only at N = 10/20/40?** | **Not supported.** −0.16 [−0.54, +0.04], −0.35 [−0.91, +0.18], **−0.74 [−1.27, −0.18]**: at N = 40 the hybrid is significantly *worse* than physics only | **Not supported.** +0.20 [−0.03, +0.41], +0.06 [−0.13, +0.47], −0.00 [−0.26, +0.53] |
| Q1 at N = 140 | Supported (+0.82; 29/30) | Supported (+0.37; 22/30) |
| **Q3: smallest N meeting median ≤ 1%, p95 ≤ 3%** | 140 | 140 (p95 3.06% at N = 70, just above the target) |
| Physics only meets target with 0 labels? | No (p95 4.55%) | |

Data only D also first meets the target at N = 140.

## What this shows
1. **Imperfect physics is still worth a great deal when labels are scarce (Q2).** With 10 high-fidelity labels, data only is unusable (18% median, 91% p95). Physics + correction gives 0.78% / 3.8%, and the hybrid 1.2% / 7.4%. With FSDT physics, no labels at all already gives 0.98% / 4.6%.
2. **A few random high-fidelity labels do not measurably remove the physics bias (Q1).** Correcting the FSDT→TSDT discrepancy took about 140 labels, and at that point data alone is as good or better.
3. **The mechanism is visible per design** (`e3b_error_vs_thickness.png`, seed 11, N = 20). The learned correction *does* fix thick plates (a/h < 8): errors fall from about 4–8% to about 0.2–1.4%. But it *adds* error to slender plates (a/h > 20), where FSDT and TSDT agree to about 0.1%, pushing them to 1–2.5%. A global polynomial correction fitted to 20 randomly placed labels spreads its error everywhere, so the median gain cancels out.
4. **The accuracy floor is set by the physics model, not by the learner.** Physics-only p95 (4.6%) sits near the exact-FSDT gap (5.7%). No amount of FSDT collocation reaches the 3% tail target. Only labels from the better model, or a better physics model, can close it.
5. **The hybrid with shared coefficients is the weakest way to combine them here.** The physics and the labels pull the same coefficients towards different answers. Keeping the physics prediction fixed and learning only a correction (Hc) is better for every N ≤ 70.

## Caveats
- TSDT is a stand-in for "high fidelity". FSDT vs 3D elasticity would be a different (and larger, for thick FG plates) gap.
- The five-field metric is dominated by the rotation fields, where the theories differ most. For deflection w alone the gap is much smaller (validation p95 1.6%). Both metrics are in the manifest.
- The P arm's hyperparameters (degree 5, ridge 1e-6 for seed 11) were chosen by the tail-aware score against TSDT validation labels. Its median (0.98%) is therefore worse than exact FSDT (0.38%), while its p95 (4.55%) is slightly better. This is a selection trade-off, not a sign that P learned TSDT.
- 30 validation and 30 test designs (pilot test set, viewed in earlier experiments); tails rest on one or two designs.

## Status
| Completed and verified | Not done |
|---|---|
| TSDT Navier stiffness (`fgplate/tsdt.py`, 5 unit tests, 2 full-validation checks) | 3D-elasticity or FE labels |
| Per-field relative ridge (`scope='block'`; E3a keeps `'mode'` and reproduces exactly) | Targeted label placement, structured corrections (proposed E3c) |
| E3b: 4 arms × 5 budgets × 3 seeds, Q1–Q3, single test evaluation | Held-out load families (H2), clamped family, inverse design |

## Recommended next experiment: E3c, physics-guided correction and label placement
Pre-register both changes **before** running. Both are motivated by the mechanism in point 3:
1. **An asymptotically consistent correction.** FSDT and TSDT coincide as h/a → 0 (the unit test shows a 1e-4 difference at a/h = 200). So restrict the correction to vanish there, e.g. δ(x) = (h/a)²·p(x). This removes the error the correction adds to slender plates. It is a physics prior, not a tuning knob.
2. **Physics-guided label placement.** Instead of random labels, spend the N high-fidelity labels on the designs where FSDT expects shear deformation to matter most. A label-free indicator from FSDT itself, such as the shear share of the (1,1) compliance, Q/(A_s k²) ÷ W_total, would pick them from the training pool. Compare with random placement at the same N, using the same Q1 and Q2 tests.

If E3c makes Q1 hold at N ≤ 40, that is the central thesis result: *cheap imperfect physics + a few well-placed high-fidelity labels beats either alone*. If it doesn't, the negative result together with this mechanism is still a defensible finding.
