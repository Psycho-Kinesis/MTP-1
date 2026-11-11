# E4: confirmation on a fresh, locked test set (executed)

9 November 2026 · Pre-registration: [`E4_PREREGISTRATION.md`](E4_PREREGISTRATION.md). It was written before any fresh design existed; its sha256 `a60526d7…` is asserted by the script and stored in the manifest. Results: `results/e4_confirmation/`.
Command: `python experiments/run_e4.py --output <new dir>` (about 70 s). An identical rerun reproduces every number and the fresh set bit for bit.

## Protocol, as pre-registered
1. **Refit the frozen E3c arms** (P, D, G-R, A-R, A-S, M-R, M-S; N = 10–140; seeds 11/22/33) from the pilot only: training labels, FSDT collocation, and validation-based selection. **All 93 refitted arms reproduced E3c's validation metrics** before any fresh data existed.
2. **Only then generate the fresh test set:** 200 designs, generator pilot-1.1, seed 20261201 (never used before), smooth-mixture loads and TSDT labels.
   - It covers the full domain: g 0.007–7.95, a/h 5.0–99.9, b/a 0.51–2.0.
   - It is asserted disjoint from all pilot and collocation designs.
3. **Evaluate every frozen model once.** Nothing was fitted, selected or tuned on the fresh designs.

## Confirmatory results (primary arm M-S)

| ID | Pre-registered question | Result on 200 fresh designs | Verdict |
|---|---|---|---|
| C1 | M-S beats physics only at N = 10, 20, 40 | median paired gain +0.12 [+0.05, +0.23], +0.11 [+0.07, +0.22], +0.15 [+0.08, +0.32] points; M-S better on 144, 137 and 146 of 200 designs | **Confirmed** |
| C2 | M-S beats data only at N = 10, 20, 40 | +17.8 [+16.0, +19.4], +7.39 [+6.14, +8.13], +1.84 [+1.57, +2.32] points; better on 200, 200 and 199 of 200 designs | **Confirmed** |
| C3 | Median ≤ 1% with 10 labels | **0.53%** (mean over seeds; seed-averaged 0.51% [0.47, 0.59]) | **Confirmed** |
| C4 | p95 ≤ 3% with 10 labels | **2.85%** (per seed 2.48 / 2.51 / 3.58); seed-averaged p95 2.52% with a 95% CI of **[2.05, 3.88]** | **Confirmed by the pre-registered rule, but marginal**: the CI straddles 3% and one seed exceeds it |

## All arms on the fresh set (median % / p95 %, mean over seeds)

| Arm | N = 10 | 20 | 40 | 70 | 140 | Smallest N meeting both targets |
|---|---|---|---|---|---|---|
| Physics only P (0 labels) | 0.98 / 5.78 | | | | | not met |
| Data only D | 17.8 / 79.9 | 6.65 / 39.0 | 1.87 / 18.6 | 0.44 / 4.98 | **0.064 / 0.92** | 140 |
| G-R (E3b correction) | 0.87 / 4.35 | 0.78 / 4.09 | 0.55 / 5.81 | 0.48 / 3.74 | 0.36 / 2.78 | 140 |
| A-R | 0.71 / 7.09 | 0.57 / 4.16 | 0.39 / 3.39 | 0.38 / 3.11 | 0.38 / 3.09 | not met |
| A-S | 0.56 / 4.11 | 0.43 / 3.50 | 0.40 / 3.00 | 0.37 / 3.22 | 0.38 / 3.09 | 40 |
| M-R | **0.48 / 2.79** | 0.46 / 2.58 | 0.42 / 2.54 | 0.42 / 2.73 | 0.36 / 3.99 | **10** |
| **M-S (primary)** | **0.53 / 2.85** | 0.43 / 4.01 | 0.38 / 2.79 | 0.37 / 2.97 | 0.36 / 3.99 | **10** |
| Exact FSDT (model-form reference) | 0.36 / 4.59 | | | | | |

**Comparison with the pilot test set (E3c), M-S at N = 10:** 0.48% / 3.27% there vs 0.53% / 2.85% here. The medians agree. The tails differ by about the width of their uncertainty: 30 designs then, 200 now.

## What E4 establishes
1. **The central claim is confirmed on unseen data.** With 10 high-fidelity (TSDT) labels, cheap imperfect physics (FSDT) plus a physics-structured correction beats physics alone (on about 70% of designs, with CIs above 0) and data alone (on 100% of designs). It reaches a median error of about 0.5%, while data alone needs about 140 labels for comparable accuracy.
2. **The correction works where the physics is wrong.** For thick plates (a/h < 15), physics-only errors equal the FSDT–TSDT gap (3–10%). The correction brings them to 0.3–2% (`e4_error_vs_thickness.png`).
3. **The form of the correction is decisive (secondary).** M-S beats the global correction G-R at N = 10–70 (paired, CIs above 0); the global one never meets the targets below 140 labels.

## What E4 does not establish (state these in the thesis)
- **The 3% tail target is not robust.**
  - At N = 10 it is met on average (2.85%), but the CI reaches 3.9% and one seed gives 3.58%.
  - M-S exceeds 3% at N = 20 (4.01%, driven by seed 33 at 7.0%) and at N = 140 (3.99%). M-R is more stable (2.5–2.8% for N = 10–70).
  - Report it as: *"the tail target is met at some budgets and not others; tail error is roughly halved relative to physics only but not reliably below 3%."*
- **The remaining tail comes from the base model's fit, not from the physics.**
  - The three worst fresh designs (8–21% error) are moderately slender (a/h ≈ 24–44), **narrow (b/a ≈ 0.53–0.68), nearly uniform (g ≈ 0–0.2)** plates whose responses are dominated by φy under loads heavy in the (1,2) mode.
  - On them, exact FSDT is within **0.3–0.75%** of TSDT, yet physics-only already errs by 8–22%, and the correction (which shrinks for slender plates, by design) leaves that error untouched.
  - For slender plates in general (a/h > 30), exact FSDT agrees with TSDT to 0.01–0.1%, while the label-free polynomial base model errs by 0.2–7%.
- **With 140 labels, data only is best** (0.064% / 0.92%). Physics pays off when high-fidelity labels are scarce.
- **TSDT stands in for high fidelity;** the (k h)² structure is specific to this FSDT→TSDT gap.
- **All results are within the simply supported, linear, smooth-mixture-load setting** of the pilot.

## Status
| Completed and verified | Not done |
|---|---|
| Pre-registered confirmation on 200 fresh designs (C1–C4 confirmed; C4 marginal); 93/93 frozen arms reproduce E3c; deterministic rerun | A base physics model accurate in the narrow / near-uniform / slender corner |
| `bootstrap_ci(..., 'p95')` (tested) | Held-out load families (H2), clamped family, FEM, inverse design |

## Recommended next step: E5, a better label-free base model, then a new locked test set
The diagnosis points at one component: the **label-free physics-only surrogate P** (a degree-5 polynomial in the three design coordinates). Options to evaluate on the pilot's train and validation sets only:
- **Physics-informed coordinates:** use log(b/a), or each mode's wavenumber (k a) and (k h), instead of b/a. The transfer functions are rational in k² (∝ 1/(D*k⁴) + 1/(A_s k²)), which polynomials in b/a fit poorly for narrow plates.
- **Higher capacity where it is needed:** log-magnitude targets (as in E2's B3L) or sparse higher-degree terms.

Success criterion for P: reach the exact-FSDT accuracy level for slender plates. Then freeze, and confirm on a **new** fresh test set (seed to be pre-registered; the E4 set is now spent). After that come H2 (held-out load shapes) and the inverse-design demonstration.
