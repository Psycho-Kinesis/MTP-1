# E5: a better label-free base model, confirmed on a new fresh test set (executed)

9 November 2026 · Development: `results/e5_base_model` (rule v1, kept as the record) and `results/e5_base_model_v2` (rules v1 + v2); pilot train/validation only. Pre-registration: [`E5_PREREGISTRATION.md`](E5_PREREGISTRATION.md) (sha256 `f25d9ea8…`, asserted). Confirmation: `results/e5_confirmation/` (200 new fresh designs, evaluated once).
Commands: `python experiments/run_e5.py --output <dir>` (about 75 s) and `python experiments/run_e5_confirm.py --output <dir>` (about 100 s). Identical reruns reproduce every number and the fresh set bit for bit.

## Starting point
E4 confirmed the E3c pipeline but left a weak tail. Its worst errors came from the **label-free base model** P (a degree-5 polynomial in the design coordinates trained on the FSDT energy). In the narrow / near-uniform / slender corner, P was far less accurate than exact FSDT itself.

## Development (pilot train/validation only)
**Idea: classical-plate scaling.** Each mode's FSDT response scales, to leading order, as:

| Field | Scaling |
|---|---|
| w | 1/(k a)⁴ |
| φx, φy | (α a, β a)/(k a)⁴ |
| u0, v0 | the rotation scaling × the neutral-axis offset z₀/h = B₁₁/(A₁₁h) |

These use only geometry and stiffness integrals: no labels and no solve. On **training designs**, dividing exact FSDT transfer functions by these factors reduces their spread across designs and modes from up to 4×10⁴ to **about 3** (u, v, rotations) and about 40 (w, from shear). The polynomial then only has to learn a smooth remainder (`fgplate.transfer.classical_prefactor`, applied inside the energy loss through `assemble_physics(..., prefactor)`).

**Four pre-declared base variants** (FSDT energy only, same collocation, grids and selection as E3c):

| Base | b/a coordinate | Prefactor | vs exact FSDT (median / p95 %) | vs TSDT alone (median / p95 %) |
|---|---|---|---|---|
| P0 (= E3c/E4) | linear | – | 0.38 / 2.42 | 0.98 / **4.55** |
| P1 | log | – | 0.39 / 3.14 | 0.96 / 5.24 |
| **P2** | linear | classical | **0.027 / 0.10** | 0.40 / 5.73 |
| P3 | log | classical | 0.082 / 0.24 | 0.48 / 5.70 |

**Rule v1 (pre-declared): choose by each base's own score against TSDT → it chose P0.** This exposed a flaw in the rule. P2 reproduces FSDT about 25× more accurately, so on its own it inherits FSDT's full model-form gap (p95 5.7%, which is exactly exact FSDT's gap). P0 scores better only because its approximation errors partly cancel that bias in the tails.

**Rule v2 (declared after seeing rule v1's outcome, before any test data): choose by the downstream M-S pipeline's validation score at N = 10, 20 and 40 → it chose P2.** Downstream scores: P0 0.71, P1 0.97, **P2 0.053**, P3 0.099. Both outcomes are recorded in the manifest. Because rule v2 was chosen after seeing data, the result was then confirmed on new, untouched designs.

All E3c arms rebuilt in E5 (P, D, M-S, M-R on P0) reproduce E3c exactly (48 checks).

## Confirmation (200 new fresh designs, pilot-1.1, seed 20261215, evaluated once)
All 147 frozen arms reproduced development before the fresh designs were generated. The fresh designs are asserted disjoint from the pilot, every collocation set and the E4 fresh set.

### Pre-registered questions: all six confirmed

| ID | Question | Result | Verdict |
|---|---|---|---|
| C1 | M-S@P2 beats its own base P2 | +0.30 [+0.18, +0.44], +0.32 [+0.16, +0.45], +0.34 [+0.18, +0.49] points at N = 10, 20, 40; better on 182–184 of 200 designs | **Confirmed** |
| C2 | M-S@P2 beats data only | +19.1, +7.9, +2.3 points at N = 10, 20, 40; better on 200/200 designs | **Confirmed** |
| C3 | Median ≤ 1% at N = 10 | **0.062%** | **Confirmed** |
| C4 | p95 ≤ 3% at N = 10 | **0.33%** | **Confirmed** |
| C5 | p95 ≤ 3% at every N ≤ 70, and the N = 10 p95 CI upper end ≤ 3% | p95 = 0.33 / 0.19 / 0.14 / 0.13% at N = 10 / 20 / 40 / 70; seed-averaged p95 CI at N = 10: **[0.20, 0.49]%** | **Confirmed** (E4's weakness resolved) |
| C6 | The new pipeline beats the old (M-S@P0) | +0.43 [+0.39, +0.48], +0.45 [+0.39, +0.49], +0.40 [+0.31, +0.44] points; better on **200/200** designs at every N | **Confirmed** |

### All arms on the new fresh set (median % / p95 %, mean over seeds)

| Arm | N = 10 | 20 | 40 | 70 | 140 |
|---|---|---|---|---|---|
| Old base P0 alone (0 labels) | 0.87 / 5.67 | | | | |
| New base P2 alone (0 labels) | 0.36 / 5.16 | | | | |
| Exact FSDT (model-form reference) | 0.36 / 5.19 | | | | |
| Data only D | 18.3 / 75.9 | 6.77 / 39.0 | 2.06 / 15.0 | 0.45 / 4.40 | 0.067 / 1.01 |
| Old pipeline M-S@P0 (E3c/E4) | 0.51 / 2.64 | 0.46 / 3.06 | 0.39 / 2.57 | 0.38 / 2.91 | 0.39 / 3.55 |
| **New pipeline M-S@P2** | **0.062 / 0.33** | **0.045 / 0.19** | **0.034 / 0.14** | **0.031 / 0.13** | **0.032 / 0.12** |
| M-R@P2 (random labels) | 0.106 / 1.02 | 0.075 / 0.60 | 0.042 / 0.26 | 0.035 / 0.15 | 0.032 / 0.12 |

- **New base accuracy against exact FSDT on fresh designs:** P2 0.030% / 0.11% vs P0 0.35% / 2.52%.
- **Worst designs (M-S@P2, N = 10):** 1.74%, 1.64% and 0.97%, all on very thick, narrow plates (a/h ≈ 5.3–6.0, b/a ≈ 0.58–0.72). This is where the FSDT–TSDT gap is largest (up to about 20%).
- **Shear-guided placement now helps clearly in the tails:** M-S vs M-R p95 is 0.33% vs 1.02% at N = 10 and 0.19% vs 0.60% at N = 20. In E3c, with the old base, it made little difference.

## What E5 establishes
1. **The final pipeline, confirmed on never-seen designs.**
   - Pipeline: a label-free FSDT surrogate with classical-plate scaling, plus a multiplicative (k h)² correction learned from **10** shear-guided high-fidelity (TSDT) labels.
   - Accuracy: median 0.06% and p95 0.33%, about 16× and 9× below the 1% / 3% targets, with tight confidence intervals.
   - Against the alternatives: it beats data only at every budget (by 19 points at N = 10, and still at N = 140), beats the physics model alone, and beats the E3c/E4 pipeline on every one of 200 designs.
2. **Two kinds of physics knowledge do the work, and each is necessary.**
   - **Scaling** makes the label-free physics surrogate accurate (E5).
   - **The structure of the model-form error** (∝ (k h)², relative to the physics prediction) makes the correction learnable from very few labels (E3c).
   - Neither alone suffices: base alone p95 5.2%; old base + correction p95 2.6–3.6%.
3. **A methodological lesson worth reporting.** Choosing a physics surrogate by how well it matches the *high-fidelity* target on its own rewards accidental bias cancellation. Choose it by how well it reproduces its *own* physics, or by downstream pipeline performance.

## Caveats (state them in the thesis)
- **The FSDT→TSDT gap is unusually well structured:** H_TSDT ≈ H_FSDT·(1 + c(k h)²) with c nearly constant. That is why 10 labels suffice. A real high-fidelity gap (3D FE, experiments, other boundary conditions) is likely to need more labels or a richer correction. The transferable method is *use the physics to choose the surrogate's scaling and the correction's form*; the specific numbers are not transferable.
- **Rule v2 was declared after seeing rule v1's outcome.** This is why the confirmation on a new fresh set was required, and it is the confirmation (not the development) that supports the claims above.
- **TSDT is itself a plate theory, not 3D elasticity.** Everything is within simply supported, linear, smooth-mixture-load conditions.
- **The confirmation used 200 designs**, so p95 rests on about 10 designs per seed; the intervals are reported.

## Status
| Completed and verified | Not done |
|---|---|
| `classical_prefactor`, `assemble_physics(..., prefactor)`, `design_coordinates(..., aspect='log')` (1 new test; 31 in total) | Held-out load families (H2) |
| E5 development (4 base variants, rules v1 and v2), E3c reproduction | Clamped plates / other boundary conditions, FEM or 3D labels |
| Pre-registered confirmation on 200 new fresh designs: C1–C6 confirmed | Inverse-design demonstration |

## Recommended next steps
1. **H2: generalization to unseen load shapes.** For example a Gaussian patch load and loads with modes outside the 5×5 training basis. Test the frozen M-S@P2 pipeline, which is linear in the load and needs no retraining for new loads within the modal basis; out-of-basis modes need basis extension.
2. **Inverse design demonstration.** Minimize mass subject to deflection and frequency limits using the M-S@P2 surrogate, and verify every candidate with the TSDT solver.
3. **Thesis write-up.** E2 → E3a → E3b → E3c → E4 → E5 now forms a complete, pre-registered argument.
