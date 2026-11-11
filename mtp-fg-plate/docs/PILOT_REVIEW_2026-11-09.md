# Independent review of the solver upgrades and 200-design pilot

**Reviewed package:** `MTP_Complete_Project_Package.zip` (pilot generator `pilot-1.0`, seed 20261109, created 2026-11-09T14:46:52Z)
**Reviewer:** Claude (independent re-run on Linux, Python 3.13, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0)
**Date:** 2026-11-09
**Scope:** mechanics, validation, data leakage and benchmarking, checked against the TID (v1.0, 7 Nov 2026). The pilot dataset was **not** modified. I regenerated it only once, in a separate scratch folder, to test reproducibility.

---

## 1. Verdict

| Area | Verdict |
|---|---|
| Mechanics (solver, closed form, eigenproblem, fields, derivatives, projection) | **Correct.** No errors found. |
| Delivered dataset `pilot_200` | **Correct.** All 600 cases independently reproduced to ~3e-13 with the original Step-1 solver. Keep it as the canonical pilot; do not regenerate. |
| Data leakage | **None found.** |
| Reproducibility | **Defect (medium).** Regenerating on a different platform changes the mixture loads of 72/200 designs (§3.1). |
| Benchmarking and statistics | Correctly caveated, but three interpretation points need stating explicitly before the next experiment (§3.2–3.4). |
| Documentation | Minor: stale body text in PROJECT_CONTEXT.md (§3.6). |

---

## 2. What I verified (completed work)

| Check | How | Result |
|---|---|---|
| Their full verification suite | `validation/run_checks.py` on a copy | mechanics **PASS** (23/23); 13 unit tests **PASS**; artifact audit **FAIL**, cause in §3.1 (not a data error) |
| Every delivered field case | Recomputed all 200 designs × 3 loads with the **original pre-upgrade** solver (`np.linalg.solve` per mode, no shared code paths), compared the w-field on the 41×41 grid | worst relative L2: sine 3.4e-13, uniform 3.4e-13, mixture 3.2e-13 |
| Scalar targets | Same independent solver | wbar worst 3.4e-13; ω̄ worst 4.7e-9 (eigensolver conditioning, §3.5) |
| Frequency branch identification | For all 1,000 retained frequencies, compared with the **lowest** eigenvalue of the same (m,n) pair | 0 mismatches |
| Closed-form thickness moments k=0,1,2 | Re-derived by hand | correct |
| Closed-form static solver (Schur complement, D* = D11 − B11²/A11) | Code read; matrix agreement 1.9e-13 in their gate | correct |
| Derivative evaluation, `field_grid` index order, `project_load` formula (4/(ab)·∫∫ q sin sin) | Code read against the TID basis definitions | correct |
| Boundary conditions | Their `test_essential_and_natural_boundary_conditions` checks v0=w0=φy=Nx=Mx=0 at x=0,a and u0=w0=φx=Ny=My=0 at y=0,b, matching the TID support definition | correct |
| Mode union count (625 + 25 − 9 = 641) | Arithmetic | correct |

### TID compliance (solver-hardening list)
Input validation ✓ · `eigh(K, M)` with symmetry and positive-definiteness checks ✓ · eigenvectors and transverse participation retained ✓ · global flexural scan with 4×4 vs 6×6 convergence ✓ · modal-load input and all five fields ✓ · vectorized closed form checked against the matrix solve ✓ · benchmark provenance with table/row/column ✓ (Zenkour's original table number still unverified, as stated).

### TID compliance (dataset protocol)
Transformed LHS ✓ · domain g∈[0,8], a/h∈[5,100], b/a∈[0.5,2] ✓ · split frozen before labels ✓ · all variants of a design in one partition ✓ · scalers and spectra from training only ✓ · selection on validation only, test evaluated once ✓ · frequencies stored once per design ✓ · no amplitude copies ✓ · smooth signed mixtures c_mn ∝ z/(m²+n²) ✓.
Not yet done (correctly listed as future work): range-corner and homogeneous verification cases inside the dataset, held-out Gaussian load family, g∈(8,10] extrapolation set, MLP/RF/GPR scalar baselines, multiple seeds, characteristic-scale field normalization.

### Data leakage audit: none found
Design-level grouping, train-only standardization, train-only SVD, validation-only model selection and a single test evaluation are all implemented as documented. Median nearest-neighbour distance to the training set (standardized transformed inputs) is 0.381 for test and 0.355 for validation, so the splits look alike. The 30 test labels have now been seen, so they should remain the **pilot** test set only. A separate locked test set is needed before the main study, as PILOT_RESULTS already says.

---

## 3. Issues found

### 3.1 Cross-platform reproducibility of design IDs and mixture loads (medium)
**Observation.** Regenerating with `run_pilot.py` on Linux/numpy 2.5.3 gives sampled parameters that differ from the delivered Windows/numpy 2.5.4 values **in the last bit** for 75 of 600 numbers (max relative difference 2.2e-16). `design_id` is a SHA-256 of the JSON of those exact floats, so **72/200 IDs change**. `design_loads` seeds its RNG from the ID, so those 72 designs get **completely different mixture loads**: mixture-field differences up to 6× in relative L2. Sine and uniform cases still match to 1e-15. `check_artifacts.py` fails at its exact `assert_array_equal` on parameters.

**Impact.** The delivered `pilot_200` is valid and internally consistent. But the statement that numerical arrays "reproduce to floating-point tolerance" holds only on an identical platform and library build. A collaborator who regenerates elsewhere silently gets a different experiment, and joins by `design_id` break.

**Fix (no regeneration of pilot_200 needed).** For the next generator version (`pilot-1.1`):
- derive the ID from canonically quantized parameters (for example, each value rounded to 12 significant digits and formatted as a fixed decimal string), or seed loads from `(seed, sample_index)`;
- make `check_artifacts.py` treat the stored `dataset.npz` parameters, IDs and loads as the source of truth, and compare regenerated parameters with a tolerance;
- reword the README and PILOT_RESULTS reproducibility claim to say "identical platform".

### 3.2 The scalar task has an exact closed form, so it cannot test H1 (interpretation, important)
The sine-load wbar is an explicit function of two thickness integrals:
wbar = 10·Ec·h³·[1/(D*·k⁴) + 1/(As·k²)], with k² = π²(1/a² + 1/b²).
Evaluating it from `plate_stiffness` reproduces all 200 targets to 6.7e-16 with **zero labels**. The fundamental frequency is a 5×5 symmetric eigenproblem. So the degree-4 polynomial result (test mean 0.68% / 0.34%) is a **pipeline sanity check**, not evidence about ML. Any "physics-informed vs data-only" comparison on this scalar task is decided in advance in favour of physics. PILOT_RESULTS avoids overclaiming; please say this explicitly as well.

### 3.3 The field problem is linear in load and diagonal in the Navier basis (interpretation, important for E2/E3)
For these supports the exact operator is W_mn = Q_mn · H_mn(g, a/h, b/a), with one 5-component transfer function per mode. Consequences:
- The mixture family's numerical rank of **25 equals the 5×5 load-basis size**. That is a property of the dataset's construction, not of the mechanics. "7 modes for 99.9%" reflects the chosen coefficient decay 1/(m²+n²) combined with the response decay ~1/k⁴.
- If a model is given the sine basis as its trunk and trained on the FSDT energy, the energy minimizer **is** the Navier solution. That "physics-informed" model would simply be the solver.
- A held-out Gaussian patch projected into the same 5×5 basis is always representable with rank 25. Generalization is guaranteed if all 25 modes are kept, and failure comes only from truncation or projection.

So the next experiment must state exactly which structure each model is given: linearity in load, the modal basis, and the energy.

### 3.4 Small test set (statistics)
With 30 test designs, 95% bootstrap intervals for the mean relative error are **0.41–1.03%** (wbar) and **0.20–0.52%** (ω̄). The 95th percentile and the worst case each rest on one or two designs. Report confidence intervals alongside point metrics, and treat the TID's target thresholds as unratified until the larger study.

### 3.5 Timing comparison and eigen conditioning (low)
- The "closed form ≈4.3× faster than matrix" result mainly measures the Python per-mode loop. On this machine a **batched** `np.linalg.solve` over the 625 prebuilt 5×5 matrices takes **0.18 ms**, against 1.70 ms for the closed form and 12.4 ms for the looped matrix path. Label it an implementation comparison, not a method comparison.
- The largest relative generalized-eigen residual is 4.7e-9, and frequencies differ across platforms by up to 4.9e-9 (thin plates, a/h ≈ 95). This is caused by the SI-unit scale disparity in K (membrane stiffness vs bending stiffness), not a bug. It is irrelevant at ML accuracy levels. If higher modes or sensitivity derivatives are needed later, nondimensionalize K and M before `eigh`.

### 3.6 Documentation (minor)
PROJECT_CONTEXT.md has a correct update banner, but the body still says "Last updated 2026-11-05 · Step 1 of 7". Sections 4.4 (research gap), 5 (title) and 6 (plan) still state the superseded novelty claim and plan. Add a decision-log entry and mark those sections superseded by the PID/TID, so a collaborator reading top to bottom is not misled.

---

## 4. Completed vs future

| Completed and verified | Future (planned, not done) |
|---|---|
| Hardened solver: validation, `eigh`, five fields and derivatives, modal and projected loads, closed form | Field surrogate (POD/reduced), PINN or energy model, DeepONet-style models |
| 200-design / 600-case pilot with frozen 140/30/30 design-level split | Locked test set, larger learning-curve pools (500/1000/2000), multiple seeds |
| Training-only shape spectra (ranks 1 / 18 / 25; 99.9%: 1 / 1 / 7) | Held-out reconstruction error and choice of rank |
| Scalar polynomial baseline (sanity check, §3.2) | MLP/RF/GPR baselines, extrapolation g∈(8,10], held-out Gaussian load family |
| Benchmark provenance (except Zenkour's original table number) | FEM verification, clamped-plate family, inverse design |
| Independent re-verification of all 600 cases (this review) | Generator `pilot-1.1` with platform-stable IDs (§3.1) |

---

## 5. Recommended next experiment: E2, reduced field baseline on the existing pilot

Uses `pilot_200` as delivered (mixture family; 140/30/30). No new data is needed.

**Step 0: prerequisites (small)**
1. Nondimensionalize the fields per design before any learning. Raw SI fields span several orders of magnitude across a/h ∈ [5,100] (w scales roughly as (a/h)³). Suggested scales: w̃ = w·Ec·h³/(qscale·a⁴), with consistent scales for u0, v0 and the rotations, fixed **before** fitting and recorded in the schema.
2. Add bootstrap CIs to `error_metrics`.
3. Fix the ID/seed derivation for future generators (§3.1). Do not touch `pilot_200`.

**Step A: projection error (no learning)**
- Bases: (i) training-only POD of the scaled five-field snapshots at ranks r ∈ {5, 7, 10, 15, 25}; (ii) the **analytic** 25-mode sine basis, which needs no data.
- Metric: per-design relative L2 reconstruction error on **validation** (median, p95, worst, with CI), for each field separately.
- Expected calibration: the analytic basis is exact for this family, so POD can only match it. This tells you whether POD adds anything here.

**Step B: learning error at a chosen rank**
Inputs are [g, a/h, b/a] plus the 25 load coefficients. Compare at **nested** training subsets of {20, 40, 70, 140} designs, with 3 seeds each:
1. **Exact operator** (closed form): the reference and the timing competitor.
2. **Black-box** data-only model (MLP or polynomial): 28 inputs → r POD coefficients.
3. **Structure-aware** model: linear in load, learning the modal transfer functions H_mn(g, a/h, b/a), i.e. 25 smooth regressions in 3 inputs.
Report projection error, learning error and total error separately, on validation only. Evaluate the test set once, at the end.

**Decision gate after E2.** If the structure-aware model (3) reaches the target accuracy (<1% median field error) with roughly 20–40 designs, the simply supported (SSSS) parametric field problem is too easy to test H1 meaningfully. In that case move the physics-vs-data comparison (E3) to a setting where the structure is **not** known in closed form:
- the **clamped-plate family** (no Navier solution; needs an independently verified solver, so start its verification early);
- loads **outside** the 5×5 basis (Gaussian patch, higher modes), where truncation matters;
- or the energy loss used as a **label-free regularizer** under scarce labels.
This gate determines whether to build the PINN/DeepONet on the SSSS family or bring the clamped extension forward.

---

## 6. How to reproduce this review
```bash
# from the review folder, with the package extracted to PKG and the original Step-1 code at ORIG
python3 -I scripts/review_independent.py PKG/mtp-fg-plate/results/pilot_200/dataset.npz   # edit sys.path line to ORIG
python3 -I scripts/review_stats.py PKG/mtp-fg-plate/results/pilot_200 PKG/mtp-fg-plate
```
`review_independent.py` hard-codes the original solver path `/home/user/mtp-fg-plate`. Point it to your copy of the Step-1 package (the `mtp-fg-plate-handoff.zip` delivered on 2026-11-05).
