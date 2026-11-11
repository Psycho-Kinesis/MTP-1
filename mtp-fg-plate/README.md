# Functionally graded plate MTP — working pilot

The solver upgrades, the 200-design pilot, experiment E2 (reduced field baseline) experiment E3a (physics-informed vs data-only at matched capacity) experiment E3b (imperfect FSDT physics + higher-fidelity TSDT labels) experiment E3c (physics-structured correction and label placement), the pre-registered confirmation E4, E5 (improved label-free base model, confirmed on a second fresh test set) H2 (generalization to unseen load shapes, pre-registered) and H3 (surrogate-guided inverse design verified with TSDT, pre-registered) are implemented. Start with [the E5 report](docs/E5_RESULTS.md) for the final confirmed result, [the H2 report](docs/H2_RESULTS.md) for unseen loads and [the H3 report](docs/H3_RESULTS.md) for inverse design, then [the E4 report](docs/E4_RESULTS.md), [the pilot report](docs/PILOT_RESULTS.md), [the E2 report](docs/E2_RESULTS.md), [the E3a report](docs/E3A_RESULTS.md), [the E3b report](docs/E3B_RESULTS.md), [the E3c report](docs/E3C_RESULTS.md) and [the data dictionary](docs/DATASET_SCHEMA.md). The independent review of the pilot is in [docs/PILOT_REVIEW_2026-11-09.md](docs/PILOT_REVIEW_2026-11-09.md). The original handoff is preserved separately in the parent workspace's source_handoff directory.

## Delivered results

- 200 physical designs, each with sine, uniform and smooth mixed loading: 600 five-field solutions.
- Frozen design-level split: 140 training / 30 validation / 30 test designs.
- Symmetric generalized eigenproblem; retained vectors and transverse participation; converged first-five flexural-mode scans within the pilot domain.
- Matrix and vectorized closed-form static solvers, all five fields and analytical spatial derivatives.
- Saved scalar polynomial model: held-out mean errors 0.676% for sine deflection and 0.338% for fundamental frequency.
- Training shape spectra: sine rank 1; varied loads need 7 modes for 99.9% of snapshot norm.
- Traceable benchmark table locations, test logs, dataset/model files and checksums.
- E2 reduced field baseline on the mixture family: projection study (POD needs rank 60 for p95 ≤ 1%; the analytic 25-mode basis is exact) and learning curves for black-box, linear-in-load and modal-transfer models at 20–140 designs and 3 seeds. Best: modal transfer, median 0.44% / p95 3.6% (validation) at 140 designs; targets not met at ≤ 40 designs. See docs/E2_RESULTS.md.
- Platform-stable generator `pilot-1.1` for new datasets; audit of `pilot_200` works on any platform.
- E3a: one polynomial model class trained three ways. Data only needs 140 labels to reach median ≤ 1% / p95 ≤ 3%; the hybrid needs 10; physics only (FSDT energy at unlabelled designs) needs 0 labels. H1 is supported at N = 10–70 (paired, 30/30 designs at N ≤ 40). The physics is the same model as the labels, so model-form error is not tested there. See docs/E3A_RESULTS.md.
- TSDT (Reddy third-order) Navier solver, verified against FSDT (3e-16 in the limit), an independent symbolic Ritz TSDT derivation (9e-16) and published higher-order values.
- TSDT free vibration (consistent mass matrix, fundamental frequency) and vectorized batch solvers, verified against the FSDT limit (exact), the symbolic Ritz TSDT derivation (6e-14) and published TSDT frequencies (4 digits).
- E3b: FSDT physics with TSDT labels. With 10–40 labels, physics-informed arms beat data only on 28–30 of 30 designs (Q2 supported; e.g. 10 labels: 18% → 0.78% median). But a few random high-fidelity labels do not remove the FSDT bias (Q1 not supported at N ≤ 40); every arm needs about 140 labels for p95 ≤ 3%. See docs/E3B_RESULTS.md.
- E3c: a correction structured by the physics (H_FSDT × (k h)² × polynomial, which vanishes for thin plates) with 10 TSDT labels beats both physics only and data only (pre-registered Q1 and Q2 supported at N = 10, 20, 40). Validation 0.50% / 2.22% at 10 labels, against 140 labels for data only. On test (M-S) the median target holds (0.33–0.48%), but p95 is 3.1–4.0%, narrowly above 3%. The correction form is decisive; shear-guided label placement adds little. See docs/E3C_RESULTS.md.
- **E4 (pre-registered confirmation on 200 fresh, never-seen designs):** all four confirmatory questions confirmed. With 10 TSDT labels, M-S beats physics only (on ~70% of designs) and data only (on 100%), with median 0.53% and p95 2.85%. The tail result is marginal (CI 2.05–3.88%) and not robust across budgets; the remaining tail comes from the label-free base model in the narrow / near-uniform corner, not from the physics. See docs/E4_RESULTS.md.
- **E5 (final, confirmed on 200 new fresh designs):** a label-free FSDT surrogate with classical-plate scaling (reproduces FSDT to 0.03% / 0.11%), plus the (k h)² correction from **10** shear-guided TSDT labels, gives **median 0.062%, p95 0.33%** (p95 CI 0.20–0.49%). All six pre-registered questions are confirmed: it beats the old pipeline on 200/200 designs and data only at every budget. The base-selection rule was revised after development (documented), which is why a second fresh set was used. See docs/E5_RESULTS.md.
- **H2 (unseen load shapes, pre-registered, 200 fresh designs):** the E5 pipeline extended from 5×5 to 15×15 modes, with E5's hyperparameters frozen and the **same 10** labelled designs, gives median 0.061–0.070% and p95 0.33–0.40% on uniform pressure, wide and narrow Gaussian patches (TSDT 25×25 reference, exact modal L2 metric). All five questions are confirmed. The frozen 5×5 pipeline sits at its modal truncation floor on narrow patches (median 1.47%, p95 3.14%); extending the basis removes it on 200/200 designs. See docs/H2_RESULTS.md.
- **H3 (inverse design, pre-registered, 100 random problems + a main problem):** minimum-mass FG plates under deflection, frequency and ceramic-use limits, optimized with the surrogate (10 TSDT analyses) and re-checked with TSDT. Designs are within 0.14% of the best possible mass and 0.23% of the limits (p95); one verification analysis (two in 38% of runs) makes every design TSDT-feasible. Total cost: 140–153 TSDT analyses against 7,938 for direct TSDT optimization (1.9%). Physics only violates limits by up to 2% on thick, metal-rich plates; data only is unusable. All five questions confirmed. See docs/H3_RESULTS.md.
- **FG Plate Lab (interactive web app, `demo/`):** the FSDT/TSDT solvers and the frozen H3 model running in the browser, in four views: plate simulator, design studio (with TSDT check and exact optimum), evidence, and method. Tested against the Python code. Open `demo/fg_plate_lab_offline.html` in any browser, or run the repository's `streamlit_app.py`.

These are results within linear, constant-nu FSDT with the specified mixed Navier supports. They are not experimental validation. Physics-informed training is implemented only for polynomial transfer functions (E3a–E5); no neural PINN or FEM extension is implemented. The inverse-design study (H3) covers uniform pressure on simply supported plates, with TSDT as the reference.

## Run

Python 3.12 is the tested interpreter. From this project directory:

```bash
python -m pip install -r requirements-tested.txt
python validation/run_checks.py
python experiments/run_pilot.py --output results/pilot_reproduction   # new data: generator pilot-1.1
python experiments/run_e2.py --output results/e2_reproduction
python experiments/run_e3a.py --output results/e3a_reproduction
python experiments/run_e3b.py --output results/e3b_reproduction
python experiments/run_e3c.py --output results/e3c_reproduction
python experiments/run_e4.py --output results/e4_reproduction     # asserts the pre-registration hash
python experiments/run_e5.py --output results/e5_reproduction
python experiments/run_e5_confirm.py --output results/e5c_reproduction   # asserts the E5 pre-registration hash
python experiments/run_h2.py --output results/h2_reproduction            # asserts the H2 pre-registration hash (~4 min)
python experiments/plot_h2.py                                          # redraws figures/h2_load_families_summary.png
python experiments/run_h3_dev.py --output results/h3_dev_reproduction  # H3 development (pilot + FSDT only, ~2 min)
python experiments/run_h3.py --output results/h3_reproduction         # asserts the H3 pre-registration hash (~60 min)
```

The delivered results are already in results/pilot_200 (generator pilot-1.0). The generator refuses to overwrite a nonempty experiment directory. Default count is 200 and seed is 20261109. It writes a fresh split before computing labels.

**Reproducibility.** `results/pilot_200/dataset.npz` is the source of truth; `validation/check_artifacts.py` audits it on any platform by re-deriving design IDs and loads from the *stored* parameters. Regenerating pilot_200 with `--generator-version pilot-1.0` reproduces it exactly only on an identical platform and library build: under pilot-1.0, last-bit differences in the sampled floats change design IDs and therefore the mixture loads (72 of 200 designs on Linux/numpy 2.5.3 vs the delivered Windows/numpy 2.5.4). New data uses `pilot-1.1`: parameters quantized to 10 significant digits, IDs from canonical decimal strings, mixture loads seeded by (seed, sample index). Timestamps, timings and archive bytes may vary. The generating source of pilot_200 is archived in provenance/pilot-1.0/.

scikit-learn is needed only for E2's MLP baseline.

**BLAS threads.** The reproduction assertions (E5 arms within relative 1e-9) are sensitive to the BLAS thread count, which changes floating-point summation order. With `OPENBLAS_NUM_THREADS=2` on the 4-core test machine, one non-selected E5 arm (M-S@P0, 10 labels, seed 11) moved beyond that tolerance and run_h3.py stopped before evaluating any design. All delivered results use the default thread count; do not run two experiment scripts at once with reduced threads.

requirements.txt lists unpinned main dependencies for a fresh compatible installation; requirements-tested.txt records the exact versions used here. The full symbolic Ritz check takes longer than ordinary tests and requires sympy. Strict quadrature roundoff warnings are retained in evidence logs.

## Solver example

```python
import numpy as np
from fgplate import NavierFSDTPlate, al_al2o3

p = NavierFSDTPlate(a=1, b=1.2, h=.1, material=al_al2o3(2))
solution = p.bending_modal({(1,1): 1.0, (2,3): -.2}, method='closed_form')
X, Y, fields = solution.field_grid()  # u0, v0, w0, phix, phiy
at_point = solution.fields(.3, .4)
dx_fields = solution.fields(.3, .4, derivative=(1,0))

pressure = lambda x,y: np.exp(-((x-.4)**2+(y-.7)**2)/.02)
Q = p.project_load(pressure, max_mode=5, quadrature_order=64)
patch_solution = p.bending_modal(Q, method='closed_form')
# Check quadrature and modal-cutoff convergence for this new pressure family.

eigenpairs = p.eigenpairs(1,1)  # five branches for ONE half-wave pair
first_five = p.flexural_modes(max_mode=6, count=5)
# Increase the scan and compare before claiming global convergence elsewhere.
```

Static pressure and moduli use Pa; geometry/displacements use m; density uses kg/m3. Rotations are dimensionless and frequencies are rad/s. Divide by 2*pi for Hz. Closed form is valid only for the implemented constant-nu constitutive model and these supports. For uniform loads n_terms=49 means 625 active odd-odd pairs.

## Files

| Path | Purpose |
| --- | --- |
| fgplate/material.py | Validated power-law FGM, exact thickness moments |
| fgplate/navier.py | Matrix/closed-form static solution, five fields, projection and eigenpairs |
| fgplate/pilot.py | Deterministic Latin hypercube designs, load realization and frozen splits |
| fgplate/baseline.py | Polynomial/RBF scalar regression, error metrics with bootstrap intervals, serialization |
| fgplate/fields.py | Field nondimensionalization, analytic Navier basis, POD |
| fgplate/transfer.py | Modal transfer model: Legendre features (linear or log b/a), scaled mode matrices, classical-plate prefactor, data/physics/hybrid quadratic fits with optional prefactors (ridge scope 'mode' or 'block') |
| fgplate/tsdt.py | Per-mode stiffness and consistent mass for FSDT and Reddy TSDT via exact moments up to z^6; exact scaled transfer functions; natural frequencies; vectorized batch versions |
| experiments/run_pilot.py | Dataset generation, quality gates, plots, model selection and metrics |
| experiments/run_e2.py | E2 projection study and learning curves on the pilot (read-only) |
| experiments/run_e3a.py | E3a data vs physics vs hybrid learning curves, H1 test and capacity study (read-only pilot) |
| experiments/run_e3b.py | E3b FSDT physics + TSDT labels: physics only, data only, hybrid, physics + correction; Q1–Q3 |
| experiments/run_e3c.py | E3c 3×2 correction form × label placement on top of physics only; asserts E3b reproduction; pre-registered Q1/Q2 |
| experiments/run_e4.py | E4 confirmation: refits frozen E3c arms (asserted identical), then generates and evaluates a fresh locked test set once |
| experiments/run_e5.py | E5 development: four label-free base variants, base-choice rules v1 and v2, corrections on every base (pilot train/validation only) |
| experiments/run_e5_confirm.py | E5 confirmation: refits frozen E5 arms (asserted identical), generates a second fresh locked set, evaluates once |
| experiments/run_h2.py | H2: refits frozen E5 arms (asserted identical), extends them to 15×15 modes (5×5 part asserted identical), generates 200 fresh designs with four load families, evaluates once |
| experiments/plot_h2.py | Presentation figure of H2, drawn from the saved manifest (no recomputation) |
| experiments/h3_common.py | H3 shared code: frozen surrogates (static centre deflection, frequency heads), TSDT truth, design optimizer, closed loop, direct-SLSQP competitor, problem generator |
| experiments/run_h3_dev.py | H3 development on pilot data and FSDT only: frequency-head selection, accuracy, truth truncation, main-problem FSDT scan |
| experiments/run_h3.py | H3 inverse design: Part A Pareto front of the main problem, Part B 100 random problems, TSDT verification, decisions Q1–Q5 |
| validation/benchmark_references.json | Numeric reference provenance with verified table/row/column locations |
| validation/test_extensions.py | Mechanics/input/eigen/field/derivative checks |
| validation/test_pilot.py | Sampling, group separation, platform-stable IDs, training-only scaling and persistence checks |
| validation/test_fields.py | Field scaling, analytic/POD bases, bootstrap intervals |
| validation/test_transfer.py | Transfer model: mode matrices vs solver, label-free exact recovery, weight invariance, ridge-scope equivariance |
| validation/test_tsdt.py | TSDT: FSDT limit (stiffness and mass), positive definiteness, published deflections and frequencies, thin/thick behaviour, scaling, batch = scalar |
| validation/check_artifacts.py | Saved checksums, reconstruction and model-output audit |
| validation/validate.py | Original 23 mechanics checks with corrected reference attribution |
| validation/ritz_crosscheck.py | Independent symbolic energy derivation |
| results/pilot_200/ | Generated dataset, model, plots, split and manifest (frozen; pilot-1.0) |
| results/e2_reduced_field/, results/e2b_log_transfer/ | E2 metrics, figures, per-design errors, manifests |
| results/e3a_transfer/ | E3a metrics, decisions, figures, per-design errors, collocation designs, manifest |
| results/e3b_model_form/ | E3b metrics, decisions, figures, TSDT/FSDT reference transfer functions, manifest |
| results/e3c_structured/ | E3c metrics, pre-registration diagnostics, decisions, label placements, figures, manifest |
| results/e4_confirmation/ | E4 fresh test set (designs, loads, TSDT/FSDT labels), per-design errors, decisions C1–C4, figures, manifest |
| results/e5_base_model/, results/e5_base_model_v2/ | E5 development (rule v1 record; rules v1 + v2), validation errors, figure, manifest |
| results/e5_confirmation/ | E5 second fresh test set, per-design errors, decisions C1–C6, figures, manifest |
| results/h2_loads/ | H2 fresh test set (designs, four load families, TSDT 25×25 truth), per-design errors, decisions Q1–Q5, figures, manifest |
| figures/h2_load_families_summary.png | H2 summary figure (median and p95 per load family and arm) |
| results/h3_development/ | H3 development record (frequency heads, validation accuracy, truncation, FSDT scan figure, manifest) |
| results/h3_inverse_design/ | H3 results: Part A and Part B records (every design, TSDT verification, closed loops, direct SLSQP), errors, decisions, figures, manifest |
| provenance/pilot-1.0/ | Exact generator source that produced pilot_200 |
| evidence/ | Check logs and pass/fail summary |
| docs/PILOT_RESULTS.md | Interpretation, timings, limitations and next research step |
| docs/E2_RESULTS.md | E2 results, decision gate and recommended E3a |
| docs/E3A_RESULTS.md | E3a results, H1 test, capacity study, caveats and recommended E3b |
| docs/E3B_RESULTS.md | E3b results, Q1–Q3, mechanism, caveats and recommended E3c |
| docs/E3C_RESULTS.md | E3c results, primary hypotheses, factor contrasts, caveats and recommended confirmation run (E4) |
| docs/E4_PREREGISTRATION.md | E4 pre-registration, written before the fresh data existed (hash asserted by run_e4.py) |
| docs/E4_RESULTS.md | E4 confirmatory results, limitations and recommended E5 |
| docs/E5_PREREGISTRATION.md | E5 pre-registration (hash asserted by run_e5_confirm.py) |
| docs/E5_RESULTS.md | E5 development and confirmation: final pipeline, caveats, next steps |
| docs/H2_PREREGISTRATION.md | H2 pre-registration (hash asserted by run_h2.py) |
| docs/H2_RESULTS.md | H2 results: unseen load shapes, truncation floor, caveats, next steps |
| docs/H3_PREREGISTRATION.md | H3 pre-registration (hash asserted by run_h3.py) |
| docs/H3_RESULTS.md | H3 results: verified inverse design, Pareto front, cost, caveats |
| docs/PILOT_REVIEW_2026-11-09.md | Independent review of the pilot |
| docs/PROJECT_CONTEXT.md | Historical Claude context with a dated implementation update |

Published FG comparison values are verified in [Nguyen et al., Table 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC6695951/), which reproduces Zenkour's results. The original Zenkour table number is still unverified. Full status and the isotropic frequency source are in benchmark_references.json. Similar results from different plate theories must not be presented as identical-theory verification.
