# Pilot implementation and results

9 November 2026. This is an executed pilot, separate from the proposed full research programme in the PID and TID.

## What is now implemented

The working solver validates material and geometry inputs, uses the symmetric generalized eigenproblem, retains eigenvectors and transverse participation, and evaluates all five FSDT fields and their spatial derivatives. It accepts signed modal transverse loads or projects a pressure function by Gauss quadrature. The existing sine/uniform API remains usable.

A vectorized closed-form static solver returns the same five coefficients as the coupled matrix solver for this constant-Poisson-ratio material model and the specified Navier supports. It retains membrane–bending coupling. The first five flexural frequencies come from scanning half-wave pairs, not from interpreting the five eigenvalues of one pair as global modes.

## Dataset

There are **200 independent physical designs and 600 static field cases**: each design has a single sine load, a uniform load and an independently sampled smooth signed mixture of 25 sine modes. The 200 count refers to designs, following the TID. Load-amplitude copies are not counted as independent designs.

Sampling is Latin hypercube in log(1+g), log(a/h) and b/a over g in [0,8], a/h in [5,100], b/a in [0.5,2]. The computational length a is 1 m. The Al/alumina constants, constant nu=0.3, ks=5/6, power law and linear FSDT assumptions are fixed. The mixture coefficient vector has unit Euclidean norm, with qscale=1 Pa; signed mixtures are not guaranteed to represent one-sided pressure.

The frozen split has **140 training, 30 validation and 30 test designs** (420/90/90 field cases). All load variants and grid points from a design stay in one partition. Scalers and shape spectra use training data only. Model choice uses validation only. No refitting with test labels was performed.

Each case stores all five fields on a 41 by 41 normalized grid, plus modal coefficients for evaluation at arbitrary points. Frequencies and mode identities are stored once per design. The archive has 641 distinct mode pairs: the union of the 625 odd-odd uniform modes up to index 49 and the 25 mixture modes up to index 5. The data dictionary is in DATASET_SCHEMA.md.

## Numerical evidence

| Check | Observed result |
| --- | --- |
| Existing mechanics checks, including independent symbolic Ritz | 23 / 23 passed |
| New solver and sampling/model tests | 13 / 13 passed |
| Matrix versus closed form, worst component-relative error across 10 designs and 3 loads | 1.91e-13 |
| Generalized eigen residual, worst across 200 designs at (1,1) | 4.73e-9 |
| Mass orthogonality error | 1.89e-15 |
| Uniform w field, largest relative difference between cutoffs 49 and 99 | 2.70e-5 (0.00270%) |
| Uniform centre w, largest relative difference | 7.61e-6 |
| Largest uniform difference across other four fields | 1.07e-5 |
| First five frequencies: scan 4 by 4 versus 6 by 6 | Identical retained frequencies for all 200 designs |
| Smallest transverse participation among retained modes | 0.8878 |

Uniform field comparisons use the stored grid. They establish neither continuous maximum error nor stress/derivative convergence. The eigen residual is ||Kv-lambda Mv||/(||Kv||+||lambda Mv||), evaluated columnwise. The symbolic/adaptive quadrature routines emit a roundoff warning at their very stringent tolerance; the warning is preserved in the logs and their numerical comparisons pass.

Published benchmark numbers now have explicit table/row/column references in validation/benchmark_references.json. FG values are verified reproductions of Zenkour's results in Nguyen et al. (2019), Table 4. Their original Zenkour table number remains unverified. The isotropic 0.0930/0.0932 pair is traced to Shimpi and Patel (2006), Table 1, p.995. Comparisons with higher-order theories and 3D elasticity are labelled as model comparisons. The thin uniform reference is independently evaluated from its CPT double series.

## First ML result

The model predicts two positive scalar quantities: sine-load wbar = 10 Ec h^3 wcentre/(qscale a^4), and omegabar = omega1 a^2/h sqrt(rhoc/Ec). Inputs are [g,a/h,b/a]. A training-only standardization is applied to the transformed coordinates and the model fits logarithms of the outputs.

Twelve polynomial candidates (degrees 1–4 and three regularization levels) and four thin-plate-spline RBF candidates were evaluated on validation data. The selected model is **degree-4 polynomial ridge regression, regularization 1e-8**, with 35 polynomial terms per output. The selection score is mean relative validation error averaged across the two outputs (0.4223%).

| Held-out relative error | Sine deflection | Fundamental frequency |
| --- | --- | --- |
| Mean | 0.676% | 0.338% |
| Median | 0.448% | 0.227% |
| 95th percentile | 2.069% | 1.060% |
| Worst | 4.471% | 2.273% |
| Constant training-log-mean baseline: mean | 69.343% | 34.789% |

95% bootstrap intervals for the test means are 0.41–1.03% (deflection) and 0.20–0.52% (frequency). The sine-load deflection also has an exact closed form from two thickness integrals (reproducing all 200 targets to 7e-16 with no labels), so this scalar model is a pipeline check, not evidence about ML or physics. These are interpolation results from one seed and 30 held-out designs. They are not evidence of arbitrary-load field prediction, extrapolation, independent physical accuracy or a general speed advantage for ML.

## How many shapes does the pilot contain?

Shape spectra use **uncentred, unit-norm deflection snapshots on normalized coordinates**, separately by load family, and only the 140 training designs. Numerical rank uses s_i/s_1 > 1e-10. The percentage criterion uses squared singular values (snapshot norm, not mechanical energy).

| Load family | Numerical rank | Modes for 99% | Modes for 99.9% |
| --- | --- | --- | --- |
| Single sine | 1 | 1 | 1 |
| Uniform | 18 | 1 | 1 |
| Smooth mixtures | 25 | 4 | 7 |

Uniform responses have small shape variations but their first component dominates this aggregate criterion. Seven mixture modes do not guarantee a small error in every individual design, at every point, or in derivatives. The next field experiment must measure held-out reconstruction error before selecting its rank.

## Timing

For one representative pilot design under the same 625-mode uniform load, ten repetitions after two warmups in the delivered run gave median coefficient times of 11.23 ms (matrix) and 1.65 ms (closed form). Including reconstruction of all five 41 by 41 fields gave 13.47 ms and 3.12 ms, respectively: approximately **4.3 times faster** for that complete static output. *(Review note: this mainly measures the Python per-mode loop; a batched `np.linalg.solve` over the 625 prebuilt 5×5 matrices took 0.18 ms on the review machine. Treat it as an implementation comparison, not a method comparison.)* An earlier complete run gave 16.28 ms and 2.85 ms (5.7 times), illustrating ordinary local timing variation. Loading and geometry construction were outside these timers; coefficient validation, solving and field reconstruction were inside. These are local CPU measurements, not universal timings. Dataset generation with convergence checks and compression took about 9.6 s in the delivered run. ML versus direct-solver timing is not claimed.

The full dataset was regenerated after tightening rejection of complex-valued callable pressures. Every numeric and identity array reproduced exactly **on the same platform and library build**. *(Correction, 9 Nov review: on Linux/numpy 2.5.3, last-bit differences in 75 of 600 sampled floats changed 72 of 200 pilot-1.0 design IDs and therefore their mixture loads; sine and uniform cases matched to 1e-15. The delivered dataset is valid and is the source of truth; the audit now re-derives IDs and loads from stored parameters, and new data uses generator pilot-1.1.)* The saved artifact audit also checks reconstruction against fresh matrix solves and verifies the persisted model's reported errors.

## Next research step

*Update 9 Nov: the reduced field baseline below has been executed as E2 — see [E2_RESULTS.md](E2_RESULTS.md). Its recommended next step is E3a (physics-informed vs data-only transfer learning at matched capacity).*


Build a reduced field baseline on the varied-load dataset, select its rank using validation reconstruction errors, and compare its learned coefficient map against the direct closed form. Preserve the current test partition and avoid tuning to its reported errors. Before a final publication comparison, create a separate locked test set and repeat training seeds. Add an unseen Gaussian load family, separate projection error from learning error, and test g in (8,10] only as an explicitly held-out extrapolation study.

The later energy-informed model should learn all five coupled fields or a mathematically justified condensation. FEM comparisons, boundary-condition extensions, physical pressure cases, inverse design and manufacturing constraints remain future work. The current pilot gives no reason to promise novelty or accuracy gains from a large neural network solely on the scalar sine-load task.

## Where to look

- results/pilot_200/manifest.json: complete numerical metrics, model candidates, runtime and source checksums.
- results/pilot_200/dataset.npz: generated data; load with allow_pickle=False.
- results/pilot_200/scalar_model.npz: reusable saved baseline.
- results/pilot_200/shape_rank.png and sample_load_responses.png: shape evidence.
- results/pilot_200/scalar_test_predictions.png: held-out scalar predictions.
- evidence/: validation logs and saved-artifact audit (evidence/2026-11-09_delivery_windows/ keeps the original delivery logs).
- docs/PILOT_REVIEW_2026-11-09.md: independent review; docs/E2_RESULTS.md: E2.

The original source_handoff directory in the parent workspace remains unchanged. The original PID/TID are dated planning snapshots; this report records the implemented subset.
