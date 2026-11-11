# E2: reduced field baseline (executed)

9 November 2026 · Uses `results/pilot_200` exactly as delivered (read-only; checksum asserted on load).
Results: `results/e2_reduced_field/` (pre-declared families) and `results/e2b_log_transfer/` (one post-hoc variant).
Command: `python experiments/run_e2.py --output <new dir>`, then `--families B3,B3L` for the variant. Both runs are deterministic: a complete rerun reproduced every validation and test metric and every model selection exactly.

## What E2 tests
Can a model predict the **full five-field response** (u0, v0, w0, φx, φy on the 41×41 grid) of a simply supported FG plate under an arbitrary smooth load from the 25-mode mixture family, given the design [g, a/h, b/a] and the 25 load coefficients? How much does the answer depend on the **structure the model is given**?

| Family | Structure given to the model | Basis |
|---|---|---|
| B0 exact | Everything (closed form) | — |
| B1 black-box MLP | None. 28 inputs → POD coefficients (3×64 tanh, Adam 1e-3, as in the TID pilot config) | training POD |
| B2 linear-in-load | Response is linear in the load: coefficients = Σ_k c_k·G_k(x), with G_k polynomial in x | training POD |
| B3 modal transfer | Linear in load **and** diagonal in the Navier basis: learns 125 transfer functions H(x), each smooth in 3 inputs | analytic 25-mode basis per field |
| B3L (post hoc) | As B3, but regresses log\|H\| (every component keeps one sign) | analytic |

**Data and protocol:**
- **Data:** smooth-mixture family only, split 140 / 30 / 30 designs as frozen in pilot_200.
- **Scaling:** fields are made dimensionless before any fitting. w0 uses qa⁴/(E_c h³), φ uses qa³/(E_c h³), and u0, v0 use qa³/(E_c h²). Raw fields span 4–5 orders of magnitude across a/h ∈ [5, 100]; scaled fields are O(1).
- **Metric:** per-design relative L2 over all five scaled fields.
- **Budgets and seeds:** nested training subsets of {20, 40, 70, 140} designs, for seeds 11, 22 and 33. At every budget, POD bases and scalers come from that subset only.
- **Selection:** hyperparameters are chosen on validation only. The test set is evaluated once, after all choices were frozen.

## Step A: projection error (no learning)
Validation designs; POD is built from the 140 training designs.

| Basis | Median % | p95 % | Worst % |
|---|---|---|---|
| POD rank 5 | 7.01 | 26.5 | 29.0 |
| POD rank 10 | 2.77 | 12.7 | 15.6 |
| POD rank 25 | 0.72 | 5.01 | 7.04 |
| POD rank 40 | 0.28 | 1.59 | 2.06 |
| **POD rank 60 (chosen: smallest with p95 ≤ 1%)** | **0.10** | **0.65** | **0.73** |
| POD rank 100 | 0.013 | 0.074 | 0.107 |
| Analytic Navier basis, 25 modes per field | — | 1.7e-13 | — |

The scaled five-field training snapshots have numerical rank **125 = 25 load modes × 5 fields**. In each mode, the ratios between the five fields change with the design, so a single data-driven basis needs far more vectors than the 25 load modes. The pilot's earlier "7 modes for 99.9%" referred to w alone and to snapshot norm. It is not a sufficient rank for field learning.

## Step B: learning curves
Values are the mean over 3 seeds; validation median shows the min–max over seeds in parentheses. Test results come from the single final evaluation.

| Family | Designs | Validation median % | Validation p95 % | Test median % | Test p95 % |
|---|---|---|---|---|---|
| B1 MLP | 20 | 39.6 (33.6–45.8) | 250 | 47.5 | 171 |
| B1 MLP | 40 | 54.0 (45.2–61.3) | 411 | 51.9 | 273 |
| B1 MLP | 70 | 50.2 (44.5–54.3) | 381 | 58.9 | 289 |
| B1 MLP | 140 | 35.0 (31.3–39.1) | 185 | 37.5 | 219 |
| B2 linear-in-load | 20 | 21.4 (15.4–25.8) | 138 | 33.8 | 151 |
| B2 linear-in-load | 40 | 18.2 (16.9–19.1) | 208 | 30.8 | 140 |
| B2 linear-in-load | 70 | 14.4 (12.9–15.9) | 89 | 21.7 | 83 |
| B2 linear-in-load | 140 | 6.68 | 70 | 12.2 | 40 |
| B3 modal transfer | 20 | 5.77 (4.77–6.54) | 48 | 6.64 | 29 |
| B3 modal transfer | 40 | 1.39 (1.02–1.84) | 15.0 | 2.04 | 20.5 |
| B3 modal transfer | 70 | 0.62 (0.56–0.68) | 5.1 | 0.62 | 5.4 |
| B3 modal transfer | 140 | **0.44** | **3.6** | **0.57** | **5.1** |
| B3L (post hoc) | 20 | 6.55 (1.85–9.43) | 23.7 | 6.33 | 19.1 |
| B3L (post hoc) | 40 | 1.99 (0.71–3.31) | 6.6 | 3.16 | 16.0 |
| B3L (post hoc) | 70 | 1.33 (0.88–2.03) | 4.1 | 1.48 | 5.4 |
| B3L (post hoc) | 140 | 1.45 | 3.6 | 1.46 | 4.6 |

At 140 designs all seeds use the same subset, so the deterministic models (B2, B3, B3L) show no seed spread. Four of the 36 MLP fits reached the 3,000-epoch limit; the convergence warnings are recorded in the manifest.

### What this shows
1. **Structure matters far more than the network.** Each step of structure given to the model cuts the error by an order of magnitude at the same label budget: B1 → B2 → B3 goes from ~35–54% to ~7–21% to ~0.4–6% median. With 140 designs or fewer, a generic MLP on 28 inputs does not learn this map at all; its error does not fall with more data.
2. **For B1 and B2, almost all the error is learning error, not basis error.** At 140 designs the projection error is 0.10% (median), against learning errors of 31% (B1) and 6.7% (B2).
3. **The remaining difficulty is regression capacity, not the field representation.** B3 represents the fields exactly. Its error comes from fitting 125 smooth three-parameter transfer functions that vary by factors of about 400 (w) and up to 9×10⁴ (u0 and v0, near uniform plates where g → 0). Its tail errors cluster at **narrow plates (b/a ≈ 0.5)**, where the higher-mode transfer functions vary fastest. Regressing log\|H\| (B3L) halves the 95th percentile at 20 and 40 designs but worsens the median at 140 designs. Neither polynomial variant reaches the targets.

## Decision gate (rule fixed before running)
**Rule:** a structure-aware model meets median ≤ 1% and p95 ≤ 3% on validation (mean over seeds) with at most 40 designs.
**Outcome: not met.** B3 and B3L do not meet the p95 target at **any** budget up to 140 designs. B3's median target is met from 70 designs.
**Recorded decision:** the simply supported family is *not* too easy; it remains a usable testbed.

**Important qualification for designing E3.** This outcome concerns learning from labels. For these supports, a physics-informed model that uses the Navier basis can obtain every transfer function **exactly and without labels**, because minimizing the FSDT energy mode by mode is the 5×5 solve. A physics-vs-data comparison on this family is therefore meaningful only if both models share the **same non-exact model class** (for example the same polynomial or network for H(x)), with physics entering as a loss at unlabelled collocation designs, not as the exact per-design solve.

## Timing (local CPU, illustrative; no speed claim)
For the same output (five scaled 41×41 fields per design), the learned models take about 0.01–0.11 ms per design in a batch of 30. The exact closed form takes about 0.3–0.5 ms per design when called one design at a time, and that includes plate construction. The closed form was not vectorized across designs, so this is not a like-for-like speed comparison. Timings varied by about 1.6× between two identical runs.

## Limitations
- One pilot dataset and one load family (signed, smooth, 5×5).
- The 30 validation and 30 test designs give coarse tails: each p95 rests on one or two designs.
- The pilot test set had already been viewed for the scalar task. It remains a *pilot* test set; a locked test set is still needed before the main study.
- The polynomial model classes have limited capacity. MLP hyperparameters were not tuned beyond the L2 penalty.
- No Gaussian or out-of-basis loads, no extrapolation in g, no FEM.

## Recommended next experiment: E3a, physics-informed vs data-only transfer learning at matched capacity
This can run on the existing pilot plus unlabelled collocation designs, entirely on CPU with linear algebra:
- **Model class:** shared by all arms. H(x; θ) with polynomial features in x (degrees 2–4; optionally log targets as in B3L, or a small network later).
- **Arm D (data only):** least squares on labelled designs, label budgets {10, 20, 40, 70, 140}.
- **Arm P (physics only):** no labels. Minimize Σ_collocation Σ_modes [½ Hᵀ K_mn H − Hᵀ f_mn] at M sampled designs, with K_mn from the solver. Because H is linear in θ for polynomial features, this energy is **quadratic in θ**, so the fit is a closed-form linear solve. Collocation designs must exclude validation and test designs.
- **Arm H (hybrid):** data loss plus a weighted energy term; the weight is chosen on validation.
- **Metrics:** the same field errors as E2, plus learning curves against labels and against collocation count.

This directly tests H1 (fewer labels for the same accuracy) with matched capacity. It will also show whether the capacity limit seen in E2 sets the floor: if physics-only stalls at the same p95 as B3 with 140 labels, the bottleneck is model capacity, which motivates a richer model class next.

**In parallel:** start verifying an independent solver for the clamped family, such as a Ritz or FE model. That is the setting where no closed-form structure exists, and it takes the longest to make ready.
