# H3 pre-registration: surrogate-guided inverse design

Written 9 November 2026, after development on **pilot data and the FSDT solver only** (`experiments/run_h3_dev.py`, `results/h3_development/`). At that point no TSDT solution existed for any design-problem design. `experiments/run_h3.py` asserts the sha256 of this file.

## Question (PID hypothesis H3)
Does optimization with the confirmed surrogate reach designs that are feasible when re-checked with the high-fidelity solver (TSDT), at competitive total computational cost?

## What is already fixed (development, pilot and FSDT only)
**Surrogates.** All are rebuilt deterministically. The E5 arms are asserted identical, and the static arms reproduce H2's per-design uniform-pressure errors exactly.

| Arm | Static: centre deflection under uniform pressure | Fundamental frequency |
|---|---|---|
| **C (primary)** | H2-extended M-S@P2 (15×15 modes), **10** shear-guided TSDT-labelled designs | exact FSDT × √(1 + (k₁₁h)² p(x)); p fitted on the **same 10** designs, degree and ridge selected on the 30 pilot validation designs |
| P (physics only) | label-free base P-ext (15×15) | exact FSDT (no labels) |
| D (data only) | D-ext (15×15), 10 random TSDT-labelled designs | polynomial in log frequency on the **same 10** random designs, selected on pilot validation |

- One TSDT analysis of a labelled design gives both its static and its modal labels, so arm C uses **10 high-fidelity analyses** in total.
- Seeds 11, 22, 33 (the label sets of E5/H2).
- **Accuracy on the pilot validation designs** (development):

  | Arm | Centre deflection (median / p95) | Frequency (median / p95) |
  |---|---|---|
  | C | 0.04–0.05% / 0.19–0.34% | 0.017–0.020% / 0.09–0.12% |
  | P | 0.06–0.09% / 1.2% | 0.016% / 0.63% |
  | D | 11–30% / 82–197% | 5–7% / 11–16% |

**Truth.**
- Deflection: TSDT Navier, odd modes m, n ≤ 49. On pilot training designs this is within 1.3e-6 (relative) of 149 modes.
- Frequency: TSDT (1,1) eigenproblem with the new consistent TSDT mass matrix. This matrix is verified against the FSDT limit (exact), an independent symbolic Ritz derivation (6e-14) and published TSDT values (Belkhodja et al. 2020, Table 5, quoting Hosseini-Hashemi et al.; 4 digits).
- The maximum deflection under uniform pressure is at the plate centre by symmetry; this is checked on a 41×41 grid at the final designs.

## Design problem (TID)
Material Al/Al₂O₃ (power law), simply supported rectangular plate a × b, uniform service pressure q₀. Design variables: thickness h and gradient index g, within the validated domain 5 ≤ a/h ≤ 100 and 0 ≤ g ≤ 8.

> minimize areal mass μ = h [ρm + (ρc − ρm)/(g + 1)]
> subject to max|w| ≤ w_lim, f₁ ≥ f_min, ceramic use h/(g + 1) ≤ C_max.

- Ceramic is stiffer per unit mass than aluminium, so without a ceramic limit the lightest plate is pure ceramic (g = 0). The ceramic-use limit, the TID's optional constraint, therefore defines the **mass–ceramic trade-off (Pareto front)**. It is reported over a range of C_max values, not one invented limit.
- **Optimizer (identical for every arm and for the TSDT reference):**
  1. An 81 × 81 grid in (log(1+g), log(a/h)).
  2. SLSQP refinement (finite-difference gradients, ftol 1e-12) from the 4 best feasible grid points, separated by ≥ 0.1 in scaled log(1+g), each restarted while it improves.
  3. The best feasible result is kept.
- **Reference optimum μ\*:** the same optimizer applied directly to TSDT.

### Part A: main problem (descriptive)
- a = 0.5 m, b = 0.75 m, q₀ = 1 MPa, w_lim = 0.14 mm, f_min = 1200 Hz.
- Chosen from an FSDT-only scan so that deflection governs for g ≲ 1, frequency for larger g, and plates are moderately thick (a/h ≈ 7–11), where FSDT and TSDT differ. w/h ≈ 0.002 (linear regime).
- C_max ∈ {∞, 40, 35, 30, 25, 20, 15, 10} mm, giving the Pareto front of mass against ceramic use.
- The FSDT scan shows a flat mass landscape at high g with two near-equal basins. The optimal (h, g) is therefore not unique in practice: designs are compared by **verified mass and feasibility**, not by distance in (h, g).

### Part B: 100 random problems (confirmatory)
- Generated in `run_h3.py` only after this file is fixed, with `numpy.random.default_rng(20270101)`, in order. For each problem:
  - b/a ~ U[0.5, 2];
  - an anchor design with log(1+g) ~ U[0.1, 0.9]·log 9 and log(a/h) ~ U[log 6, log 60];
  - perturbations δw, δf, δc ~ U[−0.1, 0.1];
  - a = 1 m, q₀ = 10 kPa;
  - w_lim = w_FSDT(anchor)(1+δw), f_min = f_FSDT(anchor)(1+δf), C_max = ceramic(anchor)(1+δc).
- **The limits use FSDT, never TSDT.** Problems whose TSDT reference is infeasible within the domain are excluded and counted.

## Metrics (per problem, arm, seed)
- **Open loop:** the surrogate optimum θ is reanalysed once with TSDT.
  - Violation v = max(0, w_T/w_lim − 1, 1 − f_T/f_min). The ceramic constraint is exact.
  - Mass gap Δ = μ(θ)/μ\* − 1.
  - Decision error E = v + |Δ|.
  - If an arm's own problem is infeasible, E = 1.
- **Closed loop (verify and correct):** after each TSDT reanalysis that shows a violation > 1e-4, the arm's deflection and frequency are rescaled by the TSDT/surrogate ratios at that design, and the problem is re-optimized. Maximum 5 TSDT analyses; the count is recorded. The loop stops at the first TSDT-feasible design: it guarantees feasibility but does not tighten a conservative (slightly heavy) design, so its final mass gap is not expected to be smaller than the open-loop gap.
- **Direct optimization cost:** SLSQP on TSDT from the box centre, same options (one start, restarted while improving). Every distinct design evaluated counts as one TSDT analysis. Its success (feasible within 1e-4 and μ within 0.1% of μ\*) is reported.

## Pre-registered questions (Part B, primary arm C)

| ID | Question | Confirmed if |
|---|---|---|
| Q1 | C's open-loop designs are accurate | pooled over 100 problems × 3 seeds: p95 of v ≤ 0.5% **and** p95 of \|Δ\| ≤ 0.5% |
| Q2 | C beats physics only | seed-averaged E per problem: 95% bootstrap CI of the median of (E_P − E_C) above 0 |
| Q3 | C beats data only | same with E_D |
| Q4 | Closed loop with C is reliable and cheap | every instance TSDT-feasible within ≤ 5 analyses; mean analyses per problem ≤ 2.0 for each seed; p95 of final \|Δ\| ≤ 0.5% |
| Q5 | Total cost is competitive | for the worst seed, 10 + Σ closed-loop analyses of C ≤ 10% of Σ direct-SLSQP analyses over the same problems |

**Secondary (descriptive):**
- Closed loop for P and D.
- Success and gaps of direct SLSQP; evaluation counts of the grid reference.
- Which constraints are active.
- Distance between designs in (g, a/h).
- w/h at the optima.
- Part A fronts for every arm (open and closed loop).
- Wall-clock times (not part of any decision, since every analysis here takes milliseconds).

## Caveats stated in advance
- The high-fidelity model is TSDT, a plate theory. Its Navier analyses take milliseconds, so the cost comparison counts **TSDT analyses as a stand-in for an expensive solver** (FE/3D). It does not claim a wall-clock speed-up here.
- Physics only (FSDT) is already accurate to about 1% for these designs. Its closed loop is expected to work too, and that will be reported.

## Reporting commitments
All Q1–Q5 outcomes are reported, confirmed or not. The problem set is generated once, and there are no re-runs with changed settings.
