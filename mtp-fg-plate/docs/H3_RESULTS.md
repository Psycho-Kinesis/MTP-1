# H3: surrogate-guided inverse design, verified with TSDT (executed)

9 November 2026 · Pre-registration: [`H3_PREREGISTRATION.md`](H3_PREREGISTRATION.md), written after development on pilot data and FSDT only (sha256 `83f35c3e…`, asserted). Development record: `results/h3_development/`. Results: `results/h3_inverse_design/`. Presentation figures: `figures/h3_main_problem_front.png`, `figures/h3_cost.png` (`experiments/plot_h3.py`).
Commands: `python experiments/run_h3_dev.py --output <dir>` (about 2 min), then `python experiments/run_h3.py --output <dir>` (about 60 min).

## Question (PID hypothesis H3)
Does optimization with the confirmed surrogate reach designs that are feasible when re-checked with the high-fidelity solver (TSDT), at competitive total cost?

## New mechanics needed: TSDT free vibration
The design problem has a frequency limit, so the TSDT solver was extended with the consistent mass matrix of the same third-order displacement field (`fgplate/tsdt.py`: `mass`, `eigenpairs`, `fundamental_batch`, and vectorized `stiffness_batch` / `transfer_batch`). Verification:
- **FSDT limit:** reproduces the existing FSDT mass matrix exactly.
- **Independent derivation:** agrees with the symbolic Ritz TSDT derivation to 6e-14.
- **Published values:** reproduces published Reddy TSDT fundamental frequencies of Al/Al₂O₃ square plates (a/h = 5, g = 0, 0.5, 1, 4, 10: 0.2113, 0.1807, 0.1631, 0.1378, 0.1301) to all 4 printed digits. These are quoted in Belkhodja et al. 2020, Table 5, from Hosseini-Hashemi et al.; the original table was not inspected.
- **Existing results unchanged:** the batch versions equal the scalar code to about 1e-13, and the existing functions are untouched (H2's truth reproduced bit for bit).

## Design problem (TID)
Simply supported Al/Al₂O₃ plate a × b under uniform pressure q₀. Choose thickness h and gradient index g (5 ≤ a/h ≤ 100, 0 ≤ g ≤ 8) to

> **minimize areal mass** h[ρm + (ρc − ρm)/(g+1)]
> subject to **max deflection ≤ w_lim**, **first frequency ≥ f_min**, **ceramic use h/(g+1) ≤ C_max**.

- Ceramic is stiffer per kilogram, so without a ceramic limit the lightest plate is pure ceramic. The limit C_max traces the **mass–ceramic trade-off**.
- Every arm, and the TSDT reference, uses the same optimizer: an 81×81 grid, then SLSQP from 4 separated starts.

| Arm | Static model | Frequency model | High-fidelity analyses |
|---|---|---|---|
| **C** | H2-extended E5 pipeline | exact FSDT × (k h)² correction from the same 10 designs | **10** (each gives static and modal labels) |
| P | physics only, label-free | exact FSDT | 0 |
| D | data only | data only | 10 |

On the 30 pilot validation designs (development), the frequency model of C has median error 0.02% and p95 0.1%; physics-only FSDT has a p95 of 0.63%.

## Part A: main problem (0.5 m × 0.75 m, 1 MPa, w ≤ 0.14 mm, f₁ ≥ 1200 Hz)

| C_max (mm) | TSDT optimum: g, a/h | Mass (kg/m²) | Governing limits | C: violation / mass gap (3 seeds) | P: violation / mass gap |
|---|---|---|---|---|---|
| none | 0 (pure ceramic), 10.7 | 177.7 | deflection | 0.36–0.44% / −0.12 to −0.15% | 0–0.44% / −0.15 to +0.01% |
| 40 | 0.28, 9.8 | 182.3 | deflection, ceramic | ≤ 0.30% / ≤ 0.12% | ≤ 0.005% / ≤ 0.01% |
| 30 | 0.96, 8.5 | 192.1 | deflection, ceramic | ≤ 0.06% / ≤ 0.02% | 0 / ≤ 0.02% |
| 25 | 1.53, 7.9 | 198.4 | frequency, ceramic | ≤ 0.01% / ≤ 0.14% | 0.07% / −0.09% |
| 20 | 2.34, 7.5 | 202.3 | frequency, ceramic | ≤ 0.03% / ≤ 0.09% | **0.77%** / −0.39% |
| 15 | 4.80, 7.1 | 203.2 | frequency | ≤ 0.14% / ≤ 0.16% | **0.77%** / −0.83% |
| 10 | 6.12, 7.0 | 203.5 | frequency, ceramic | ≤ 0.15% / ≤ 0.17% | **0.80%** / −0.92% |

- **The trade-off:** grading the plate cuts ceramic use from 46.8 mm to 10 mm (−79%) for 14.5% more mass. Most of that mass is paid before C_max ≈ 20 mm; below that the front is almost flat.
- **Where physics only goes wrong:** where frequency governs (thick, metal-rich plates), FSDT over-predicts the frequency. Its designs then violate the frequency limit by 0.8%, while looking 0.4–0.9% lighter than possible. The correction learned from 10 TSDT analyses removes this (C ≤ 0.15%).
- **Data only:** designs miss the optimum mass by 0.5–8% or violate a limit by 9%.
- **Direct TSDT optimization** needed 18–64 TSDT analyses per budget.

## Part B: 100 random problems (pre-registered); all five questions confirmed
- Problems were generated after the pre-registration, with FSDT-only limits. All 100 are feasible under TSDT.
- At the optima: deflection governs in 57 problems, frequency in 43, and the ceramic limit is active in 51. w/h ≤ 0.05 (linear regime), and the maximum deflection is at the centre (checked on a grid).

| ID | Question | Result | Verdict |
|---|---|---|---|
| Q1 | C's designs, checked once with TSDT, are accurate (p95 ≤ 0.5%) | violation p95 **0.23%** (median 0, max 0.76%); mass gap p95 **0.14%** (median 0.02%) | **Confirmed** |
| Q2 | C beats physics only | violation + \|gap\|: P − C = +0.074 points [+0.041, +0.159]; C better on 74/100 | **Confirmed** |
| Q3 | C beats data only | D − C = +10.7 points [+9.6, +12.6]; C better on 100/100 | **Confirmed** |
| Q4 | Verify-and-correct with C is reliable and cheap | all 300 runs TSDT-feasible; 1 analysis in 187 runs, 2 in 113 (mean 1.30 / 1.40 / 1.43 per seed); final mass gap p95 0.11% | **Confirmed** |
| Q5 | Total cost is competitive | C: 140–153 TSDT analyses in total (incl. its 10 labels) vs **7,938** for direct TSDT optimization (**1.9%**) | **Confirmed** |

**All arms (100 problems × 3 seeds):**

| | C (10 labels) | P (physics only) | D (data only) |
|---|---|---|---|
| Open loop: violation, median / p95 / max | 0 / 0.23% / 0.76% | 0.06% / 1.36% / 2.0% | 2.6% / 33% / 57% |
| Open loop: \|mass gap\|, median / p95 | 0.02% / 0.14% | 0.05% / 0.60% | 3.4% / 19% |
| Runs violating a limit by > 0.01% | 38% | 69% | 63% (+7 runs with no feasible surrogate design) |
| Verify-and-correct: feasible | 100% | 100% | 80% within 5 analyses |
| Analyses per problem, mean (max) | 1.38 (2) | 1.76 (3) | 2.7 (5) |
| **Total TSDT analyses, 100 problems** | **140–153** | 174–180 | 256–314 |

- **Direct optimization on TSDT** (SLSQP from the box centre) used 79 analyses per problem on average. It reached the reference optimum (within 0.1%) in 89% of problems.
- **The grid reference** used about 6,800 analyses per problem.

## What H3 establishes
1. **Surrogate-guided design works and is checkable.** With 10 high-fidelity analyses, the pipeline's optimal designs are within 0.14% of the best possible mass and within 0.23% of the limits (p95). One extra TSDT analysis, needed in 38% of runs, makes every design strictly feasible.
2. **It is far cheaper than optimizing the high-fidelity model directly:** about 1.4 TSDT analyses per problem plus 10 once, against about 79 per problem for direct SLSQP, about 2% of the cost.
3. **The learned correction matters where the physics model is biased.** FSDT is unconservative for thick, metal-rich plates: it under-predicts their deflection and over-predicts their frequency. Physics-only designs therefore violate limits by up to 2% (p95 1.4%), with the largest cases deflection limits at a/h ≈ 8 and g ≈ 4. Verify-and-correct repairs this too, but P needs more TSDT analyses (1.76 against 1.38 per problem). The 10 training analyses of C pay for themselves after about 26 problems.
4. **Data only is not usable for design with 10 labels:** median violation 2.6% and mass gap 3.4%, tails up to 33%. 20% of its runs reach no feasible design within 5 analyses.
5. **The answer is a design family, not a single point.** The mass landscape is flat at high g, so nearly equal masses occur at quite different (g, h). C's designs are close to the reference anyway (median difference 0.0015 in log(1+g)), but conclusions should rest on verified mass and feasibility, as the TID advises.

## Caveats
- **The cost is counted, not timed.** Every TSDT analysis here takes milliseconds, so the direct optimization is affordable in wall-clock time. The cost argument applies when the high-fidelity model is expensive (FE/3D), and it counts analyses as a stand-in for that case.
- **Without the check, C is not exactly safe.** 38% of its single-shot designs exceed a limit by more than 0.01% (at most 0.76%). Any design must be verified once with the high-fidelity solver, as the closed loop does.
- **Verify-and-correct stops at the first feasible design.** It guarantees feasibility, not optimality, so slightly conservative designs are not tightened; final gap p95 is 0.11%.
- **Physics only is already a strong competitor.** FSDT is within about 1–2% of TSDT here, and with verification it reaches feasible designs too; C's advantage is accuracy and fewer analyses, not feasibility as such.
- **Scope:** uniform pressure only; simply supported edges; plate theory (TSDT) as the reference, not 3D or experiment.
- **BLAS threads affect reproduction checks.** The reproduction assertions are sensitive to the BLAS thread count. A first launch with 2 threads per process stopped at the E5 reproduction check before evaluating any design, and the run was repeated with the default thread count (see README).

## Status
| Completed and verified | Not done |
|---|---|
| TSDT free vibration (FSDT limit, symbolic Ritz, published values); H3 development (pilot + FSDT only); pre-registered H3 with Part A front and 100 random problems; Q1–Q5 confirmed; determinism rerun identical (decisions, every record, per-design errors) | FE/3D or experimental check; clamped edges; non-uniform service loads; multi-objective (Pareto) optimization with an evolutionary method |

## Recommended next steps
1. **MTP report and presentation.** The chain from E2 to E5 plus H2 and H3 now answers all three PID hypotheses with pre-registered tests.
2. **Optional:** one higher-fidelity check (FE or 3D elasticity) of the final designs, or a clamped-edge plate, to show the method beyond Navier solutions.
