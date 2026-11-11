# MTP Project — Complete Context & Handoff Document

> **Implementation update — 9 November 2026:** Solver upgrades, the 200-design/600-field-case pilot and the first scalar baseline are complete. Read [PILOT_RESULTS.md](PILOT_RESULTS.md) and [DATASET_SCHEMA.md](DATASET_SCHEMA.md) for the current executed results. The sections below preserve the original Claude handoff and its historical claims/plans; they are not independent evidence of literature novelty or completed later work. The dated PID/TID in the parent output folder remain planning snapshots.

> The current split is 140/30/30 physical designs. The selected polynomial model's test mean errors are 0.676% (sine deflection) and 0.338% (fundamental frequency). No neural field model, FEM comparison or inverse-design optimization has yet been implemented. Reference attribution is corrected in validation/benchmark_references.json; reproduced Zenkour values are different-theory comparisons.

> **Purpose of this file.** This is the single source of truth for the project. It records
> everything decided and done so far, so that any collaborator (or an AI assistant) can pick
> up the work at any point without the original conversations. Keep it updated: add a dated
> entry to the **Decision & progress log** (Section 12) every time something changes.
>
> **Last updated:** 2026-11-09  ·  **Status:** solver hardening, 200-design pilot, E2 (reduced field baseline), E3a (physics vs data at matched capacity) E3b (imperfect physics + higher-fidelity labels) E3c (physics-structured correction) E4 (pre-registered confirmation: C1–C4 confirmed, tail marginal) E5 (improved base model, confirmed on a second fresh set: C1–C6 confirmed) H2 (unseen load shapes: Q1–Q5 confirmed) and H3 (TSDT-verified inverse design: Q1–Q5 confirmed) complete; next: thesis write-up. Sections 1–9 below are the original 5 Nov handoff; where they conflict with the PID/TID or the executed results, the latter win (superseded sections are marked).

---

## Table of contents
1. Project at a glance
2. Source material (the 17 PDFs)
3. Paper-by-paper summaries
4. Synthesis: themes, the ML "ladder", and the research gap
5. Chosen project direction (and backups)
6. Work plan (MTP-1 / MTP-2)
7. Work completed so far — Step 1: Navier FSDT solver
8. Code, environment and how to run
9. Next steps (Step 2 specification) and open questions
10. Explaining the project to a non-expert
11. Glossary
12. Decision & progress log

---

## 1. Project at a glance

| Item | Detail |
|---|---|
| Programme | Final-year **MTP** (Master's Thesis Project), **IIT Kharagpur** |
| Owner | Aditya (singhkumaraditya870@gmail.com) |
| Input from guide | 17 PDFs = the complete contents of one journal issue: *Advances in Engineering Software*, Vol. 220 (2026) — 16 research papers + 1 "Editorial Board" page |
| Requirement | Project must involve **machine learning (ML)** |
| Chosen direction (proposed, to be confirmed with guide) | **Physics-informed deep-learning surrogates (PINN + DeepONet) for fast analysis and inverse design of functionally graded (FG) plates** |
| Current stage | Step 1 done: exact Navier FSDT solver for FG plates, fully validated (23/23 checks) |
| Next stage | Step 2: dataset generation |

**One-sentence summary:** build an AI model that has learned the physics of functionally graded
plates, so the bending and vibration of *any* design can be predicted in about a millisecond (instead of minutes
with FEM), and use it to automatically find the best plate design.

---

## 2. Source material (the 17 PDFs)

All are from *Advances in Engineering Software*, Vol. 220 (2026), Elsevier. The PDFs are
copyrighted ("All rights reserved, including text and data mining, AI training") and are **not**
stored in this repository. Get them from the project owner or via the DOIs on ScienceDirect.
The numbering below (#1–#16) is used everywhere in this document.

| # | Title (short) | Authors (first) | DOI (10.1016/j.advengsoft.2026.…) |
|---|---|---|---|
| 1 | A coupled 2D homogenization and Navier series method for predicting low-velocity impact response of arbitrary truss-type lattice sandwich panels | Jin et al. (Central South Univ.) | 104241 |
| 2 | Algorithmic integration of superplastic forming constraints in the topology optimization of stiffened thin-walled structures | Cui et al. (Shandong Univ.) | 104226 |
| 3 | An efficient multi-physics computational framework for coupled friction-wear evolution and elastodynamic characterization of helical gears | Mo et al. (Guangxi Univ.) | 104223 |
| 4 | A response surface approach for uncertainty propagation in unreinforced masonry walls with interval mechanical properties | Sofi et al. (Reggio Calabria) | 104218 |
| 5 | Digital twin modeling for shrinkage porosity prediction of casting using reduced order model and machine learning | Zhang P. et al. (SJTU) | 104235 |
| 6 | Efficient hybrid FEM+ALE implementation for large deformation impact simulation of shallow foundations in granular media | Yosef et al. (Univ. Nebraska–Lincoln) | 104225 |
| 7 | From brittle fracture to elastoplastic fatigue: An open-source phase field framework in Julia | Zhang, Zhao, Tang (Tongji) | 104222 |
| 8 | Multi-physical topology optimization for thermo-mechanical coupled designs of hydraulic humanoid robot with experimental analysis | Li J. et al. (HUST) | 104237 |
| 9 | Pixel-level finite difference–enhanced physics-informed U-Net for heat conduction and effective properties prediction in heterogeneous composites | Shen & Zhao (UCAS) | 104221 |
| 10 | Pseudo-supervised physics-informed neural networks with transfer learning for nonlinear bending analysis of metamaterial beams | Yang et al. (RMIT) | 104239 |
| 11 | Scalable parametric acoustic field prediction via hierarchical physics-informed deep operator network with modal compression | Kim et al. (Jeonbuk Nat. Univ. / LG) | 104234 |
| 12 | Spatio-temporal dispersion analysis of dissipative time integration schemes for coupled peridynamic wave propagation | Yu & Noh (Korea Univ.) | 104236 |
| 13 | Tool Path Master Suite (TPMSuite): a direct slicing software for laser additive manufacturing of TPMS lattices | Huang et al. (SJTU) | 104219 |
| 14 | Transient analysis of viscoelastic graphene origami-enabled auxetic metamaterial plates | Karamanli (Duy Tan / Istinye) | 104220 |
| 15 | Variational energy-embedded multiphysics informed neural network framework for nonlinear bending, free vibration, and buckling of magneto-electro-elastic plates | Ly et al. (Van Lang Univ.; NIT Silchar) | 104224 |
| 16 | Vibroacoustic characteristics and control of irregular plate structures in sea-launch vessels | Zhang A. et al. (Ludong Univ.) | 104233 |
| – | Editorial Board | — | not a research paper (list of journal editors); ignored |

---

## 3. Paper-by-paper summaries

Format: **Problem → Method → Key results → Limitations / future work → Relevance to our project**.

### #1 Lattice sandwich panel under low-velocity impact (Jin et al.)
- **Problem:** dynamic response of sandwich panels with a truss lattice core (BCC) hit by a spherical impactor.
- **Method:** 2D asymptotic homogenization turns the lattice core into an equivalent plate (A, B, D + shear stiffness); Mindlin plate theory + Hamilton's principle; impactor as a mass–spring contact; **Navier double sine series + Duhamel integral** give a closed-form solution. The framework works for any 2D-periodic truss lattice (Table 5 gives several topologies).
- **Results:** validated against FEM and published data. A thicker core relative to the face sheets and a larger aspect ratio reduce impact resistance; strut diameter dominates (r_c 0.5→1.0 mm: peak force 12.32→14.06 kN, peak deflection 0.428→0.116 mm). BCC panel deflects about 10.76% less than an equal-mass solid panel. The mapping geometry → response is explicit, so it can be inverted for design.
- **Limits:** small deformation, elastic, simply supported, no thermal loads. Future: plasticity, high-velocity impact, thermo-mechanical.
- **Relevance:** ⭐ source of the **Navier method** used in our Step 1 solver; backup project option (ML surrogate for lattice panels).

### #2 SPF-constrained topology optimization of stiffened thin-walled structures (Cui et al.)
- **Problem:** design stiffened aerospace panels manufacturable by superplastic forming (SPF) + diffusion bonding.
- **Method:** stiffener **height** as the design variable; modified Heaviside projection maps it to element density; SPF constraints (average/local height, minimum pre-weld area, max/min size, cavity connectivity); MATLAB, based on the 88-line code, MMA optimizer.
- **Results:** uniform, continuous stiffeners without stacking; demonstrated on a plate and an aircraft cabin door.
- **Limits:** simplified SPF physics (no thinning or forming-limit model); results need CAD post-processing.
- **Relevance:** background on optimization; possible "ML-accelerated topology optimization" alternative (not chosen).

### #3 Friction–wear evolution of helical gears (Mo et al.)
- **Problem:** how tooth-surface wear changes the time-varying mesh stiffness (TVMS) over about 10⁹ cycles.
- **Method:** semi-analytical stacked-beam axial bending stiffness; Dowson–Higginson EHL lubrication, friction model, modified Archard wear; **triple-loop iteration** (contact stiffness / friction–lubrication / wear time-stepping).
- **Results:** >60× faster than 3D FEM, 4–6% TVMS error. Wear is self-limiting (running-in); peak wear migrates to the trailing end when the axial contact ratio ≈ 1.
- **Relevance:** background (fast solvers for digital twins). No ML.

### #4 Interval uncertainty in masonry walls via response surface (Sofi et al.)
- **Problem:** masonry properties are uncertain and data is too scarce for probability distributions.
- **Method:** **interval** (lower/upper bound) model with Improved Interval Analysis (extra unitary interval); ABAQUS simplified micro-model (cohesive joints, Drucker–Prager, XFEM) driven by Python scripts; **ratio-of-polynomials response surface** built from a few FE runs.
- **Results:** accurate lower/upper bounds of in-plane and out-of-plane resistance and failure modes with few deterministic analyses.
- **Relevance:** classical surrogate + uncertainty quantification; template for our UQ step (Step 6 option).

### #5 Digital twin for casting shrinkage porosity (Zhang P. et al.) — ML ✅
- **Problem:** predict shrinkage porosity in investment casting in real time.
- **Method:** 64 ProCAST simulations (Latin Hypercube over pouring temperature, shell preheat, emissivity, transfer time) → **two-level POD + RBF** reduced-order model of the temperature field → 22 physics-based features (solidification times, gradients, Laplacians, cooling rate, Niyama) → ML (RF, NN, SVM, KNN, GPR). Greedy algorithm selects 8 thermocouple locations; measurements update the POD coefficients online.
- **Results:** classification accuracy >97% (AUC 0.995); RF regression R² = 0.98; temperature error <20 °C; hundreds of times faster than ProCAST. Good practice: **train/test split by simulation case** (avoids leakage), 5-fold CV, feature importance.
- **Limits:** geometry/alloy-specific; needs retraining for new castings.
- **Relevance:** ⭐ template for our dataset + baseline ML (Steps 2–3) and for POD.

### #6 Hybrid FEM+ALE for pole foundations in sand (Yosef et al.)
- **Problem:** vehicle impact on luminaire-pole foundations embedded in sand (very large soil deformation).
- **Method:** LS-DYNA, soil as ALE (multi-material), structure as Lagrangian FEM; validated by full-scale bogie crash tests.
- **Results:** a **critical embedment depth** exists beyond which extra depth gives little benefit; rotation-point depth scales with embedment depth.
- **Relevance:** background only (expensive LS-DYNA; no ML).

### #7 Open-source phase-field fracture/fatigue in Julia (Zhang, Zhao, Tang)
- **Problem:** one code for brittle fracture, ductile fracture, elastic fatigue and elastoplastic fatigue.
- **Method:** Julia/Gridap; AT1/AT2 phase field; Amor/spectral splits; J2 plasticity; fatigue degradation; **cycle-jump** (~100× speed-up). Code: github.com/FCMzp05/PhaseFieldFracture.
- **Results:** 8 benchmarks; GH4169 superalloy at 550 °C within 10% of experiments; runs take 17–300 min.
- **Future work they list:** use their data to train neural-network / physics-informed surrogates.
- **Relevance:** possible alternative data source for an ML fatigue surrogate (not chosen).

### #8 Thermo-mechanical topology optimization of a hydraulic humanoid robot leg (Li J. et al.)
- **Method:** node-based topology optimization on tetrahedral meshes, steady heat conduction with weak thermo-mechanical coupling, adjoint sensitivities, Helmholtz filter, Heaviside projection, MMA, STL output.
- **Results:** shank 51.7% lighter, 57.1% lower moment of inertia, 58.4% more heat dissipation (tested experimentally).
- **Relevance:** background on optimization (Step 6 inspiration).

### #9 Physics-informed FD-U-Net for composites (Shen & Zhao) — ML ✅
- **Problem:** steady temperature field in a two-phase composite from a 100×100 microstructure image.
- **Method:** U-Net (encoder–decoder CNN); **label-free physics loss**: the network output is pushed through a finite-difference stencil written as a convolution kernel, and the loss is the mismatch with the next FD iterate (best with about 200 kernel iterations; P-OHEM loss; batch size 1). 12,000 synthetic images (QSGS/RGGM); RTX 4090.
- **Results:** MAE ≈ 0.145 on a 0–10 °C range (<1.5%); generalizes to unseen hybrid microstructures (MAE 0.12); a data-driven U-Net overfits. Also handles multi-phase materials, heat sources, and **effective thermal conductivity** (homogenization).
- **Relevance:** physics-informed CNN idea; alternative project option ("physics-informed U-Net for effective properties").

### #10 Pseudo-supervised PINN for nonlinear bending of GOEAM beams (Yang et al.) — ML ✅
- **Problem:** nonlinear bending of FG graphene-origami auxetic metamaterial (GOEAM) beams on a Winkler–Pasternak foundation (FSDT + von Kármán).
- **Method:** **energy-based PINN** (minimize total potential energy) + **adaptive Gauss quadrature** + **two networks** (linear and nonlinear) exchanging stop-gradient **pseudo-labels** + **transfer learning** of encoder weights.
- **Results:** max-deflection error 4.14% (plain energy PINN) → <0.1%; quadrature about 10× better than Monte Carlo; X-type GOri distribution is the most efficient.
- **Limits:** 1D static beam; **one configuration per training run**. Future: plates/shells, transient, vibration.
- **Relevance:** ⭐ core method for our PINN step; motivates the research gap.

### #11 Hierarchical physics-informed DeepONet for acoustics (Kim et al., with LG) — ML ✅
- **Problem:** parametric Helmholtz acoustic field of a washing machine (interior + exterior), any frequency/source.
- **Method:** **POD-compressed DeepONet** (POD modes from cheap low-fidelity snapshots = fixed trunk; branch net maps parameters → modal coefficients); **data-driven pretraining then physics-informed fine-tuning** (PDE residual + boundary/interface losses); **binary mask** for interior/exterior; real and imaginary parts coupled.
- **Results:** 40–70% lower MSE than POD-DeepONet; R² > 0.99 on unseen sources; **~1 ms per query, 10³–10⁴× faster than FEM**.
- **Limits:** generalization tested only within ±5% source shifts; linear POD basis may fail out of distribution.
- **Relevance:** ⭐ core architecture for our main contribution (Step 5).

### #12 Dispersion analysis for coupled peridynamics–FEM (Yu & Noh)
- **Problem:** spurious wave reflections at PD/FE coupling interfaces corrupt crack paths.
- **Method/result:** combined spatial + temporal dispersion analysis; the dissipative **Noh–Bathe** explicit scheme damps bad high-frequency modes better than central differences; new reflection indicator.
- **Relevance:** background only.

### #13 TPMSuite — direct slicing of TPMS lattices for LPBF (Huang et al.)
- **Method:** laser paths generated directly from the implicit TPMS equation (no STL); overhang detection and down-skin path planning; parallel computing. Code: github.com/TPMSuite/Tool_Path_Master_Suite.
- **Results:** −87% preprocessing time, −99% peak memory, 5.62× parallel speed-up; parts with 3,700 cells, relative density down to 4.3%.
- **Relevance:** background (lattices/manufacturing).

### #14 Transient viscoelastic GOEAM plates (Karamanli)
- **Method:** three higher-order FE plate models; Kelvin–Voigt viscoelasticity; step and blast loads; parameters: GOri distribution, hydrogen coverage, weight fraction, layers, BCs, aspect ratio, damping.
- **Results:** a good GOri/hydrogen configuration stiffens the plate and lowers deflection; hydrogenation softens it; clamped edges and higher viscosity reduce and damp oscillations. Its literature review cites many **ML studies on GOEAM** (DNN, RF, genetic programming).
- **Relevance:** ⭐ material model and benchmarks for the GOEAM extension of our project; transient/blast extension.

### #15 Energy-based multiphysics PINN for magneto-electro-elastic plates (Ly et al.) — ML ✅
- **Problem:** nonlinear bending, free vibration and buckling of FG magneto-electro-elastic plates (BaTiO₃/CoFe₂O₄), Murakami zigzag theory + von Kármán.
- **Method:** energy-based PINN; network outputs 11 fields from (x, y); **adaptive (learnable) Fourier features** against spectral bias; hard BCs via distance functions; two-stage sampling (Chebyshev then Sobol); Adam then L-BFGS.
- **Results:** matches FEM; finds buckling modes without being told them; **~12.7 min per case** (RTX 5060). Piezo-rich plates are softest in linear bending but stiffest at large deflection.
- **Limits:** one configuration per training run. Future: inverse design, optimization, control.
- **Relevance:** ⭐ core method for our PINN step; with #10 defines the research gap.

### #16 Vibroacoustics of irregular FGM plates for sea-launch vessels (Zhang A. et al.)
- **Method:** FSDT; irregular quadrilateral mapped to a square; **Chebyshev spectral collocation**; Rayleigh integral for radiated sound; dynamic vibration absorber (DVA); validated by FEM and experiment (isotropic).
- **Results:** a tuned DVA splits the resonance peak and reduces vibration and noise; thickness, gradient index and excitation location shift the resonances.
- **Relevance:** FG plate benchmarks; future extension (irregular geometry, absorber optimization with surrogates).

---

## 4. Synthesis

### 4.1 Common thread of the whole issue
High-fidelity simulation (FEM) is accurate but slow, so tasks that need many runs (design
optimization, uncertainty quantification, real-time digital twins) are expensive. Every
paper offers a faster or better route: analytical solutions, semi-analytical solvers, better
numerics, classical surrogates, or **machine learning**.

### 4.2 Clusters
| Cluster | Papers | Relevance to us |
|---|---|---|
| A. Scientific ML / surrogates | 4, 5, **9, 10, 11, 15** | ⭐⭐⭐ core |
| B. Plates, beams, FG/metamaterials | **1, 10, 14, 15, 16** (+ 9, 13 for lattices/composites) | ⭐⭐ application |
| C. Topology optimization | 2, 8 | ⭐ background for design |
| D. Specialized numerics | 3, 6, 7, 12, 13 | background |

### 4.3 The ML "ladder" (useful for the viva)
1. **Data-driven ML** (#5): needs many simulations, knows no physics.
2. **PINN** (#10, #15): no data, obeys physics, but **retrains for every new case** (~13 min each).
3. **Physics-informed CNN / operator learning** (#9, #11): learns the whole parameter → solution map with physics in the loss; ~1 ms per query; generalizes.

### 4.4 Research gap (the project's contribution)

> ⚠️ **Superseded (9 Nov 2026).** The novelty claim below was based on one journal issue only. The Research Review found closely related external work [A1]–[A4]; the working question is now label efficiency and load-family generalization under matched budgets (PID H1–H3). Kept for history; see planning/ (PID, TID) and docs/PILOT_RESULTS.md, docs/E2_RESULTS.md for the current position.

- #10 and #15 show energy-based PINNs solve FG/metamaterial beams and plates accurately, but each design needs a separate training run, which is useless for design loops. Both list inverse design/optimization and plates/transient as future work.
- #11 shows **physics-informed, POD-compressed DeepONet** learns parametric solution operators (≈1 ms/query, R² > 0.99) — but only for acoustics.
- #1, #14, #15, #16 show FG / metamaterial plates are an active area with a rich design space.
- **Gap:** no work in this set applies physics-informed operator learning to the parametric analysis and inverse design of FG (or GOEAM) plates.

---

## 5. Chosen project direction (and backups)

> ⚠️ **Superseded (9 Nov 2026).** Current proposed title (PID): *Data efficient physics informed surrogates for bending and vibration analysis and inverse design of functionally graded plates*. Speed-up over FEM is no longer a claimed benefit: the Navier reference solver is already sub-millisecond. Kept for history; see planning/ (PID, TID) and docs/PILOT_RESULTS.md, docs/E2_RESULTS.md for the current position.


### 5.1 Proposed title (to confirm with guide)
**"Physics-informed deep learning surrogates for fast analysis and inverse design of functionally graded plates"**
(start with ceramic–metal FGM; extend to graphene-origami metamaterial plates if time permits)

### 5.2 Pipeline
```
Navier FSDT analytical solver (exact, ~0.2 ms/design)        <- technique from #1; materials from #10/#14
   -> dataset via Latin Hypercube Sampling                    <- #5
   -> baseline surrogates: MLP / GPR / Random Forest          <- #5, #4
   -> single-case energy-based PINN                           <- #10, #15
   -> POD-DeepONet + physics-informed fine-tuning (MAIN)      <- #11 (+ POD from #5)
        inputs: n, a/h, b/a, (load), later GOri params; outputs: full deflection field, frequency, buckling
   -> inverse design / optimization + uncertainty quantification  <- #2, #8, #4
   -> validation against FEM (ANSYS / ABAQUS) on selected designs
```

### 5.3 Why this choice
Real scientific ML (PINN, DeepONet); feasible (own analytical solver = free, exact data; no licences or huge compute);
grounded in 8 of the 16 papers; low risk (Steps 1–5 already make a complete project); publishable.

### 5.4 Backup options (if the guide prefers)
- **B1 (simplest):** ML surrogate + inverse design for BCC lattice sandwich panels under impact (#1).
- **B2:** physics-informed U-Net for effective properties of microstructures (#9), extended to elasticity.
- **B3:** ML surrogate for fatigue life using the open-source phase-field code (#7) — small data (100–300 runs) → GPR / transfer learning.
- Not recommended: #6 / #4 (need many LS-DYNA / nonlinear ABAQUS runs), #3 (complex solver first), topology optimization + DL (large effort).

---

## 6. Work plan

> ⚠️ **Superseded (9 Nov 2026).** The plan is now the PID's gated 16-week envelope. Done: solver hardening, 200-design pilot (140/30/30), scalar baseline, E2. Next: E3a. Kept for history; see planning/ (PID, TID) and docs/PILOT_RESULTS.md, docs/E2_RESULTS.md for the current position.


| Step | Work | Semester | Status |
|---|---|---|---|
| 0 | Literature review of 16 papers (clusters, gap) | MTP-1 | ✅ analysis done (this document); report chapter to write |
| 1 | Navier FSDT solver for SSSS FG plates + validation | MTP-1 | ✅ **done** |
| 2 | Dataset generation (LHS) | MTP-1 | ⏭ next |
| 3 | Baseline ML (MLP, GPR, RF) for max deflection & frequency | MTP-1 | |
| 4 | Energy-based PINN for one plate; compare with Navier | MTP-1 | |
| 5 | POD-DeepONet + physics-informed fine-tuning; compare PINN / data-driven DeepONet / PI-DeepONet / FEM (accuracy, data, speed) | MTP-2 | |
| 6 | Inverse design / optimization (+ UQ) | MTP-2 | |
| 7 | FEM validation (ANSYS/ABAQUS), thesis, possible paper | MTP-2 | |
| ext | GOEAM material; von Kármán nonlinearity; transient/blast (#14) | MTP-2 (optional) | |

**MTP-1 deliverable:** literature review + validated solver + dataset + first ML surrogate + first PINN.

---

## 7. Work completed — Step 1: Navier FSDT solver

### 7.1 Theory implemented
- **Plate:** rectangular a × b × h, all four edges **simply supported (SSSS)**, linear elastic, small deflection.
- **Material:** two-phase power-law FGM, ceramic fraction `Vc(z) = (z/h + 1/2)^n` (n = 0 → fully ceramic; n → ∞ → metal);
  rule of mixtures `P(z) = (Pc − Pm) Vc + Pm` for E and ρ; constant ν. Default: **Al/Al₂O₃**,
  Em = 70 GPa, Ec = 380 GPa, ρm = 2702, ρc = 3800 kg/m³, ν = 0.3.
- **Thickness integrals** `∫P(z) z^k dz` are computed **exactly in closed form** (no quadrature error, including n < 1):
  with ζ = z/h + 1/2: `h^(k+1) Σ_j C(k,j)(−1/2)^(k−j) [(Pc−Pm)/(n+j+1) + Pm/(j+1)]`.
- **Stiffness:** with E0, E1, E2 = ∫E(1, z, z²)dz: A = E0·c, B = E1·c, D = E2·c, where c11 = 1/(1−ν²), c12 = ν/(1−ν²), c66 = 1/(2(1+ν));
  shear As = ks·c66·E0, **ks = 5/6**. Inertias I0, I1, I2 = ∫ρ(1, z, z²)dz.
- **Kinematics (FSDT/Mindlin):** u = u0 + zφx, v = v0 + zφy, w = w0.
- **Navier expansion:** u0 = U cos αx sin βy, v0 = V sin αx cos βy, w0 = W sin αx sin βy, φx = X cos αx sin βy, φy = Y sin αx cos βy; α = mπ/a, β = nπ/b.
- **Per-mode 5×5 system** (symmetric), unknowns [U V W X Y]:
  ```
  K11 = A11 α² + A66 β²       K12 = (A12+A66) αβ        K14 = B11 α² + B66 β²     K15 = (B12+B66) αβ
  K22 = A66 α² + A11 β²       K24 = (B12+B66) αβ        K25 = B66 α² + B11 β²
  K33 = As (α² + β²)          K34 = As α                K35 = As β
  K44 = D11 α² + D66 β² + As  K45 = (D12+D66) αβ        K55 = D66 α² + D11 β² + As
  M11 = M22 = M33 = I0,  M44 = M55 = I2,  M14 = M25 = I1
  ```
  Bending: `K·[U V W X Y] = [0 0 Q_mn 0 0]`; vibration: `(K − ω²M)·x = 0` (lowest ω = flexural mode).
- **Loads:** sinusoidal `q0 sin(πx/a) sin(πy/b)` (only m = n = 1); uniform `Q_mn = 16 q0/(π² m n)` for odd m, n (default series up to 49).
- **Normalizations used:** `w̄ = 10 Ec h³ w(a/2,b/2)/(q0 a⁴)`; `ω̂ = ω h √(ρc/Ec)` (also `ω a²/h √(ρc/Ec)` in plots).

### 7.2 Validation results (`python validation/validate.py --full` → 23/23 PASS)
| Check | Result | Reference | Error |
|---|---|---|---|
| Thin isotropic, uniform load, w·D/(q a⁴) | 0.00406237 | 0.00406235 (Timoshenko) | 6e-6 |
| Thin isotropic, sinusoidal, exact CPT | ✓ | closed form | 6e-6 |
| Thin isotropic, frequency, exact CPT | ✓ | closed form | 6e-6 |
| Thick isotropic a/h=10, ω h √(ρ/G) | 0.09303 | 0.0930 (Mindlin FSDT) / 0.0932 (exact 3D) | 3e-4 / 1.9e-3 |
| FG a/h=10, w̄, n=0 | 0.2961 | 0.2960 | 2e-4 |
| FG a/h=10, w̄, n=1 | 0.5890 | 0.5889 | 1e-4 |
| FG a/h=10, w̄, metal | 1.6072 | 1.6070 | 1e-4 |
| Neutral-surface closed form (n = 0.5, 2, 4, 10; b/a = 1, 1.5; a/h = 5, 20) | ✓ | exact | ≤1e-14 |
| Uniform-load series convergence (9 / 19 / 49 terms) | ✓ | 199 terms | 1e-4 / 1e-5 / 6e-7 |
| Independent symbolic Ritz FSDT (deflection & frequency, n = 0.5, 4) | ✓ | sympy energy solution | ≤3e-14 |

**Important note on literature values (must know for comparisons and the viva):** many FG tables
(e.g. Zenkour 2006) list w̄ = 0.7573 (n=2), 0.8819 (n=4), 1.0089 (n=10). Those come from **higher-order**
shear theories. Our independent Ritz implementation of **Reddy's TSDT** reproduces them (0.7573, 0.8815, 1.0087),
while **FSDT** gives 0.7552, 0.8736, 0.9966. This ~1% gap is a known theory difference, **not a bug**.
Always check which plate theory a reference used.

### 7.3 Reference outputs (square Al/Al₂O₃, sinusoidal load) — from `demo.py`
| a/h | n | w̄ | ω̂ = ω h √(ρc/Ec) |
|---|---|---|---|
| 5 | 0 / 1 / 5 / 10 | 0.3435 / 0.6691 / 1.0495 / 1.1802 | 0.21121 / 0.16309 / 0.13797 / 0.13238 |
| 10 | 0 / 1 / 5 / 10 | 0.2961 / 0.5890 / 0.9014 / 0.9966 | 0.05769 / 0.04419 / 0.03788 / 0.03658 |
| 20 | 0 / 1 / 5 / 10 | 0.2842 / 0.5690 / 0.8644 / 0.9508 | 0.01480 / 0.01131 / 0.00973 / 0.00942 |
| 100 | 0 / 1 / 5 / 10 | 0.2804 / 0.5625 / 0.8526 / 0.9361 | 0.00060 / 0.00046 / 0.00039 / 0.00038 |

FG frequencies were cross-checked against the independent Ritz FSDT (agreement ~1e-13); TSDT
frequencies differ from FSDT by <0.6% at n = 10, a/h = 10.

**Speed:** ≈ 0.2 ms per design (bending + frequency) → ≈ 4,900 designs/second (single CPU core).

**Physical trends (sanity):** larger n (more metal) → softer plate → larger deflection, lower frequency;
thick plates (a/h = 5) deviate most from thin-plate theory because shear deformation matters.

### 7.4 Current limitations
SSSS only; linear small deflection; static bending + free vibration only (no buckling, transient or blast yet);
power-law Voigt FGM, constant ν, ks = 5/6.

---

## 8. Code, environment and how to run

```
mtp-fg-plate/
├── README.md                     quick start + validation summary
├── requirements.txt              numpy, scipy, matplotlib, sympy
├── fgplate/
│   ├── __init__.py
│   ├── material.py               FGMaterial, al_al2o3(), isotropic(), exact moments, plate_stiffness()
│   └── navier.py                 NavierFSDTPlate (matrices, bending, frequencies), BendingSolution
├── validation/
│   ├── validate.py               19 fast checks (+4 with --full)
│   └── ritz_crosscheck.py        independent symbolic Ritz solver (FSDT and TSDT)
├── demo.py                       parametric study, timing, figures
├── figures/                      1_material_profile.png, 2_deflection_shape.png, 3_parametric_study.png
└── docs/PROJECT_CONTEXT.md       ← this file
```

```bash
pip install -r requirements.txt
python validation/validate.py --full    # expect: 23/23 checks passed
python demo.py                          # tables + figures in figures/
```

```python
from fgplate import NavierFSDTPlate, al_al2o3
plate = NavierFSDTPlate(a=1.0, b=1.0, h=0.1, material=al_al2o3(n=2))
w_c   = plate.bending(q0=1.0, load="uniform").center_deflection()   # m
X, Y, W = plate.bending(1.0, "uniform").deflection_grid(41, 41)      # full field
omega = plate.fundamental_frequency()                               # rad/s
```

Developed and tested with Python 3.13, numpy, scipy 1.18, matplotlib 3.11, sympy. PyTorch is **not** installed yet (needed from Step 4).

---

## 9. Next steps and open questions

> ⚠️ **Superseded (9 Nov 2026).** The dataset specification below was replaced by the TID protocol and executed as the pilot (generator pilot-1.0; new data uses pilot-1.1). The guide questions in 9.2 remain open. Kept for history; see planning/ (PID, TID) and docs/PILOT_RESULTS.md, docs/E2_RESULTS.md for the current position.


### 9.1 Step 2 — dataset generation (proposed specification)
- **Inputs (sampled with Latin Hypercube):** gradient index n ∈ [0, 10]; slenderness a/h ∈ [5, 100]; aspect ratio b/a ∈ [0.5, 2];
  load type (sinusoidal / uniform); optionally Ec/Em ratio or different material pairs.
- **Outputs:** w̄ (centre), ω̂ (fundamental), first few frequencies, and the **full deflection field on a fixed grid**
  (e.g. 32×32 or 41×41 in normalized coordinates) for POD/DeepONet.
- **Size:** ~5,000–20,000 samples (seconds to generate). Save as `.npz` / `.csv` with a fixed random seed.
- **Splits:** train/validation/test (e.g. 70/15/15) plus an **extrapolation test set** (e.g. n ∈ [8, 10] held out) to test generalization.
- Consider sampling n on a log or transformed scale (responses change fastest for small n).

### 9.2 Questions to settle with the guide
1. Confirm the main direction (FG plates + PINN/DeepONet) vs. a backup (B1 lattice panel is the simplest).
2. Ceramic–metal FGM only, or include the GOEAM (graphene origami) material?
3. GPU access (lab / department / Colab)?
4. FEM software for final validation (ANSYS or ABAQUS licence)?
5. Report/presentation format and MTP-1 evaluation dates.

### 9.3 Learning path for team members
Python + NumPy → plate theory (Reddy, *Theory and Analysis of Elastic Plates and Shells*; FSDT, Navier) → scikit-learn regression →
PyTorch → PINN (DeepXDE examples; papers #10, #15) → POD/SVD (`np.linalg.svd`) → DeepONet (DeepXDE; paper #11).

---

## 10. Explaining the project to a non-expert

**30-second version:** Engineers design parts like aircraft panels with computer simulations, but those
simulations are slow — testing thousands of design options can take weeks. This project builds an AI that
learns the physics of these panels, so it predicts how any design will behave in a fraction of a second, and
then uses it to automatically find the best design.

**Key analogies:**
- *FG plate:* like a glass of lassi where sugar settles — ceramic on top, metal at the bottom, smooth change in between.
- *FEM:* solving thousands of puzzle pieces together — accurate but slow.
- *Surrogate model:* an experienced engineer who has learned the pattern and answers instantly.
- *PINN:* a student who learns the rules (physics) instead of memorizing answers — but must re-study for every new question.
- *DeepONet:* a pattern system that instantly fits a shirt to any body size, instead of a tailor stitching one shirt at a time.
- *Inverse design:* asking "what design meets my requirement?" instead of "how does this design behave?".

---

## 11. Glossary
| Term | Meaning |
|---|---|
| FEM | Finite Element Method: splits a structure into small elements and solves the physics numerically |
| FGM / FG plate | Functionally graded material: composition varies smoothly through the thickness |
| Gradient index n | Exponent of the power law controlling the ceramic → metal transition |
| GOEAM | Graphene origami-enabled auxetic metamaterial (folded graphene in copper; negative Poisson's ratio possible) |
| CPT / FSDT / TSDT | Classical, first-order, third-order shear deformation plate theories |
| ks | Shear correction factor (5/6) used in FSDT |
| SSSS | All four edges simply supported |
| Navier solution | Exact double sine-series solution for SSSS rectangular plates |
| Ritz method | Approximate/exact solution by minimizing energy over assumed functions |
| Surrogate model | Fast approximate model trained to replace a slow simulation |
| LHS | Latin Hypercube Sampling: evenly spread design-of-experiments sampling |
| MLP / GPR / RF | Multilayer perceptron (neural net) / Gaussian process regression / random forest |
| PINN | Physics-informed neural network: loss built from the governing equations or energy |
| Energy-based (variational) PINN | PINN that minimizes total potential energy instead of PDE residuals |
| POD | Proper orthogonal decomposition: extracts a few dominant "basic shapes" from many solutions (via SVD) |
| DeepONet | Deep operator network: learns the map from parameters/functions to whole solution fields (branch + trunk nets) |
| UQ | Uncertainty quantification |
| Inverse design | Finding design parameters that achieve a target response |

---

## 12. Decision & progress log
| Date | Entry |
|---|---|
| 2026-10-21 | Received and analysed PDFs in four batches (5 + 5 + 5 + 2 = 17). Identified that all are from *Adv. Eng. Softw.* Vol. 220 (2026); the "Editorial Board" file is not a paper (16 research papers). |
| 2026-10-21 | Clustered the papers (Section 4); identified the research gap (PINN per-case retraining vs. operator learning for FG plates). |
| 2026-10-21 | **Decision (proposed):** main direction = physics-informed operator learning for FG plates; backups B1–B3. To be confirmed with the guide. |
| 2026-10-21 | **Step 1 done:** Navier FSDT solver (`fgplate/`), validation suite, independent Ritz FSDT/TSDT cross-check, demo + figures. |
| 2026-10-21 | Fixed: Gauss quadrature gave ~1e-6 error for n < 1 → replaced with exact closed-form thickness integrals. |
| 2026-10-21 | Found: published 0.8819 / 1.0089 (n = 4 / 10) are higher-order-theory values; FSDT gives 0.8736 / 0.9966 (confirmed via TSDT). |
| 2026-11-05 | Created this handoff document for collaborators. |
| 2026-11-07 | Collaborator (Astra) delivered Research Review, PID and TID: reframed the claim to data efficiency; found sine-load fields are rank 1 and the Navier solver already fast. |
| 2026-11-09 | Collaborator delivered solver hardening and the 200-design pilot (pilot-1.0): five fields, symmetric eigenproblem, 600 field cases, frozen 140/30/30 split, scalar baseline (test mean 0.68% / 0.34%). |
| 2026-11-09 | Independent review (docs/PILOT_REVIEW_2026-11-09.md): all 600 cases re-verified with the original solver (≤3e-13); no leakage; found platform-dependent IDs/mixture loads under pilot-1.0; scalar task has an exact closed form (pipeline check only). |
| 2026-11-09 | Fixes: generator pilot-1.1 (platform-stable IDs/loads), platform-independent audit with provenance snapshot, bootstrap CIs in metrics. pilot_200 left unchanged. |
| 2026-11-09 | E2 executed (docs/E2_RESULTS.md): POD needs rank 60 (five fields); modal-transfer model best (val median 0.44%, p95 3.6% at 140 designs); targets not met at ≤40 designs; gate: SSSS remains a usable testbed if physics and data arms share a non-exact model class. |
| 2026-11-09 | E3a executed (docs/E3A_RESULTS.md): same polynomial model class; data only needs 140 labels for the target, hybrid 10, physics only 0 (FSDT energy at ≥140 unlabelled designs). H1 supported at N = 10–70. Caveat: physics and labels are the same FSDT model; no cost saving on this family; hybrid tails grow with labels under median-first selection. |
| 2026-11-09 | TSDT Navier solver added and verified (FSDT limit 3e-16; symbolic Ritz TSDT 9e-16; published higher-order values). Per-field ridge scope added (E3a unchanged). |
| 2026-11-09 | E3b executed (docs/E3B_RESULTS.md): FSDT physics + TSDT labels, tail-aware selection. Q2 supported (physics-informed arms beat data only at N = 10–40, e.g. 18% → 0.78% median at N = 10). Q1 not supported (few random TSDT labels do not remove FSDT bias at N ≤ 40); all arms need ~140 labels for p95 ≤ 3%. Mechanism: global correction fixes thick plates but adds error to slender ones. |
| 2026-11-09 | E3c pre-registered after training-only diagnostics (discrepancy ∝ (k h)², correlation 0.99; FSDT shear share tracks the gap, Spearman 0.98). Executed (docs/E3C_RESULTS.md): primary M-S (H_P·(k h)² correction, shear-guided labels) beats physics only and data only at N = 10, 20, 40 (Q1, Q2 supported); validation target met with 10 labels; test median 0.33–0.48%, p95 3.1–4.0% (narrowly above 3%). Correction form decisive; placement adds little (hurts the global form). P, D, G-R reproduce E3b exactly. |
| 2026-11-09 | E4 pre-registered (docs/E4_PREREGISTRATION.md, sha256 a60526d7…) before any fresh data; executed once on 200 fresh designs (pilot-1.1, seed 20261201). 93/93 frozen arms reproduced E3c. C1 (M-S > physics only), C2 (M-S > data only), C3 (median 0.53% ≤ 1%) and C4 (p95 2.85% ≤ 3%, CI 2.05–3.88, marginal) confirmed. Remaining tail traced to the label-free base model in the narrow / near-uniform / slender corner (exact FSDT within 0.3–0.75% there). |
| 2026-11-09 | E5 development (pilot train/validation only): classical-plate scaling prefactor makes the label-free FSDT surrogate 25× more accurate (P2: 0.027% / 0.10% vs FSDT). Pre-declared rule v1 (base scored alone vs TSDT) chose the old base P0 because its errors partly cancel the FSDT bias; rule v2 (downstream M-S score, declared after v1's outcome, before any test data) chose P2. Both recorded. |
| 2026-11-09 | E5 pre-registered (docs/E5_PREREGISTRATION.md, sha256 f25d9ea8…) and confirmed once on 200 new fresh designs (seed 20261215): all six questions confirmed. M-S@P2 with 10 TSDT labels: median 0.062%, p95 0.33% (CI 0.20–0.49%); beats the old pipeline on 200/200 designs and data only at every N. Caveat: the FSDT→TSDT gap is unusually well structured. |
| 2026-11-09 | H2 pre-registered (docs/H2_PREREGISTRATION.md, sha256 5b8e0ef2…) after feasibility checks on training designs only; executed once on 200 fresh designs (seed 20261231) with four load families (training-type mixture, uniform, wide and narrow Gaussian patches) against a TSDT 25×25 truth, exact modal L2 metric. Q1–Q5 confirmed. M-S@P2 extended to 15×15 modes (frozen hyperparameters, same 10 labelled designs): median 0.061–0.070%, p95 0.33–0.40% on every family. The frozen 5×5 pipeline sits at its truncation floor on narrow patches (1.47% / 3.14%); extension removes it on 200/200 designs. Caveat: patches narrower than one thickness are outside plate-theory validity (agreement with TSDT only). |
| 2026-11-09 | TSDT free vibration added (consistent mass matrix; FSDT limit exact, symbolic Ritz 6e-14, published TSDT frequencies to 4 digits). H3 developed on pilot data and FSDT only (frequency correction on the same 10 labelled designs: validation p95 0.1%), pre-registered (docs/H3_PREREGISTRATION.md, sha256 83f35c3e…) and executed: main-problem Pareto front (ceramic use −79% for +14.5% mass) and 100 random problems verified with TSDT. Q1–Q5 confirmed: open-loop designs within 0.14% mass / 0.23% limits (p95); verify-and-correct feasible in all 300 runs with ≤ 2 TSDT analyses; total 140–153 TSDT analyses vs 7,938 for direct TSDT optimization. Physics only violates by up to 2% on thick metal-rich plates. A first launch with 2 BLAS threads stopped at the E5 reproduction check (no designs evaluated) and was repeated with default threads. |
| 2026-11-09 | Session handoff: SESSION_HANDOFF.md (workspace root) written; package MTP_Project_v9. |
| 2026-11-10 | H3 determinism rerun completed: identical decisions, records and per-design errors. FG Plate Lab built (demo/): browser port of the solvers and frozen surrogates, tested against Python; simulator, design studio, evidence and method views; offline file and demo video. |
| 2026-11-11 | Project pushed to GitHub (Psycho-Kinesis/MTP-1) with a Streamlit entry point for a live web app. |
| next | Thesis report and presentation; optional higher-fidelity (FE/3D or clamped-edge) check. |
