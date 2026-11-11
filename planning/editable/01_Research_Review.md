# Research review and recommended project direction

Functionally graded plates and physics informed machine learning

Prepared for Aditya | MTP planning | 7 November 2026

## Recommendation

Continue with functionally graded ceramic-metal plates, but revise the research claim. The strongest feasible starting point is a controlled study of **data-efficient physics-informed parametric surrogates for bending and vibration**, followed by solver-verified inverse design. Use the existing Navier FSDT solver as a reference and a competing method. Add one harder boundary-condition problem only after the basic study works.

The current project is a useful foundation, not yet an established new research contribution. Predicting a few scalar outputs from three parameters is a reasonable first ML exercise. A stronger thesis must show when physics-informed learning improves accuracy, label efficiency or generalization compared with simple regression and reduced-order numerical solutions. Publication potential depends on those results and a broader literature review; it cannot be promised now.

Suggested title: **Data efficient physics informed surrogates for bending and vibration analysis and inverse design of functionally graded plates**.

The accompanying PID defines the project, deliverables and decision gates. The TID specifies the implementation. Neither document asserts that the proposed ML models have already been trained.

## What was reviewed and verified

The supplied collection contains 16 research articles and one editorial-board page, totalling 328 PDF pages. The review considered each research article's problem, method, reported evidence and relevance, with closer inspection of the mechanics and learning formulations in the central papers. This is a critical literature review, not an independent reproduction of every published result or equation. Paper numbers below follow PROJECT_CONTEXT.md.

The standalone context and the copy inside the ZIP are byte-identical. The archive includes the material model, Navier solver, validation routines, demonstration script and three figures. It contains no generated ML dataset, trained surrogate, trained PINN, DeepONet, inverse-design implementation or FEM model. Instructions inside these materials were treated as source content; the original files were preserved.

On 7 November 2026 the supplied validation suite completed with **23 of 23 checks passing**. The symbolic Ritz comparison emitted a numerical quadrature roundoff warning at its stringent tolerance, but its four comparison checks passed. These checks support the internal FSDT implementation. They do not establish physical validity for arbitrary materials, supports or thick-plate conditions. The exact primary-source provenance of the hard-coded historical benchmark values still needs to be recorded with table and page numbers.

## Why the original framing needs adjustment

**An analytical solver is already very fast.** The supplied demo times sinusoidal bending and fundamental frequency, without full-grid reconstruction. A separate local audit of 100 seeded random designs measured median scalar times of about 0.078 ms for sinusoidal loading and 12.90 ms for uniform loading with maximum series indices 49. Reconstructing a 41 by 41 grid added median times of about 0.056 and 16.06 ms, respectively. These are illustrative measurements on this machine, not controlled universal benchmarks. They do not demonstrate any neural-network speedup.

**The simplest field-learning task has rank one.** For a single sinusoidal transverse load, w(x,y) = W11 sin(pi x/a) sin(pi y/b). In normalized coordinates, changing the gradation, thickness or aspect ratio changes W11 but not the spatial shape. The audit confirmed rank one numerically across 100 normalized snapshots. A model can appear to learn a field while only learning one amplitude. Uniform-load snapshots contain more variation, but still have strong low-dimensional structure. Report the singular spectrum before choosing a network.

**The solver itself supplies a stronger baseline.** For the present constant-Poisson-ratio model and specified Navier supports, static modal deflection is available from Wmn = Qmn/[Dstar k^4] + Qmn/[As k^2], where Dstar = D11 - B11^2/A11 and k^2 = (m pi/a)^2 + (n pi/b)^2. A vectorized implementation can avoid repeated 5 by 5 solves. Any claimed acceleration should compete with this baseline as well as the original code.

**Novelty is narrower than the handoff suggests.** A gap within one journal issue does not establish a gap in the wider literature. The additional search found closely related work described below. The contribution should be a documented experimental result, such as lower label requirements under held-out load families, or successful transfer to a more expensive independently verified model.

**Physical accuracy and surrogate accuracy are different.** A network matching FSDT labels extremely well inherits FSDT assumptions. The power-law rule of mixtures, constant Poisson ratio, fixed shear factor, small deflection and support definition all constrain the claim. Use “semi-analytical FSDT reference solution” rather than “exact ground truth for real plates.”

## Paper by paper review

### Paper 1 Lattice sandwich impact

Jin et al. couple two-dimensional asymptotic homogenization with Mindlin plate kinematics, Navier series, a mass-spring contact model and Duhamel integration. They compare low-velocity elastic impact results with FEM and published benchmarks, then study lattice dimensions and panel geometry. The equivalent stiffness formulation is reusable beyond the demonstrated BCC lattice when the stated periodicity and equivalent-plate assumptions apply. Their limitations include small deformation, elasticity and omitted thermal loading. For this project, use the Navier formulation and validation logic; do not interpret the paper as direct validation of an Al/alumina FGM constitutive model. Impact and lattice homogenization would introduce substantial new implementation. Evidence: PDF pp. 1, 7-8 and 17-18; Appendix B gives the modal stiffness structure. [1]

### Paper 2 Manufacturing constraints in topology optimization

Cui et al. use stiffener height as the design variable and project it to a density field. They impose constraints associated with superplastic forming, cavity connectivity and bonding regions. Numerical examples show how structural optimization can incorporate manufacturability rather than optimizing compliance alone. The authors explicitly distinguish these geometric constraints from a detailed forming simulation: thickness evolution, strain localization and forming limits remain outside the model, and CAD reconstruction is still needed. This is useful guidance for specifying admissible design variables and constraints in our inverse-design task. It does not require us to implement topology optimization or add a neural network to it. Evidence: PDF pp. 1, 8 and 13-14. [2]

### Paper 3 Coupled gear wear

Mo et al. combine a reduced axial bending-stiffness model with lubrication, friction, wear and load redistribution. The iterative framework is checked against three-dimensional FEM and reports more than a sixtyfold acceleration for its tested comparison. Its central lesson is that a physics-based reduced model can be an effective computational competitor without ML. The coupled feedback also changes the predicted wear evolution, so computational savings cannot justify omitting essential mechanics. The problem is far from the present plate code and would require a new contact and tribology solver. Retain it as background for fair timing and model-reduction comparisons. Evidence: PDF pp. 1, 13, 16 and 24. [3]

### Paper 4 Interval uncertainty in masonry

Sofi et al. represent uncertain mechanical properties by intervals and approximate nonlinear ABAQUS response using a ratio-of-polynomials response surface. The method estimates resistance bounds and associated failure patterns using relatively few deterministic analyses. The motivation is insufficient evidence for reliable probability distributions. This is a useful alternative to automatically assuming normally distributed material parameters. For our plates, begin with bounded sensitivity studies in modulus, thickness and gradation; use probabilistic UQ only if distributions are justified. The masonry model and its nonlinear failure mechanisms do not transfer directly. Computed sampled bounds must not be described as rigorous global bounds without further analysis. Evidence: PDF pp. 1, 3-4 and 16-17. [4]

### Paper 5 Casting digital twin

Zhang et al. combine a POD-RBF reduced temperature model with physics-derived thermal features and ML prediction of shrinkage porosity. The database uses 64 process simulations, and measured temperatures update modal coefficients. Reported classification accuracy exceeds 97 percent and regression R squared reaches 0.98 for their setting. The paper explicitly discusses splitting by simulation case to avoid leakage from correlated nodes, and restricts trained-model transferability to the studied geometry, alloy and process window. Adopt its case-level splits, compact field representation and feature-based baselines. Our project should not be called a digital twin without an actual measurement-update loop. Evidence: PDF pp. 5, 7, 12-13 and 17-18. [5]

### Paper 6 Foundation impact with FEM and ALE

Yosef et al. combine Lagrangian structural modelling with ALE soil modelling in LS-DYNA to handle large soil deformation during impact. Full-scale tests support the modelling and reveal a critical embedment-depth effect. The numerical model involves material calibration, contact and coupling choices; conclusions are tied to the examined dry granular conditions. This paper illustrates why independent physical validation matters and why an apparent numerical speed improvement is not sufficient. It offers little reusable mechanics for the current plate pipeline and would require expensive simulation and specialist calibration. Keep it as background, not an additional work package. Evidence: PDF pp. 1, 6-8 and 16-17. [6]

### Paper 7 Open source fracture and fatigue

Zhang, Zhao and Tang provide a Julia/Gridap implementation joining brittle fracture, elastic fatigue, ductile fracture and elastoplastic fatigue through shared variational and staggered-solution machinery. Eight benchmarks and open source code make this a potentially reproducible alternative project. Its fatigue cost remains substantial. In the reported cycle-jump example, the approximately hundredfold comparison includes an extrapolated no-jump runtime, not a directly completed paired run. This distinction matters when reporting speedups. A fatigue surrogate could be worthwhile if the guide prefers fracture, but it would replace the present mechanics project rather than be a small extension. Evidence: PDF pp. 1, 7-8 and 17-18. [7]

### Paper 8 Robot thermo mechanical optimization

Li et al. use node-based topology optimization, tetrahedral finite elements and adjoint sensitivities for a hydraulic robot leg with weak steady thermo-mechanical coupling. Fabrication and tests provide evidence beyond a purely numerical optimization. The reported shank reductions in mass and inertia, and improved heat dissipation, belong to that comparison and cannot be transferred to an FG plate design. Its useful lesson is to state multiple performance requirements, compare against a specified baseline and independently check the final design. Reproducing the full multiphysics topology pipeline would greatly expand our scope. Evidence: PDF pp. 1, 3-5 and 15-16. [8]

### Paper 9 Finite difference physics informed U Net

Shen and Zhao learn temperature fields from heterogeneous microstructure images. The physics loss compares the predicted field with finite-difference update operations implemented as convolutions; several update steps improve the training signal. The method needs no labelled temperature fields for training, but still uses input microstructures, governing equations and boundary conditions. Effective conductivity is obtained by post-processing the field. This supports testing a discretized physical residual rather than assuming that all physics-informed methods require high-order automatic differentiation. For FSDT plates, construct the correct coupled mechanics residual; copying the thermal stencil is invalid. Inspect performance against an independent reference and a tighter discretization, since training and testing against the same discretization can share bias. Evidence: PDF pp. 1, 3-6, 9-10 and 18. [9]

### Paper 10 Pseudo supervised beam PINNs

Yang et al. solve nonlinear FG graphene-origami beam bending using FSDT, von Karman kinematics, foundation terms and energy-based learning. Two networks exchange stop-gradient pseudo-labels; adaptive Gauss quadrature and encoder transfer improve training. The paper reports sub-percent errors in most cases and a worst displacement-field error below 4 percent. Its very small maximum-deflection error is not a guarantee of equally small error everywhere. Pseudo-supervision is annealed away so the final solution is governed by physics, and the authors discuss the limits of direct energy minimization for multiple equilibrium branches. Adopt careful quadrature, hard essential conditions and ablation studies. Treat two-network training as an optional ablation after a single-network baseline works. Evidence: PDF pp. 1, 3-7 and 16-17. [10]

### Paper 11 Hierarchical acoustic operator learning

Kim et al. use a fixed POD trunk and a parameter-to-coefficient branch network, with data pretraining followed by physics-informed refinement. The formulation includes acoustic PDE and interface conditions, and treats real and imaginary fields. The paper reports improved MSE and approximately millisecond inference relative to its acoustic FEM comparisons. Generalization tests remain within a controlled perturbation envelope, including source shifts up to approximately five percent of cavity width. A fixed POD basis cannot represent an arbitrarily new response outside its span. This is the main architecture reference for our project, but acoustics timings and accuracy cannot be assigned to plates. Explicitly measure POD projection error and distinguish parameter interpolation from new load-family tests. Evidence: PDF pp. 3-5 and 14-15. [11]

### Paper 12 Peridynamic dispersion

Yu and Noh analyse spatial and temporal dispersion together and compare the dissipative Noh-Bathe method with central differences for peridynamic and coupled PD-FE wave propagation. Damping inaccurate high-frequency content reduces interface reflections and improves consistency of simulated crack paths. The paper also cautions that excessive numerical dissipation can remove physically relevant components; agreement between two calculations does not establish a uniquely correct fracture path. This matters if transient response is added later: time-step and dissipation studies must accompany any ML comparison. It provides no direct reason to add peridynamics to the plate thesis. Evidence: PDF pp. 1, 4-7 and 17-18. [12]

### Paper 13 TPMS direct slicing

Huang et al. present TPMSuite, which constructs laser paths from implicit TPMS geometry, with overhang-aware processing and parallelization. Reported reductions in preprocessing time and memory address manufacturing geometry representation rather than mechanics prediction. Printed examples support the manufacturing method. The software still uses boundary representations in parts of its workflow, so “no STL anywhere” would overstate its scope. For this project it is background on representation choice and reproducible software. A TPMS mechanics or manufacturing surrogate would be a different project requiring a new geometry-to-response dataset. Evidence: PDF pp. 1, 4-8 and 18-19. [13]

### Paper 14 Transient viscoelastic graphene origami plates

Karamanli develops three higher-order finite-element plate models with Kelvin-Voigt viscoelasticity and studies step and blast loading across graphene distribution, hydrogen coverage, restraint and damping choices. It is a useful eventual extension because spatial and temporal responses are richer than a single static sine shape. However, it introduces a new effective material model, thickness kinematics and time integration. It cannot be implemented by changing only the current Young modulus and density. Constitutive calibration and domain validity must be established. Use it as a stretch route only after the ceramic-metal study succeeds; do not simultaneously add GOEAM, nonlinearity, transients and new supports. Evidence: PDF pp. 1, 22 and 24-25. [14]

### Paper 15 Energy embedded multiphysics plate PINNs

Ly et al. study nonlinear bending, vibration and buckling of FG magneto-electro-elastic plates with zigzag kinematics. The network predicts coupled fields and combines Fourier features, essential-boundary embedding and staged optimization. The plate theory, constitutive couplings and outputs are substantially richer than the existing five-field elastic FSDT model. Its energy formulation is a methodological reference, not a like-for-like numeric benchmark for our material. The authors identify difficulty extracting higher eigenmodes. Adopt a correctly normalized Rayleigh-quotient formulation if learning vibration eigenfields, with orthogonality for higher modes. Do not infer physical accuracy merely from a small optimization loss. Evidence: PDF pp. 1, 6-8 and 19-20. [15]

### Paper 16 Irregular plate vibroacoustics

Zhang et al. map irregular FG plates to a regular domain and use an FSDT spectral formulation, a Rayleigh radiation model and a dynamic vibration absorber. Numerical comparisons support the graded-plate model, while the experimental plate is isotropic and tests geometry and modal behaviour. The paper states that FSDT and a constant shear factor have limitations for thick plates and steep gradients. Its radiation model is not a complete water-loaded fluid-structure model. This is the best mechanics reference for a later boundary/geometry extension or absorber-design problem. That extension needs separate validation and cannot be produced by giving the current SSSS solver a new boundary label. Evidence: PDF pp. 1, 3-5, 10 and 15-16. [16]

The editorial-board PDF lists editors and advisory-board members. It is not research evidence and is excluded from the scientific comparison.

## Additional literature changes the novelty claim

The following primary publisher or author records were found in a targeted web search on 7 November 2026. This search is not exhaustive; full-text comparison of the closest papers remains an early project task.

- **A physics-informed data-driven framework for FGM plates: From forward analysis to inverse design**, Engineering Structures 357 (2026), 122578. The publisher abstract describes an HSDT PINN, a lightweight DNN surrogate and NSGA-II inverse design. This directly overlaps the broad handoff concept. DOI: https://doi.org/10.1016/j.engstruct.2026.122578. Assessment here is based on the publisher abstract and highlights, not full-text reproduction. [A1]
- **Inverse co-design of material and shape in multidirectional functionally graded plates via physics-informed neural networks: Bending and free vibration optimization**, Engineering Structures 343 Part B (2025), 121090. The publisher record describes joint material-shape design and bending/vibration objectives. It rules out claiming that inverse design of FG plates with PINNs is itself new. DOI: https://doi.org/10.1016/j.engstruct.2025.121090. [A2]
- **Physics-informed Fourier neural operator for the bending analysis of bi-directional functionally graded beams with variable cross-sections**, Vo and Lee, European Journal of Mechanics A/Solids 115 (2026), 105798. This already applies physics-informed operator learning to graded structural members. The structural setting and operator representation differ from our proposed plate benchmark. DOI: https://doi.org/10.1016/j.euromechsol.2025.105798. [A3]
- **Applications of Scientific Machine Learning for the Analysis of Functionally Graded Porous Beams**, Eshaghi et al., Neurocomputing 619, 129119; author manuscript arXiv:2408.02698. It compares PINN, energy and neural-operator approaches, including varying material functions and traction. It is useful prior art for a fair comparative study. https://arxiv.org/abs/2408.02698. [A4]

Defensible working question: **Under matched training budgets, when does a physics-informed reduced field surrogate improve label efficiency and load-family generalization for graded plates, and do those gains survive independent solver checks and constrained inverse design?** This is a proposed investigation, not a verified claim of novelty.

## Feasible routes and choice

| Route | Main output | Assessment |
| --- | --- | --- |
| Scalar ML on existing Navier data | Deflection and first-frequency prediction | Fastest starting point; limited novelty alone |
| Varied-load field surrogate with ablations | Learning curves and held-out-load tests | Recommended core; exposes whether physics helps |
| One clamped-plate extension | Independent solver and transfer comparison | Recommended strengthening after a feasibility gate |
| GOEAM transient response | Space-time surrogate and damping study | Valuable stretch; high mechanics and validation effort |
| Lattice impact or phase-field fatigue | New application-specific surrogate | Alternative project, not an extra chapter |

The core sequence should be: repair solver interfaces and benchmark provenance; generate a small audited dataset; fit simple baselines; demonstrate a single-case energy model; compare field surrogates with and without physics; then verify inverse designs. Add the clamped-plate extension if there is enough time and a suitable reference solver. Keep buckling, thermal effects, irregular geometry, UQ and GOEAM out of the initial success criteria.

## Immediate priorities

1. Confirm scope, remaining semester time and the guide's expected contribution. Take [A1]-[A4] to the discussion.
2. Document the precise in-plane and rotational supports represented by the current Navier basis. Retrieve primary table references for the historical validation values.
3. Harden parameter checks and use a symmetric generalized eigenvalue solver; preserve the existing passing results.
4. Generate a 200-case pilot, including varying load mixtures and normalized field singular spectra. Do not start with 20,000 repetitive sine fields.
5. Compare an optimized direct solver, scalar regression and POD coefficient regression before adding DeepONet complexity.
6. Start a small independent FEM verification exercise early, so discrepancies appear before the thesis deadline.

## References for the supplied issue

All numbered references are in Advances in Engineering Software 220 (2026). The DOI suffix is also the article number. Full titles are reproduced below for traceability; the page references above refer to the supplied PDFs.

[1] Jin et al. A coupled 2D homogenization and Navier series method for predicting low-velocity impact response of arbitrary truss-type lattice sandwich panels. https://doi.org/10.1016/j.advengsoft.2026.104241

[2] Cui et al. Algorithmic integration of superplastic forming constraints in the topology optimization of stiffened thin-walled structures. https://doi.org/10.1016/j.advengsoft.2026.104226

[3] Mo et al. An efficient multi-physics computational framework for coupled friction-wear evolution and elastodynamic characterization of helical gears. https://doi.org/10.1016/j.advengsoft.2026.104223

[4] Sofi et al. A response surface approach for uncertainty propagation in unreinforced masonry walls with interval mechanical properties. https://doi.org/10.1016/j.advengsoft.2026.104218

[5] Zhang et al. Digital twin modeling for shrinkage porosity prediction of casting using reduced order model and machine learning. https://doi.org/10.1016/j.advengsoft.2026.104235

[6] Yosef et al. Efficient hybrid FEM+ALE implementation for large deformation impact simulation of shallow foundations in granular media. https://doi.org/10.1016/j.advengsoft.2026.104225

[7] Zhang, Zhao and Tang. From brittle fracture to elastoplastic fatigue: An open-source phase field framework in Julia. https://doi.org/10.1016/j.advengsoft.2026.104222

[8] Li et al. Multi-physical topology optimization for thermo-mechanical coupled designs of hydraulic humanoid robot with experimental analysis. https://doi.org/10.1016/j.advengsoft.2026.104237

[9] Shen and Zhao. Pixel-level finite difference-enhanced physics-informed U-Net for heat conduction and effective properties prediction in heterogeneous composites using digital microstructure images. https://doi.org/10.1016/j.advengsoft.2026.104221

[10] Yang et al. Pseudo-supervised physics-informed neural networks with transfer learning for nonlinear bending analysis of metamaterial beams. https://doi.org/10.1016/j.advengsoft.2026.104239

[11] Kim et al. Scalable parametric acoustic field prediction via hierarchical physics-informed deep operator network with modal compression. https://doi.org/10.1016/j.advengsoft.2026.104234

[12] Yu and Noh. Spatio-temporal dispersion analysis of dissipative time integration schemes for coupled peridynamic wave propagation. https://doi.org/10.1016/j.advengsoft.2026.104236

[13] Huang et al. Tool Path Master Suite (TPMSuite): A Direct Slicing Software for Laser Additive Manufacturing of Triply Periodic Minimal Surface Lattices. https://doi.org/10.1016/j.advengsoft.2026.104219

[14] Karamanli. Transient analysis of viscoelastic graphene origami-enabled auxetic metamaterial plates. https://doi.org/10.1016/j.advengsoft.2026.104220

[15] Ly et al. Variational energy-embedded multiphysics informed neural network framework for nonlinear bending, free vibration, and buckling of magneto-electro-elastic plates. https://doi.org/10.1016/j.advengsoft.2026.104224

[16] Zhang et al. Vibroacoustic characteristics and control of irregular plate structures in sea-launch vessels. https://doi.org/10.1016/j.advengsoft.2026.104233
