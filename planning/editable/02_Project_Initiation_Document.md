# Project initiation document

Data efficient physics informed surrogates for functionally graded plates

PID version 1.0 | 7 November 2026 | Proposed scope for supervisor review

## Purpose and proposed decision

Develop and critically evaluate machine-learning surrogates for static bending and free vibration of functionally graded ceramic-metal plates. Use a verified mechanics solver to generate reference data, test whether adding physical constraints reduces the need for labelled solutions, and demonstrate constrained inverse design with final designs checked by a solver.

The recommended decision is to retain the existing FG plate direction and approve a staged comparative study. The project will measure the benefit of ML rather than assume that it is faster or more accurate than the already inexpensive analytical solution. A single clamped-boundary extension can strengthen the work if its reference model is validated early enough.

## Project identity and responsibilities

| Item | Definition |
| --- | --- |
| Project | MTP on physics-informed learning for graded plates |
| Student | Aditya, as recorded in the supplied context |
| Institution | IIT Kharagpur, as recorded in the supplied context |
| Academic sponsor | Project supervisor; name to be confirmed |
| Student responsibility | Mechanics verification, software, experiments, records and thesis |
| Supervisor responsibility | Scope, academic novelty, validation standards and milestone review |
| Planning basis | Relative work weeks; actual submission dates remain to be confirmed |

## Problem and motivation

A functionally graded plate changes material composition through its thickness. Its deflection and vibration depend on geometry, material gradation, loading and supports. Repeated evaluations arise in design and sensitivity studies. Machine learning can approximate these relationships, but its value depends on the cost and complexity of the underlying mechanics problem.

The supplied code already solves a restricted family of simply supported, linear elastic FSDT plates. Its 23 supplied checks were rerun successfully on 7 November 2026. This provides a practical starting point for controlled learning experiments. It does not yet provide a dataset, trained model, independent FEM validation or inverse-design tool.

Related work already combines PINNs, surrogates and inverse design for FG plates. Therefore, the broad combination is not the novelty claim. The thesis will test a narrower question about label efficiency, generalization to unseen load configurations, reduced-basis limitations and verified design feasibility. The research review gives the paper-by-paper evidence and closest prior work.

## Research objectives

1. Establish a reproducible FSDT reference model with explicit supports, units, convergence checks and traceable benchmarks.
2. Compare simple regression, reduced-basis regression and physics-informed field learning at matched label budgets.
3. Separate interpolation performance from held-out parameter intervals and held-out load families.
4. Demonstrate a mass-constrained or minimum-mass design study with deflection and frequency requirements, followed by direct solver checks.
5. If feasible, assess adaptation to one clamped-plate model using an independently verified solver.

Hypothesis H1: physical constraints improve errors at small label budgets compared with the same network trained only on data. Hypothesis H2: those gains persist on withheld load families within a stated admissible load class. Hypothesis H3: surrogate-guided optimization reaches solver-feasible designs with competitive total computational cost. Each hypothesis can be rejected; a well-supported negative result remains a valid research outcome.

## Scope and boundaries

The core uses rectangular Al/alumina plates, through-thickness power-law gradation, constant Poisson ratio, linear small-deflection FSDT and a fixed shear correction factor. Start with simply supported Navier-compatible supports. Compute bending fields and fundamental flexural frequency. Use sinusoidal and uniform loads for verification, then smooth multi-mode load combinations for field learning.

The recommended extension is a separate fully clamped rectangular-plate family, retaining the same basic material and linear kinematics. Train and assess that family explicitly; generalization from simply supported training data to clamped plates will not be assumed. Begin with 20-30 verification cases before committing to a larger dataset.

Buckling, geometric nonlinearity, blast response, viscoelastic GOEAM, temperature-dependent properties, irregular geometry, manufacturing experiments and live digital twins are outside the initial scope. Select at most one replacement extension if the supervisor prefers a different scientific direction.

## Deliverables

| Deliverable | Required contents | Completion evidence |
| --- | --- | --- |
| Literature and gap review | Supplied papers, closest external work, precise research question | Supervisor-reviewed comparison |
| Reference solver package | Validity limits, benchmarks, convergence and input checks | Reproducible test report |
| Dataset release | Case metadata, units, seeds, grouped splits and provenance | No leakage; regeneration instructions |
| Baseline comparison | Direct solver, scalar regression and reduced field models | Held-out metrics and timing |
| Physics-informed study | Matched models, label budgets and ablations | Multiple-seed learning curves |
| Inverse-design study | Objectives, constraints, candidates and reanalysis | Feasibility verified by solver |
| Thesis and presentation | Methods, evidence, limitations and reproducibility | Department format and submission |

## Success criteria

These are proposed targets to be ratified after the pilot, not achieved results. The solver must preserve all 23 existing checks and pass new load, parameter and support tests. Dataset splitting must occur by complete physical design, keeping all derived coordinates and load variants in the same partition where required by the experiment.

For in-domain reference-solver data, target median relative field L2 error no greater than 1 percent and 95th-percentile error no greater than 3 percent. Target median first-frequency error no greater than 1 percent and 95th-percentile error no greater than 2 percent. Use normalized absolute errors when reference values are near zero. Report all stress-test errors rather than imposing an unrealistic extrapolation guarantee.

The physics-informed comparison should use at least three random training seeds and the same labels, splits and evaluation cases as its data-only counterpart. A positive finding requires lower error at a matched label budget or fewer labels at a matched target accuracy. If that benefit does not appear, document where and why. No numerical success criterion depends on publication acceptance.

Every reported optimized design must satisfy the declared constraints when reevaluated by the reference solver, including a separately stated numerical tolerance. Independent same-theory FEM comparisons should target approximately 2 percent on selected quantities after convergence; discrepancies with higher-order or 3D models must be reported as model-form differences rather than hidden inside ML error.

## Work plan and decision gates

| Relative weeks | Work | Review gate |
| --- | --- | --- |
| 1-2 | Novelty review, supports, solver hardening, pilot dataset | Reference formulation accepted |
| 3-4 | Scalar and modal baselines; field-rank study; FEM pilot | Dataset and baselines credible |
| 5-6 | Single-case energy model and data-only field model | Physical loss verified |
| 7-8 | Physics-informed comparison and label-budget study | Continue or simplify architecture |
| 9-11 | Optional clamped family and transfer experiments | Extension has validated labels |
| 12-13 | Inverse design and solver reanalysis | Candidates satisfy constraints |
| 14-16 | Consolidation, thesis, reproducibility and presentation | Final academic review |

This is a planning envelope, not a claim about the remaining academic calendar. If only 6-8 weeks are available, finish the simply supported comparative study and a verified design example, and remove the extension. If two semesters remain, align the first three gates with MTP-1 and the remaining gates with MTP-2.

## Resources and dependencies

CPU computation is sufficient for the current reference solver and early baselines. A GPU would help repeated PINN and field-model experiments but is not a dependency for beginning the work. Required software includes a pinned Python environment, numerical libraries, an ML library and versioned experiment configurations. Use an available laboratory FEM package or an agreed open implementation for independent verification. Confirm licence access before scheduling large FE datasets.

No monetary budget is assigned because available hardware, licences and institutional access have not been confirmed. The student should record actual solve and training times during the pilot to set the later compute budget. Each expanded experiment requires a defined purpose and stopping rule.

## Risks and responses

| Risk | Response and decision rule |
| --- | --- |
| Broad topic already published | Compare closest papers before finalizing claims |
| Networks learn a nearly fixed shape | Audit snapshot rank; introduce varied load shapes |
| No speed advantage over Navier | Keep direct solver baseline; focus on measured data efficiency |
| Physics loss shares modelling errors | Independent quadrature, residual and FEM checks |
| Data leakage | Group by design; fit preprocessing and POD on training only |
| Clamped extension takes too long | Drop it if verification is not ready at its gate |
| Optimizer exploits surrogate error | Reanalyse candidates and add corrective samples |
| Excessive scope | One material family, one main response study and one optional extension |

## Governance and pending decisions

Hold a short weekly progress review using evidence: completed benchmark, error curve, convergence plot or design check. Maintain an experiment log with versioned code, parameter ranges and datasets. Any change to mechanics, supports or fidelity creates a separately identified dataset and requires new validation.

Before fixing the thesis scope, settle the actual deadline, supervisor's preferred contribution, GPU availability, FEM access, required report template and whether the clamped extension is desired. These are planning decisions; they do not prevent starting solver provenance and the pilot dataset.

## Reference basis

The accompanying Research review and recommended project direction contains the full numbered bibliography. Core sources are Jin et al. [1] for Navier mechanics, Zhang et al. [5] for case-level splitting and field compression, Yang et al. [10] for energy training, Kim et al. [11] for reduced operator learning, Ly et al. [15] for plate energy formulations and Zhang et al. [16] for broader plate validation.

Closest external overlap: Engineering Structures 357 (2026), 122578, https://doi.org/10.1016/j.engstruct.2026.122578; Engineering Structures 343 (2025), 121090, https://doi.org/10.1016/j.engstruct.2025.121090. These motivate the narrower comparative contribution.
