# MTP project files

Updated 9 November 2026 (review fixes, experiments E2–E5 with two pre-registered confirmations, H2: unseen load shapes, and H3: TSDT-verified inverse design).

MTP_Complete_Project_Package.zip contains the completed solver/pilot work plus the earlier research review, Project Initiation Document (PID) and Technical Implementation Document (TID).

## Start here

**Continuing in a new Claude session?** Read SESSION_HANDOFF.md first: it has the current status, open items, project rules and a message to paste.

**Live web app:** run `streamlit run streamlit_app.py` from the repository root, or open mtp-fg-plate/demo/fg_plate_lab_offline.html in a browser (FG Plate Lab: simulator, design studio, evidence).

0. Read CHANGES_2026-11-09.md for what changed in this update and why.
1. Read mtp-fg-plate/docs/PILOT_RESULTS.md for the executed results and their limits.
1a. Read mtp-fg-plate/docs/E2_RESULTS.md for experiment E2 (reduced field baseline) and the recommended next experiment (E3a).
1b. mtp-fg-plate/docs/PILOT_REVIEW_2026-11-09.md is the independent review of the pilot.
1c. Read mtp-fg-plate/docs/E3A_RESULTS.md for experiment E3a (physics-informed vs data-only at matched capacity) and the recommended E3b.
1d. Read mtp-fg-plate/docs/E3B_RESULTS.md for experiment E3b (imperfect FSDT physics + higher-fidelity TSDT labels) and the recommended E3c.
1e. Read mtp-fg-plate/docs/E3C_RESULTS.md for experiment E3c (physics-structured correction: 10 high-fidelity labels beat physics only and data only) and the recommended confirmation run E4.
1f. mtp-fg-plate/docs/E4_RESULTS.md: first pre-registered confirmation on 200 fresh designs (C1–C4 confirmed; tail marginal).
1g. **Final headline result:** mtp-fg-plate/docs/E5_RESULTS.md, the improved base model confirmed on a second set of 200 fresh designs (C1–C6 confirmed; median 0.062%, p95 0.33% with 10 labels). Pre-registration: docs/E5_PREREGISTRATION.md.
1h. **Unseen load shapes:** mtp-fg-plate/docs/H2_RESULTS.md (pre-registered H2, 200 fresh designs, Q1–Q5 confirmed: with the same 10 labels, median 0.06–0.07% and p95 0.33–0.40% on uniform pressure and Gaussian patches). Summary figure: mtp-fg-plate/figures/h2_load_families_summary.png.
1i. **Inverse design:** mtp-fg-plate/docs/H3_RESULTS.md (pre-registered H3, Q1–Q5 confirmed: minimum-mass plates designed with the surrogate are within 0.14% of the best mass and 0.23% of the limits after one TSDT check, at about 2% of the cost of optimizing TSDT directly). Figures: mtp-fg-plate/figures/h3_main_problem_front.png and h3_cost.png.
2. Read mtp-fg-plate/README.md for installation, examples and commands.
3. Read mtp-fg-plate/docs/DATASET_SCHEMA.md to use the dataset and saved model.
4. Use planning/pdf/ for the research review, PID and TID; planning/editable/ contains their Markdown sources.

The code package contains 200 designs and 600 field cases, frozen training/validation/test partitions, the fitted scalar model, numerical validation logs, all five-field modal solutions, plotted shape spectra and benchmark citations.

The planning documents are dated 7 November and describe the proposed broader project. The new pilot report records what is implemented as of 9 November. The original Claude context is retained as historical context in docs/PROJECT_CONTEXT.md with an update at its top. Its older claims do not override the new validation evidence.

Research PDFs supplied by the user are not duplicated in this generated-files package. Python dependencies and temporary working files are also excluded; installation requirements are included.
