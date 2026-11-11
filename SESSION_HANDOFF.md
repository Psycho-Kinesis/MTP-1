# Session handoff: continue the MTP project in a new session

Written 9 November 2026 at the end of a long Claude Code session. This file, together with this package, is everything a new session needs. Read this first, then START_HERE.md.

## Whose project, and what it is
- **Student:** Aditya, final-year MTP (Master's Thesis Project), IIT Kharagpur.
- **Collaborator:** "Astra", who wrote the research review, PID and TID (`planning/`) and delivered the 200-design pilot.
- **Topic source:** 16 research papers plus an editorial-board page from *Advances in Engineering Software* Vol. 220 (2026). The PDFs are not in this package; the summaries are in `mtp-fg-plate/docs/PROJECT_CONTEXT.md`.
- **Working title:** *Data-efficient physics-informed surrogates for bending and vibration analysis and inverse design of functionally graded plates*.
- **Plain-language summary:** we predict how ceramic–metal (Al/Al₂O₃) graded plates bend and vibrate. A cheap physics model (FSDT) plus a small correction learned from only 10 expensive high-accuracy solutions (TSDT) is as accurate as 140 solutions used without physics. It then designs light plates that pass the high-accuracy check.

## How to resume (new session)
1. Start the new session with the GitHub repository `Psycho-Kinesis/MTP-1` selected (or upload the latest zip and extract it in a new empty folder).
2. Install and check, from the project folder:
   ```bash
   cd mtp-fg-plate          # inside the repository (or the extracted package)
   python -m pip install -r requirements-tested.txt
   python validation/run_checks.py        # expect: mechanics PASS (27/27), unit tests PASS (35), artifact audit PASS
   ```
3. Read `START_HERE.md`, then `mtp-fg-plate/docs/PROJECT_CONTEXT.md` (status line and log at section 12), then the results reports in order: E2 → E3A → E3B → E3C → E4 → E5 → H2 → H3 (`mtp-fg-plate/docs/*_RESULTS.md`).

## Status at handoff (all executed and verified unless marked)
| Step | Result | Report |
|---|---|---|
| Solver | FSDT and Reddy TSDT Navier solvers. TSDT bending is verified against the FSDT limit, a symbolic Ritz derivation and published values. TSDT vibration is verified against the FSDT limit, symbolic Ritz (6e-14) and published frequencies (4 digits) | README, H3_RESULTS |
| Pilot | 200 designs, 600 field cases, frozen 140/30/30 split (`results/pilot_200`, never regenerate) | PILOT_RESULTS, PILOT_REVIEW_2026-11-09 |
| E2–E3c | Structure matters; physics replaces labels; model-form gap; structured (k h)² correction | E2…E3C_RESULTS |
| E4, E5 | Pre-registered confirmations on fresh designs. Final pipeline (M-S@P2): 10 TSDT labels give median 0.062%, p95 0.33% | E4_RESULTS, E5_RESULTS |
| H2 | Unseen load shapes (uniform, wide and narrow patches). The pipeline extended to 15×15 modes with the same 10 labels gives median 0.06–0.07%, p95 0.33–0.40%. Q1–Q5 confirmed | H2_RESULTS |
| H3 | Inverse design (minimum mass; deflection, frequency and ceramic-use limits). Designs are within 0.14% of the optimum mass and 0.23% of the limits (p95). One or two TSDT checks make every design feasible. Cost is 140–153 TSDT analyses against 7,938 for direct TSDT optimization. Q1–Q5 confirmed | H3_RESULTS |

## Open items
1. ~~H3 determinism rerun~~ **Done (10 Nov):** a full rerun reproduced every decision, every Part A/B record and `h3_errors.npz` exactly.
2. **Next work the student can choose:**
   - (a) the MTP report and presentation (recommended; E2–E5 + H2 + H3 answer all three PID hypotheses);
   - (b) an optional higher-fidelity check (FE or 3D elasticity) or a clamped-edge plate.
3. **Still unverified:** the original Zenkour table number (see `validation/benchmark_references.json`). The published TSDT frequencies are quoted from secondary sources; the original Hosseini-Hashemi table was not inspected.

## Where the project lives now
- **GitHub repository:** https://github.com/Psycho-Kinesis/MTP-1 (all code, docs, results, the web app). A new session can open this repository directly instead of uploading a zip.
- **Live web app:** https://psycho-kinesis.github.io/MTP-1/ (GitHub Pages, served from the `gh-pages` branch, a copy of `mtp-fg-plate/demo/web/index.html`; after changing the app, rebuild and copy that file to `gh-pages`). `streamlit_app.py` at the repository root can also be deployed on share.streamlit.io from the owner's account. The same app is `mtp-fg-plate/demo/fg_plate_lab_offline.html` (works offline) and `mtp-fg-plate/demo/web/index.html` (any static host, e.g. GitHub Pages).
- **App internals:** `mtp-fg-plate/demo/` (see CHANGES, ninth update). Rebuild after model changes with `python experiments/export_demo_model.py && python demo/build_demo.py && node demo/test_demo_math.js`.

## Package history (all zip files of this project)
| File | Made by | Contents |
|---|---|---|
| `mtp-fg-plate-handoff.zip` (5 Nov) | Claude | First handoff to the collaborator: Navier FSDT solver, validation, demo, PROJECT_CONTEXT |
| `MTP_Project_Planning_Package.zip` (6 Nov) | Astra | Research review, PID, TID (`planning/`) |
| `MTP_Complete_Project_Package.zip` (9 Nov) | Astra | Solver hardening and the 200-design pilot |
| `MTP_Complete_Project_Package_v2.zip` | Claude | Pilot review fixes (pilot-1.1, audit, bootstrap CIs) + E2 |
| `MTP_Complete_Project_Package_v3.zip` | Claude | + E3a |
| `MTP_Complete_Project_Package_v4.zip` / `MTP_Project_v4.zip` | Claude | + TSDT solver and E3b (same content; v4 re-packed with one top folder) |
| `MTP_Project_v5.zip` | Claude | + E3c |
| `MTP_Project_v6.zip` | Claude | + E4 confirmation |
| `MTP_Project_v7.zip` | Claude | + E5 confirmation |
| `MTP_Project_v8.zip` | Claude | + H2 (unseen load shapes) |
| `MTP_Project_v9.zip` | Claude | + TSDT vibration, H3 (inverse design), SESSION_HANDOFF.md |
| GitHub `Psycho-Kinesis/MTP-1` | Claude | Everything above + FG Plate Lab web app, demo video, H3 rerun confirmation |

## Rules this project follows (keep them)
- **Pre-registration first.**
  - Every confirmatory experiment has `docs/*_PREREGISTRATION.md`, written before its test data exist. Its sha256 is asserted in the run script.
  - Frozen arms are refitted and asserted identical to development before any evaluation.
  - Evaluate once; report every outcome, confirmed or not; no re-runs with changed settings.
- **Never regenerate or modify `results/pilot_200`** (generator pilot-1.0). New data uses generator `pilot-1.1`.
- **One heavy experiment at a time, default BLAS threads.** Setting `OPENBLAS_NUM_THREADS=2` changes floating-point results enough to fail the 1e-9 reproduction assertions; this happened once in H3, before any design was evaluated.
- **Determinism:** rerun each experiment into a scratch folder and compare before delivering.
- **Honesty in docs:** distinguish completed from planned work; check every number in a report against the saved manifest; state caveats (TSDT is a plate theory, not 3D; Navier analyses are cheap, so cost is counted in analyses, not wall-clock).
- **Delivery to the student after every finished step:**
  - a complete versioned zip `MTP_Project_vN.zip` whose single top folder is `MTP_Project_vN`, rebuilt from the whole project with `FILE_MANIFEST.json` regenerated and verified from a clean extraction;
  - he saves it into `Desktop\mtp1\project` on his Windows PC and uses Extract All;
  - also send the key new figures.
  - The next version after this package is **v10**.
- **Explanations to the student:** simple, step-by-step language; he prefers easy explanations.

## Frozen hashes (sha256)
| File | sha256 |
|---|---|
| `mtp-fg-plate/results/pilot_200/dataset.npz` | `4c812a0791a8df541b12050ff9b11dd5f0bb88410656c5e9f72cef5c19a85f2d` |
| `docs/E4_PREREGISTRATION.md` | `a60526d7dc6bb11c74336de4ca9ffbe7bb016710689ac69dafc22abb8f03f589` |
| `docs/E5_PREREGISTRATION.md` | `f25d9ea8964659e86be4519d543471c39f77aa222f2dbed88fe5bb056779864e` |
| `docs/H2_PREREGISTRATION.md` | `5b8e0ef2061a8104307a818deca1089a094ab2f51838fde69aef36956b35c088` |
| `docs/H3_PREREGISTRATION.md` | `83f35c3ee0526a3ee6797f3b201d120d5f90529b78fb12df4ae2813bc98f0251` |

`FILE_MANIFEST.json` lists the sha256 of every file in the package.

## Message to paste into the new session
> I'm continuing my MTP project from a previous session. The project is in my GitHub repository Psycho-Kinesis/MTP-1. Please read SESSION_HANDOFF.md and START_HERE.md, install the requirements, and run mtp-fg-plate/validation/run_checks.py. Then tell me in simple words where the project stands and what we should do next. Keep the project's rules from SESSION_HANDOFF.md, and after every finished step give me a complete MTP_Project_vN.zip (next is v10) to save in Desktop\mtp1\project.
