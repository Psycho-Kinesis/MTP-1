# Physics-Informed Learning for Functionally Graded Plates

MTP project, IIT Kharagpur. Accurate bending and vibration prediction and lightweight design of functionally graded
Al/Al₂O₃ plates from **ten** high-fidelity solutions.

**Idea.** A cheap plate theory (FSDT) is learned without any data from its energy equations. Only its error against the
accurate third-order theory (TSDT), which grows like (k·h)², is learned from 10 TSDT solutions. The resulting model
predicts deflection fields and natural frequencies for any design and smooth load, and designs minimum-mass plates that
pass a TSDT check.

## Live web app: FG Plate Lab
**Open it in any browser: https://psycho-kinesis.github.io/MTP-1/**

`streamlit_app.py` serves the interactive lab (plate simulator, design studio, evidence, method). All mechanics run in
the browser. To run it yourself:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Without Python, open `mtp-fg-plate/demo/fg_plate_lab_offline.html` in any browser (works offline). A recorded tour is in
`mtp-fg-plate/demo/FG_Plate_Lab_demo.mp4`.

## Key results (pre-registered, evaluated once on fresh designs)
| Experiment | Result |
|---|---|
| E5 (200 fresh designs) | 10 TSDT solutions: median field error **0.062 %**, 95th percentile **0.33 %**; data-only ML with the same 10: about 19 % |
| H2 (unseen load shapes) | uniform pressure and Gaussian patches: median 0.06–0.07 %, 95th percentile ≤ 0.40 % with the same 10 solutions |
| H3 (inverse design, 100 problems) | designs within 0.14 % of the optimal mass and 0.23 % of the limits (p95); one TSDT check makes every design feasible; **153** TSDT analyses vs **7,938** for direct optimization (1.9 %) |

Scope: simply supported rectangular plates, linear elasticity, smooth loads; TSDT (a plate theory) is the accurate
reference, standing in for FEM or 3D analysis.

## Repository layout
| Path | Contents |
|---|---|
| `START_HERE.md`, `SESSION_HANDOFF.md` | where to begin; status, rules, history |
| `mtp-fg-plate/` | code (`fgplate/`), experiments, validation, results, docs, the web app (`demo/`) |
| `mtp-fg-plate/docs/` | results reports E2–E5, H2, H3 and pre-registrations |
| `planning/` | research review, PID, TID (collaborator Astra) |
| `CHANGES_2026-11-09.md` | change log of every update |

Verification: `cd mtp-fg-plate && pip install -r requirements-tested.txt && python validation/run_checks.py`
(27 mechanics checks, 35 unit tests, artifact audit).
