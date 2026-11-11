# FG Plate Lab

Interactive simulator and design studio for functionally graded (Al/Al₂O₃) plates, from the MTP project
*Physics-informed learning for functionally graded plates* (IIT Kharagpur).

The page runs everything in the browser: the Navier FSDT and TSDT plate solvers, the natural-frequency solver, and the
physics-informed ML model (trained with 10 TSDT solutions), compared live. It has four views: plate simulator, design
studio (lightest plate under deflection, frequency and ceramic limits, checked with TSDT), evidence, and how it works.

- **Website (GitHub Pages):** `index.html` is the whole app; open it in any browser.
- **Streamlit:** `streamlit run streamlit_app.py` (or deploy on share.streamlit.io with `streamlit_app.py`).

Scope: simply supported rectangular plates, linear elasticity, smooth loads; TSDT is the accurate reference.
