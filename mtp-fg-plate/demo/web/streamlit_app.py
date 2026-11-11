"""FG Plate Lab as a Streamlit app: serves the same self-contained page (index.html) inside Streamlit.

Run locally:   pip install -r requirements.txt && streamlit run streamlit_app.py
Deploy:        share.streamlit.io -> New app -> this repository -> main file streamlit_app.py
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FG Plate Lab", page_icon="🧱", layout="wide")
st.markdown("<style>.block-container{padding:0.6rem 0.8rem 0 0.8rem;max-width:100%}header[data-testid='stHeader']{display:none}"
            "footer{display:none}</style>", unsafe_allow_html=True)
html = (Path(__file__).parent/"index.html").read_text(encoding="utf-8")
components.html(html, height=1900, scrolling=True)
