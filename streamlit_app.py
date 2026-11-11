"""FG Plate Lab — live web app for the MTP project (Streamlit entry point).

Deploy: share.streamlit.io -> Create app -> repository Psycho-Kinesis/MTP-1, branch main, main file streamlit_app.py
Local:  pip install -r requirements.txt && streamlit run streamlit_app.py

The app itself is a self-contained page (mtp-fg-plate/demo/web/index.html) that runs the project's FSDT/TSDT solvers and
the frozen physics-informed surrogates in the browser; Streamlit only serves it.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FG Plate Lab", page_icon="🧱", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    "<style>.block-container{padding:0.4rem 0.6rem 0 0.6rem;max-width:100%}"
    "header[data-testid='stHeader']{display:none}footer{display:none}</style>",
    unsafe_allow_html=True,
)
page = Path(__file__).parent/"mtp-fg-plate"/"demo"/"web"/"index.html"
components.html(page.read_text(encoding="utf-8"), height=1900, scrolling=True)
