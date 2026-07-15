from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="Short Video OCR Annotation", layout="wide")

st.title("Short Video OCR Annotation")
st.caption("Use the sidebar to choose an annotation stage.")

st.markdown(
    """
    ### Available stages

    - **Funnel**: classify videos by text/subtitle matching.
    - **Text Frame Correction**: correct OCR text frame by frame for selected videos.
    """
)
