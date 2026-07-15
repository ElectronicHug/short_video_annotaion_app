from __future__ import annotations

import streamlit as st

from annotation_app.common.auth import logout, require_login


st.title("Розмітка OCR для коротких відео")
active_user = require_login(form_key="app_login_form")
if active_user is None:
    st.stop()

with st.sidebar:
    st.caption(f"Профіль: {active_user['display_name']}")
    st.caption(f"Роль: {active_user['role']}")
    if st.button("Вийти", use_container_width=True):
        logout()
        st.rerun()

pg = st.navigation(
    [
        st.Page("pages/1_Funnel.py", title="Відбір відео"),
        st.Page("pages/2_Text_Frame_Correction.py", title="Виправлення тексту"),
        st.Page("pages/3_Stats.py", title="Статистика"),
    ]
)
pg.run()
