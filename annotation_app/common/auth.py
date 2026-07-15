from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import extra_streamlit_components as stx
import streamlit as st

from .hf_tokens import get_config_value


COOKIE_NAME = "short_video_ocr_auth"
COOKIE_DAYS = 1


def streamlit_secret_value(name: str) -> Any:
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def load_auth_users() -> dict[str, dict[str, str]]:
    raw_users = streamlit_secret_value("auth_users")
    if not isinstance(raw_users, Mapping):
        return {}

    users: dict[str, dict[str, str]] = {}
    for user_id, raw_profile in raw_users.items():
        if isinstance(raw_profile, Mapping):
            password = str(raw_profile.get("password", ""))
            display_name = str(raw_profile.get("display_name", user_id))
            role = str(raw_profile.get("role", "annotator"))
        else:
            password = str(raw_profile)
            display_name = str(user_id)
            role = "annotator"
        if password:
            users[str(user_id)] = {
                "password": password,
                "display_name": display_name,
                "role": role,
            }
    return users


def current_annotator_id() -> str:
    return str(st.session_state.get("auth_user_id") or get_config_value("ANNOTATOR_ID", "default"))


def cookie_manager() -> stx.CookieManager:
    manager = st.session_state.get("_short_video_ocr_cookie_manager")
    if manager is None:
        manager = stx.CookieManager(key="short_video_ocr_cookie_manager")
        st.session_state["_short_video_ocr_cookie_manager"] = manager
    return manager


def auth_signing_secret(users: dict[str, dict[str, str]]) -> str:
    configured = get_config_value("AUTH_COOKIE_SECRET")
    if configured:
        return configured
    password_material = "|".join(
        f"{user_id}:{profile.get('password', '')}"
        for user_id, profile in sorted(users.items())
    )
    return hashlib.sha256(password_material.encode("utf-8")).hexdigest()


def sign_payload(payload: dict[str, Any], secret: str) -> str:
    raw_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded_payload = base64.urlsafe_b64encode(raw_payload).decode("ascii")
    signature = hmac.new(secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{encoded_payload}.{signature}"


def verify_token(token: str, secret: str) -> dict[str, Any] | None:
    if "." not in token:
        return None
    encoded_payload, signature = token.rsplit(".", 1)
    expected = hmac.new(secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload.encode("ascii")).decode("utf-8"))
    except Exception:
        return None
    expires_at = payload.get("expires_at")
    if not isinstance(expires_at, str):
        return None
    try:
        expires_dt = datetime.fromisoformat(expires_at)
    except ValueError:
        return None
    if expires_dt.tzinfo is None:
        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
    if expires_dt <= datetime.now(timezone.utc):
        return None
    return payload if isinstance(payload, dict) else None


def set_auth_cookie(user_id: str, users: dict[str, dict[str, str]]) -> None:
    expires_at = datetime.now(timezone.utc) + timedelta(days=COOKIE_DAYS)
    token = sign_payload(
        {
            "user_id": user_id,
            "expires_at": expires_at.isoformat(),
        },
        auth_signing_secret(users),
    )
    cookie_manager().set(COOKIE_NAME, token, expires_at=expires_at)


def clear_auth_cookie() -> None:
    cookie_manager().delete(COOKIE_NAME)


def restore_auth_from_cookie(users: dict[str, dict[str, str]]) -> None:
    if st.session_state.get("auth_user_id") in users:
        return
    token = cookie_manager().get(COOKIE_NAME)
    if not token:
        return
    payload = verify_token(str(token), auth_signing_secret(users))
    if not payload:
        clear_auth_cookie()
        return
    user_id = str(payload.get("user_id") or "")
    if user_id in users:
        st.session_state["auth_user_id"] = user_id


def current_user(users: dict[str, dict[str, str]] | None = None) -> dict[str, str] | None:
    users = users or load_auth_users()
    if not users:
        annotator_id = get_config_value("ANNOTATOR_ID", "default")
        return {
            "id": annotator_id,
            "display_name": annotator_id,
            "role": "local",
        }
    restore_auth_from_cookie(users)
    user_id = st.session_state.get("auth_user_id")
    if user_id in users:
        return {"id": str(user_id), **users[str(user_id)]}
    return None


def logout() -> None:
    st.session_state.pop("auth_user_id", None)
    clear_auth_cookie()


def require_login(*, form_key: str = "login_form") -> dict[str, str] | None:
    users = load_auth_users()
    active_user = current_user(users)
    if active_user is not None:
        return active_user

    st.subheader("Вхід")
    with st.form(form_key):
        login = st.text_input("Логін").strip().lower()
        password = st.text_input("Пароль", type="password")
        submitted = st.form_submit_button("Увійти", type="primary", use_container_width=True)

    if submitted:
        user = users.get(login)
        expected = user["password"] if user else ""
        if user and hmac.compare_digest(password, expected):
            st.session_state["auth_user_id"] = login
            set_auth_cookie(login, users)
            return {"id": login, **user}
        st.error("Неправильний логін або пароль")
    return None
