from __future__ import annotations

import os
import re
from pathlib import Path


def _streamlit_secret(name: str) -> str:
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:
        return ""
    return str(value).strip() if value else ""


def _parse_token_file(path: Path, name: str) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def get_config_value(name: str, default: str = "") -> str:
    value = _streamlit_secret(name) or os.getenv(name, "")
    return str(value).strip() if value else default


def get_hf_token(name: str, *, token_file: Path | None = None, fallback_name: str = "HF_TOKEN") -> str:
    value = _streamlit_secret(name) or os.getenv(name, "")
    if value:
        return str(value).strip()

    fallback = _streamlit_secret(fallback_name) or os.getenv(fallback_name, "")
    if fallback:
        return str(fallback).strip()

    if token_file is None:
        return ""
    token = _parse_token_file(token_file, name)
    if token:
        return token
    return _parse_token_file(token_file, fallback_name)


def get_storage_backend() -> str:
    backend = get_config_value("STORAGE_BACKEND").lower()
    if backend:
        return backend
    return "hf" if get_config_value("HF_DATASET_REPO") else "local"


def get_hf_dataset_repo(default: str = "ElectronicHug/short_video_ocr_dataset") -> str:
    return get_config_value("HF_DATASET_REPO", default)
