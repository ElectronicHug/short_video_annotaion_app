from __future__ import annotations

import base64
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from annotation_app.common.auth import current_annotator_id, logout, require_login
from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore
from annotation_app.common.hf_tokens import get_config_value


ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
TARGET_FUNNEL_CATEGORIES = {"matched", "title_matched"}
DEFAULT_CLAIM_TTL_MINUTES = 60
FRAME_CACHE_DIR = ROOT / ".cache" / "text_frames"
TEXT_ANNOTATIONS_CACHE_SECONDS = 60


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def frame_key(video_id: str, frame_id: str) -> str:
    return f"{video_id}/{frame_id}"


def current_session_id() -> str:
    session_id = st.session_state.get("text_annotation_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        st.session_state["text_annotation_session_id"] = session_id
    return str(session_id)


def get_claim_ttl_minutes() -> int:
    raw_value = get_config_value("TEXT_CLAIM_TTL_MINUTES") or get_config_value(
        "CLAIM_TTL_MINUTES",
        str(DEFAULT_CLAIM_TTL_MINUTES),
    )
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_CLAIM_TTL_MINUTES


def is_own_claim(claim: dict[str, Any]) -> bool:
    return (
        str(claim.get("annotator_id") or "") == current_annotator_id()
        or str(claim.get("session_id") or "") == current_session_id()
    )


def locked_video_ids(active_claims: dict[str, dict[str, Any]]) -> set[str]:
    return {video_id for video_id, claim in active_claims.items() if not is_own_claim(claim)}


@st.cache_data(show_spinner=False)
def load_text_rows() -> list[dict[str, Any]]:
    store = HfDatasetStore.from_config(root=ROOT)
    funnel_state = store.load_funnel_state({"decisions": {}})
    selected_videos = {
        video_id
        for video_id, decision in (funnel_state.get("decisions") or {}).items()
        if isinstance(decision, dict) and decision.get("category") in TARGET_FUNNEL_CATEGORIES
    }
    frames = store.load_frames_manifest()
    ocr_by_key = {
        frame_key(str(row.get("video_id")), str(row.get("frame_id"))): row
        for row in store.load_qwen_frame_ocr()
    }
    rows = []
    for frame in frames:
        video_id = str(frame.get("video_id"))
        frame_id = str(frame.get("frame_id"))
        if video_id not in selected_videos:
            continue
        ocr_row = ocr_by_key.get(frame_key(video_id, frame_id), {})
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "video_id": video_id,
                "frame_id": frame_id,
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "source_category": frame.get("source_category"),
                "frame_path": frame.get("frame_path"),
                "frame_gcs_url": frame.get("frame_gcs_url"),
                "ocr_text": clean_text(ocr_row.get("prediction_text")),
                "ocr_raw_text": clean_text(ocr_row.get("raw_text")),
                "ocr_model": ocr_row.get("model", "qwen2_vl_frame_ocr"),
            }
        )
    return sorted(rows, key=lambda row: (row["video_id"], row.get("timestamp_seconds") or 0, row["frame_id"]))


def group_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["video_id"], []).append(row)
    return grouped


def next_video_id(
    grouped_rows: dict[str, list[dict[str, Any]]],
    annotations: dict[str, dict[str, Any]],
    locked_ids: set[str],
) -> str | None:
    current_video_id = st.session_state.get("text_current_video_id")
    if current_video_id in grouped_rows and current_video_id not in locked_ids:
        if any(frame_key(row["video_id"], row["frame_id"]) not in annotations for row in grouped_rows[current_video_id]):
            return str(current_video_id)

    for video_id, video_rows in grouped_rows.items():
        if video_id in locked_ids:
            continue
        if any(frame_key(row["video_id"], row["frame_id"]) not in annotations for row in video_rows):
            return video_id
    return None


def next_frame_index(video_rows: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> int | None:
    current_key = st.session_state.get("text_current_frame_key")
    if current_key:
        for index, row in enumerate(video_rows):
            if frame_key(row["video_id"], row["frame_id"]) == current_key:
                if current_key not in annotations:
                    return index
                break

    for index, row in enumerate(video_rows):
        if frame_key(row["video_id"], row["frame_id"]) not in annotations:
            return index
    return None


def download_frame(row: dict[str, Any]) -> Path:
    frame_path = Path(str(row["frame_path"]))
    local_path = FRAME_CACHE_DIR / frame_path
    if local_path.exists():
        return local_path
    url = row.get("frame_gcs_url")
    if not url:
        raise ValueError(f"No frame_gcs_url for {frame_key(row['video_id'], row['frame_id'])}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(str(url), timeout=(10, 60)) as response:
        response.raise_for_status()
        local_path.write_bytes(response.content)
    return local_path


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_frame(row: dict[str, Any], *, compact: bool = False) -> None:
    image_path = download_frame(row)
    css_class = "previous-frame" if compact else "current-frame"
    st.markdown(
        f"""
        <div class="{css_class}">
          <div class="frame-label">{row['frame_id']} | {row.get('timestamp_seconds')}s</div>
          <img src="{image_data_uri(image_path)}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def previous_annotation_for_video(
    video_rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for previous_index in range(index - 1, -1, -1):
        previous_row = video_rows[previous_index]
        annotation = annotations.get(frame_key(previous_row["video_id"], previous_row["frame_id"]))
        if annotation:
            return previous_row, annotation
    return None, None


def set_textarea_value(key: str, value: str) -> None:
    st.session_state[key] = value


def set_textarea_values(values: dict[str, str]) -> None:
    for key, value in values.items():
        st.session_state[key] = value


def load_text_annotations_cached(store: FirestoreDecisionStore) -> dict[str, dict[str, Any]]:
    now_ts = datetime.now(timezone.utc).timestamp()
    cache = st.session_state.get("text_annotations_cache")
    cache_loaded_at = float(st.session_state.get("text_annotations_cache_loaded_at") or 0)
    if isinstance(cache, dict) and now_ts - cache_loaded_at < TEXT_ANNOTATIONS_CACHE_SECONDS:
        return cache
    annotations = store.load_text_frame_annotations(DATASET_ID)
    st.session_state["text_annotations_cache"] = annotations
    st.session_state["text_annotations_cache_loaded_at"] = now_ts
    return annotations


def update_text_annotations_cache(key: str, annotation: dict[str, Any]) -> None:
    cache = st.session_state.setdefault("text_annotations_cache", {})
    if isinstance(cache, dict):
        cache[key] = annotation
        st.session_state["text_annotations_cache_loaded_at"] = datetime.now(timezone.utc).timestamp()


def save_annotation(
    store: FirestoreDecisionStore,
    row: dict[str, Any],
    *,
    subtitle_text: str,
    static_text: str,
    other_text: str,
    status: str = "accepted",
) -> None:
    annotation = {
        "dataset_id": DATASET_ID,
        "task": "text_frame_correction",
        "video_id": row["video_id"],
        "frame_id": row["frame_id"],
        "timestamp_seconds": row.get("timestamp_seconds"),
        "source_category": row.get("source_category"),
        "frame_path": row.get("frame_path"),
        "frame_gcs_url": row.get("frame_gcs_url"),
        "ocr_text": row.get("ocr_text", ""),
        "ocr_model": row.get("ocr_model", "qwen2_vl_frame_ocr"),
        "subtitle_text": subtitle_text.strip(),
        "static_text": static_text.strip(),
        "other_text": other_text.strip(),
        "status": status,
        "annotator_id": current_annotator_id(),
        "annotated_at": now_iso(),
    }
    store.upsert_text_frame_annotation(
        dataset_id=DATASET_ID,
        video_id=row["video_id"],
        frame_id=row["frame_id"],
        annotation=annotation,
        annotator_id=current_annotator_id(),
    )
    update_text_annotations_cache(frame_key(row["video_id"], row["frame_id"]), annotation)


def main() -> None:
    st.set_page_config(page_title="Text Frame Correction", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; max-width: 1500px; }
        .current-frame, .previous-frame {
            border: 1px solid #d7dde5;
            background: #f8fafc;
            overflow: hidden;
        }
        .current-frame img {
            display: block;
            max-height: 68vh;
            object-fit: contain;
            width: 100%;
        }
        .previous-frame img {
            display: block;
            max-height: 30vh;
            object-fit: contain;
            width: 100%;
        }
        .frame-label {
            color: #475569;
            font-size: 0.85rem;
            padding: 0.35rem 0.55rem;
        }
        .ocr-box {
            border-left: 3px solid #2563eb;
            background: #f8fafc;
            color: #111827;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
            padding: 0.55rem 0.7rem;
            white-space: pre-wrap;
        }
        div.stButton > button { min-height: 2.7rem; font-weight: 650; }
        textarea { font-size: 1rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Text Frame Correction")
    active_user = require_login()
    if active_user is None:
        return

    decision_store = FirestoreDecisionStore.from_config()
    claim_ttl_minutes = get_claim_ttl_minutes()

    try:
        rows = load_text_rows()
    except Exception as exc:
        st.error(f"Failed to load HF frame/OCR data: {type(exc).__name__}: {exc}")
        return

    if not rows:
        st.warning("No frames found for matched/title_matched videos with Qwen OCR outputs.")
        return

    annotations = load_text_annotations_cached(decision_store)
    active_claims = decision_store.load_active_text_video_claims(DATASET_ID)
    locked_ids = locked_video_ids(active_claims)
    grouped_rows = group_by_video(rows)
    video_id = next_video_id(grouped_rows, annotations, locked_ids)

    with st.sidebar:
        st.caption(f"Профіль: {active_user['display_name']}")
        st.caption(f"Роль: {active_user['role']}")
        if st.button("Вийти", use_container_width=True):
            logout()
            st.rerun()
        st.divider()
        st.caption(f"Videos: {len(grouped_rows)}")
        st.caption(f"Frames: {len(rows)}")
        st.caption(f"Annotated frames: {len(annotations)}")
        st.caption(f"Active video locks: {len(locked_ids)}")
        st.caption(f"Lock TTL: {claim_ttl_minutes} min")
        if st.button("Reload HF/OCR data", use_container_width=True):
            load_text_rows.clear()
            st.session_state.pop("text_annotations_cache", None)
            st.session_state.pop("text_annotations_cache_loaded_at", None)
            st.rerun()
        if st.button("Choose next video", use_container_width=True):
            st.session_state.pop("text_current_video_id", None)
            st.session_state.pop("text_current_frame_key", None)
            st.rerun()

    total = len(rows)
    done = len(annotations)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Annotated", done)
    metric_cols[1].metric("Remaining", max(0, total - done))
    metric_cols[2].metric("Frames", total)
    metric_cols[3].metric("Videos", len(grouped_rows))
    if total:
        st.progress(min(1.0, done / total))

    if video_id is None:
        st.success("All available videos are annotated or temporarily locked.")
        return

    claim_result = decision_store.claim_text_video(
        dataset_id=DATASET_ID,
        video_id=video_id,
        annotator_id=current_annotator_id(),
        session_id=current_session_id(),
        ttl_minutes=claim_ttl_minutes,
    )
    if not claim_result.get("claimed"):
        st.info("This video was just taken by another annotator. Choosing another one.")
        st.session_state.pop("text_current_video_id", None)
        st.session_state.pop("text_current_frame_key", None)
        st.rerun()

    st.session_state["text_current_video_id"] = video_id
    video_rows = grouped_rows[video_id]
    index = next_frame_index(video_rows, annotations)
    if index is None:
        st.session_state.pop("text_current_video_id", None)
        st.session_state.pop("text_current_frame_key", None)
        st.rerun()

    row = video_rows[index]
    key = frame_key(row["video_id"], row["frame_id"])
    st.session_state["text_current_frame_key"] = key

    previous_row, previous_annotation = previous_annotation_for_video(video_rows, annotations, index)
    subtitle_key = f"text_subtitle::{key}"
    static_key = f"text_static::{key}"
    other_key = f"text_other::{key}"
    active_key = st.session_state.get("active_text_frame_key")
    if active_key != key:
        st.session_state["active_text_frame_key"] = key
        existing = annotations.get(key, {})
        st.session_state[subtitle_key] = existing.get("subtitle_text", row.get("ocr_text", ""))
        st.session_state[static_key] = existing.get("static_text", "")
        st.session_state[other_key] = existing.get("other_text", "")

    video_done = sum(1 for item in video_rows if frame_key(item["video_id"], item["frame_id"]) in annotations)
    st.caption(
        f"Video {video_id} | Frame {index + 1}/{len(video_rows)} | "
        f"Annotated in video: {video_done}/{len(video_rows)}"
    )

    left_col, form_col, prev_col = st.columns([1.3, 1.0, 0.9], gap="large")
    with left_col:
        render_frame(row)

    with form_col:
        st.markdown(f"<div class='ocr-box'>{row.get('ocr_text') or ' '}</div>", unsafe_allow_html=True)
        with st.form(f"text_frame_form::{key}"):
            subtitle_text = st.text_area("Субтитри", key=subtitle_key, height=150)
            static_text = st.text_area("Статичний текст", key=static_key, height=110)
            other_text = st.text_area("Інше", key=other_key, height=90)
            save_col, empty_col = st.columns(2)
            save_clicked = save_col.form_submit_button("Зберегти", type="primary", use_container_width=True)
            empty_clicked = empty_col.form_submit_button("Порожній кадр", use_container_width=True)

        if save_clicked:
            save_annotation(
                decision_store,
                row,
                subtitle_text=subtitle_text,
                static_text=static_text,
                other_text=other_text,
                status="accepted",
            )
            st.session_state.pop("text_current_frame_key", None)
            st.rerun()
        if empty_clicked:
            save_annotation(
                decision_store,
                row,
                subtitle_text="",
                static_text="",
                other_text="",
                status="empty",
            )
            st.session_state.pop("text_current_frame_key", None)
            st.rerun()

    with prev_col:
        st.subheader("Попередній кадр")
        if previous_row and previous_annotation:
            render_frame(previous_row, compact=True)
            for label, field, target_key in [
                ("Субтитри", "subtitle_text", subtitle_key),
                ("Статичний текст", "static_text", static_key),
                ("Інше", "other_text", other_key),
            ]:
                value = clean_text(previous_annotation.get(field))
                st.caption(label)
                st.code(value or " ", language=None)
                st.button(
                    f"Скопіювати {label.lower()}",
                    key=f"copy::{field}::{key}",
                    disabled=not value,
                    on_click=set_textarea_value,
                    args=(target_key, value),
                    use_container_width=True,
                )
            copy_all_values = {
                subtitle_key: clean_text(previous_annotation.get("subtitle_text")),
                static_key: clean_text(previous_annotation.get("static_text")),
                other_key: clean_text(previous_annotation.get("other_text")),
            }
            st.button(
                "Скопіювати все",
                key=f"copy_all::{key}",
                on_click=set_textarea_values,
                args=(copy_all_values,),
                use_container_width=True,
            )
        else:
            st.caption("Немає попередньої анотації у цьому відео.")


if __name__ == "__main__":
    main()
