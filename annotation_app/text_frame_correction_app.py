from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from PIL import Image

from annotation_app.common.auth import current_annotator_id, logout, require_login
from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore
from annotation_app.common.hf_tokens import get_config_value


ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
TARGET_FUNNEL_CATEGORIES = {"matched", "title_matched"}
DEFAULT_CLAIM_TTL_MINUTES = 60
FRAME_CACHE_DIR = ROOT / ".cache" / "text_frames"
FRAME_PREVIEW_CACHE_DIR = ROOT / ".cache" / "text_frame_previews"
TEXT_ANNOTATIONS_CACHE_SECONDS = 60
TEXT_CLAIMS_CACHE_SECONDS = 15
FRAME_PRELOAD_AHEAD = 20
FRAME_PRELOAD_WORKERS = 6
DEFAULT_FRAME_RENDER_WIDTH = 600


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


def local_frame_path(row: dict[str, Any]) -> Path:
    frame_path = Path(str(row["frame_path"]))
    return FRAME_CACHE_DIR / frame_path


def download_frame(row: dict[str, Any]) -> Path:
    local_path = local_frame_path(row)
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


def local_preview_path(row: dict[str, Any], width: int) -> Path:
    frame_path = Path(str(row["frame_path"]))
    return FRAME_PREVIEW_CACHE_DIR / f"w{width}" / frame_path.with_suffix(".jpg")


def preview_frame_path(row: dict[str, Any], width: int) -> Path:
    preview_path = local_preview_path(row, width)
    if preview_path.exists():
        return preview_path

    source_path = download_frame(row)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        if image.width > width:
            height = max(1, round(image.height * (width / image.width)))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        image.save(preview_path, format="JPEG", quality=82, optimize=True)
    return preview_path


def preload_frame_batch(video_rows: list[dict[str, Any]], start_index: int, *, limit: int = FRAME_PRELOAD_AHEAD) -> None:
    batch = video_rows[start_index : start_index + limit]
    missing = [row for row in batch if not local_frame_path(row).exists()]
    if not missing:
        return

    with st.spinner(f"Loading frames... {len(missing)}"):
        with ThreadPoolExecutor(max_workers=FRAME_PRELOAD_WORKERS) as executor:
            futures = {executor.submit(download_frame, row): row for row in missing}
            failed = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    row = futures[future]
                    failed.append(f"{row.get('frame_id')}: {type(exc).__name__}")
            if failed:
                st.warning(f"Some frames were not preloaded: {', '.join(failed[:3])}")


def render_frame(row: dict[str, Any], *, width: int) -> None:
    image_path = preview_frame_path(row, width)
    st.caption(f"{row['frame_id']} | {row.get('timestamp_seconds')}s")
    st.image(str(image_path), width=width)


def render_previous_text_tools(
    previous_annotation: dict[str, Any] | None,
    *,
    subtitle_key: str,
    static_key: str,
    other_key: str,
    frame_key_value: str,
) -> None:
    if not previous_annotation:
        st.caption("У цьому відео ще немає попередньої анотації.")
        return

    previous_values = {
        "Субтитри": (subtitle_key, clean_text(previous_annotation.get("subtitle_text"))),
        "Статичний текст": (static_key, clean_text(previous_annotation.get("static_text"))),
        "Інше": (other_key, clean_text(previous_annotation.get("other_text"))),
    }
    for label, (target_key, value) in previous_values.items():
        if not value:
            continue
        st.caption(label)
        st.text(value)
        st.button(
            f"Скопіювати {label.lower()}",
            key=f"copy::{target_key}::{frame_key_value}",
            on_click=set_textarea_value,
            args=(target_key, value),
            use_container_width=True,
        )

    copy_all_values = {target_key: value for target_key, value in previous_values.values()}
    st.button(
        "Скопіювати все з попереднього",
        key=f"copy_all::{frame_key_value}",
        on_click=set_textarea_values,
        args=(copy_all_values,),
        use_container_width=True,
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


def remove_static_text_from_ocr(ocr_text: str, static_text: str) -> str:
    ocr_text = clean_text(ocr_text)
    static_text = clean_text(static_text)
    if not ocr_text or not static_text:
        return ocr_text

    static_lines = {" ".join(line.split()).casefold() for line in static_text.splitlines() if line.strip()}
    kept_lines = []
    removed_any = False
    for line in ocr_text.splitlines():
        normalized_line = " ".join(line.split()).casefold()
        if normalized_line and normalized_line in static_lines:
            removed_any = True
            continue
        kept_lines.append(line)
    if removed_any:
        return "\n".join(line for line in kept_lines if line.strip()).strip()

    compact_ocr = " ".join(ocr_text.split())
    compact_static = " ".join(static_text.split())
    index = compact_ocr.casefold().find(compact_static.casefold())
    if index < 0:
        return ocr_text
    cleaned = (compact_ocr[:index] + compact_ocr[index + len(compact_static) :]).strip()
    return " ".join(cleaned.split())


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


def load_text_claims_cached(store: FirestoreDecisionStore) -> dict[str, dict[str, Any]]:
    now_ts = datetime.now(timezone.utc).timestamp()
    cache = st.session_state.get("text_claims_cache")
    cache_loaded_at = float(st.session_state.get("text_claims_cache_loaded_at") or 0)
    if isinstance(cache, dict) and now_ts - cache_loaded_at < TEXT_CLAIMS_CACHE_SECONDS:
        return cache
    claims = store.load_active_text_video_claims(DATASET_ID)
    st.session_state["text_claims_cache"] = claims
    st.session_state["text_claims_cache_loaded_at"] = now_ts
    return claims


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
    active_claims = load_text_claims_cached(decision_store)
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
        frame_render_width = st.slider(
            "Ширина кадру",
            min_value=360,
            max_value=900,
            value=DEFAULT_FRAME_RENDER_WIDTH,
            step=40,
        )
        show_previous_tools = st.toggle("Показати текст з попереднього кадру", value=False)
        if st.button("Reload HF/OCR data", use_container_width=True):
            load_text_rows.clear()
            st.session_state.pop("text_annotations_cache", None)
            st.session_state.pop("text_annotations_cache_loaded_at", None)
            st.session_state.pop("text_claims_cache", None)
            st.session_state.pop("text_claims_cache_loaded_at", None)
            st.rerun()
        if st.button("Choose next video", use_container_width=True):
            st.session_state.pop("text_current_video_id", None)
            st.session_state.pop("text_current_frame_key", None)
            st.rerun()

    total = len(rows)
    done = len(annotations)
    if total:
        st.caption(f"Annotated {done}/{total} frames | Remaining {max(0, total - done)}")
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

    preload_frame_batch(video_rows, index)

    row = video_rows[index]
    key = frame_key(row["video_id"], row["frame_id"])
    st.session_state["text_current_frame_key"] = key

    _, previous_annotation = previous_annotation_for_video(video_rows, annotations, index)
    subtitle_key = f"text_subtitle::{key}"
    static_key = f"text_static::{key}"
    other_key = f"text_other::{key}"
    active_key = st.session_state.get("active_text_frame_key")
    if active_key != key:
        st.session_state["active_text_frame_key"] = key
        existing = annotations.get(key, {})
        previous_static_text = clean_text((previous_annotation or {}).get("static_text"))
        default_static_text = existing.get("static_text", previous_static_text)
        default_subtitle_text = existing.get(
            "subtitle_text",
            remove_static_text_from_ocr(row.get("ocr_text", ""), previous_static_text),
        )
        st.session_state[subtitle_key] = default_subtitle_text
        st.session_state[static_key] = default_static_text
        st.session_state[other_key] = existing.get("other_text", "")

    video_done = sum(1 for item in video_rows if frame_key(item["video_id"], item["frame_id"]) in annotations)
    st.caption(
        f"Video {video_id} | Frame {index + 1}/{len(video_rows)} | "
        f"Annotated in video: {video_done}/{len(video_rows)}"
    )

    left_col, form_col = st.columns([1.35, 1.0], gap="large")
    with left_col:
        render_frame(row, width=frame_render_width)

    with form_col:
        st.caption("OCR")
        st.text(row.get("ocr_text") or " ")
        if previous_annotation and clean_text(previous_annotation.get("static_text")):
            st.caption("Субтитри заповнені як OCR мінус статичний текст з попереднього кадру.")
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

        if show_previous_tools:
            with st.expander("Текст з попереднього кадру", expanded=False):
                render_previous_text_tools(
                    previous_annotation,
                    subtitle_key=subtitle_key,
                    static_key=static_key,
                    other_key=other_key,
                    frame_key_value=key,
                )

if __name__ == "__main__":
    main()
