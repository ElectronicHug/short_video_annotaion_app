from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from annotation_app.common.auth import current_annotator_id, require_login
from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore


CATEGORIES = [
    ("no_usable_speech", "Без корисного мовлення"),
    ("speech_no_text", "Мовлення без тексту"),
    ("text_without_subtitles", "Текст без субтитрів"),
    ("insufficient_subtitle_alignment", "Недостатній збіг субтитрів"),
    ("partially_matched", "Субтитри частково збігаються"),
    ("title_matched", "Субтитри + додатковий текст"),
    ("matched", "Субтитри збігаються"),
    ("problem", "Проблема"),
]
CATEGORY_LABELS = dict(CATEGORIES)
TEXT_STATUSES = ["accepted", "empty", "problem", "skipped"]
TRANSCRIPT_STATUSES = ["accepted", "empty", "problem"]
ALLOWED_ROLES = {"owner", "reviewer", "local"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def frame_key(video_id: str, frame_id: str) -> str:
    return f"{video_id}/{frame_id}"


def video_url(store: HfDatasetStore, video: dict[str, Any]) -> str:
    gcs_url = video.get("video_gcs_url")
    if isinstance(gcs_url, str) and gcs_url:
        return gcs_url
    return store.video_url(video)


@st.cache_data(show_spinner=False, ttl=120)
def load_hf_repair_data() -> dict[str, Any]:
    store = HfDatasetStore.from_config(root=Path(__file__).resolve().parents[1])
    manifest = {str(row.get("video_id")): row for row in store.load_manifest()}
    frames_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in store.load_frames_manifest():
        frames_by_video.setdefault(str(row.get("video_id")), []).append(row)
    for video_id, rows in frames_by_video.items():
        frames_by_video[video_id] = sorted(
            rows,
            key=lambda row: (float(row.get("timestamp_seconds") or 0), str(row.get("frame_id") or "")),
        )
    ocr_by_key = {
        frame_key(str(row.get("video_id")), str(row.get("frame_id"))): row
        for row in store.load_qwen_frame_ocr()
    }
    return {
        "manifest": manifest,
        "frames_by_video": frames_by_video,
        "ocr_by_key": ocr_by_key,
    }


def annotation_history_entry(reason: str, previous: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixed_at": now_iso(),
        "fixed_by": current_annotator_id(),
        "reason": reason,
        "previous": {
            "subtitle_text": previous.get("subtitle_text", ""),
            "static_text": previous.get("static_text", ""),
            "other_text": previous.get("other_text", ""),
            "status": previous.get("status", ""),
        },
    }


def add_history(annotation: dict[str, Any], reason: str) -> dict[str, Any]:
    updated = dict(annotation)
    history = updated.get("manual_fix_history")
    if not isinstance(history, list):
        history = []
    history.append(annotation_history_entry(reason, annotation))
    updated["manual_fix_history"] = history
    updated["manual_fixed_at"] = now_iso()
    updated["manual_fix_reason"] = reason
    updated["fixed_by"] = current_annotator_id()
    return updated


def make_frame_editor_rows(
    video_id: str,
    frames: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    ocr_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        frame_id = str(frame.get("frame_id") or "")
        key = frame_key(video_id, frame_id)
        annotation = annotations.get(key, {})
        ocr = ocr_by_key.get(key, {})
        rows.append(
            {
                "frame_id": frame_id,
                "time": frame.get("timestamp_seconds"),
                "subtitle_text": clean_text(annotation.get("subtitle_text")),
                "static_text": clean_text(annotation.get("static_text")),
                "other_text": clean_text(annotation.get("other_text")),
                "status": clean_text(annotation.get("status")) or "accepted",
                "ocr_text": clean_text(ocr.get("prediction_text") or ocr.get("raw_text")),
            }
        )
    return rows


def save_funnel_fix(
    store: FirestoreDecisionStore,
    video_id: str,
    video: dict[str, Any],
    category: str,
    reason: str,
    existing_decision: dict[str, Any],
) -> None:
    timestamp = now_iso()
    decision = {
        **existing_decision,
        "dataset_id": DATASET_ID,
        "video_id": video_id,
        "category": category,
        "classified_at": timestamp,
        "duration_seconds": video.get("duration_seconds"),
        "title": video.get("title"),
        "uploader": video.get("uploader"),
        "webpage_url": video.get("webpage_url"),
        "video_path": video.get("video_path"),
        "video_gcs_url": video.get("video_gcs_url"),
        "info_path": video.get("info_path"),
        "manual_fixed_at": timestamp,
        "manual_fix_reason": reason,
        "fixed_by": current_annotator_id(),
    }
    store.upsert_funnel_decision(
        dataset_id=DATASET_ID,
        video_id=video_id,
        decision=decision,
        annotator_id=current_annotator_id(),
    )


def save_frame_fixes(
    store: FirestoreDecisionStore,
    video_id: str,
    edited_rows: pd.DataFrame,
    existing_annotations: dict[str, dict[str, Any]],
    reason: str,
) -> int:
    changed = 0
    for row in edited_rows.to_dict("records"):
        frame_id = str(row.get("frame_id") or "")
        if not frame_id:
            continue
        key = frame_key(video_id, frame_id)
        previous = existing_annotations.get(key, {})
        updated = add_history(previous, reason)
        updated.update(
            {
                "dataset_id": DATASET_ID,
                "video_id": video_id,
                "frame_id": frame_id,
                "subtitle_text": clean_text(row.get("subtitle_text")),
                "static_text": clean_text(row.get("static_text")),
                "other_text": clean_text(row.get("other_text")),
                "status": clean_text(row.get("status")) or "accepted",
                "annotated_at": now_iso(),
            }
        )
        comparable_previous = {
            "subtitle_text": clean_text(previous.get("subtitle_text")),
            "static_text": clean_text(previous.get("static_text")),
            "other_text": clean_text(previous.get("other_text")),
            "status": clean_text(previous.get("status")) or "accepted",
        }
        comparable_updated = {
            "subtitle_text": updated["subtitle_text"],
            "static_text": updated["static_text"],
            "other_text": updated["other_text"],
            "status": updated["status"],
        }
        if comparable_previous == comparable_updated:
            continue
        store.upsert_text_frame_annotation(
            dataset_id=DATASET_ID,
            video_id=video_id,
            frame_id=frame_id,
            annotation=updated,
            annotator_id=current_annotator_id(),
        )
        changed += 1
    return changed


def save_transcript_fix(
    store: FirestoreDecisionStore,
    video_id: str,
    transcript_text: str,
    status: str,
    reason: str,
    existing_annotation: dict[str, Any],
) -> None:
    annotation = {
        **existing_annotation,
        "dataset_id": DATASET_ID,
        "task": "video_transcript_correction",
        "video_id": video_id,
        "transcript_text": clean_text(transcript_text),
        "status": status,
        "annotator_id": current_annotator_id(),
        "annotated_at": now_iso(),
        "manual_fixed_at": now_iso(),
        "manual_fix_reason": reason,
        "fixed_by": current_annotator_id(),
    }
    store.upsert_video_transcript_annotation(
        dataset_id=DATASET_ID,
        video_id=video_id,
        annotation=annotation,
        annotator_id=current_annotator_id(),
    )


def main() -> None:
    active_user = require_login(form_key="problem_fixes_login_form")
    if active_user is None:
        st.stop()
    if str(active_user.get("role") or "") not in ALLOWED_ROLES:
        st.error("Ця сторінка доступна тільки owner/reviewer.")
        st.stop()

    st.header("Виправлення проблем")
    st.caption("Ручний repair tool. Нічого не змінюється без натискання окремої кнопки збереження.")

    hf_store = HfDatasetStore.from_config(root=Path(__file__).resolve().parents[1])
    firestore_store = FirestoreDecisionStore.from_config()
    hf_data = load_hf_repair_data()

    with st.sidebar:
        if st.button("Оновити HF/Firestore cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    video_id = clean_text(st.text_input("video_id", placeholder="005daf50-8ebb-4a48-bad5-7b93bcaeb74d"))
    if not video_id:
        st.info("Встав video_id, щоб завантажити поточний стан.")
        return

    video = hf_data["manifest"].get(video_id)
    if not video:
        st.error("Не знайшов video_id у manifest.")
        return

    decisions = firestore_store.load_funnel_decisions(DATASET_ID)
    text_annotations = firestore_store.load_text_frame_annotations(DATASET_ID)
    transcript_annotations = firestore_store.load_video_transcript_annotations(DATASET_ID)
    current_decision = decisions.get(video_id, {})
    current_transcript = transcript_annotations.get(video_id, {})
    frames = hf_data["frames_by_video"].get(video_id, [])

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.video(video_url(hf_store, video), width=520)
        st.caption(video_id)
    with right:
        st.write("Поточний стан")
        st.json(
            {
                "category": current_decision.get("category"),
                "text_frames": sum(1 for frame in frames if frame_key(video_id, str(frame.get("frame_id"))) in text_annotations),
                "transcript_status": current_transcript.get("status"),
            },
            expanded=False,
        )
        st.caption(clean_text(video.get("title")))

    reason = clean_text(st.text_input("Причина виправлення", value="manual problem fix"))

    st.subheader("1. Класифікація відео")
    current_category = clean_text(current_decision.get("category"))
    category_ids = [category_id for category_id, _ in CATEGORIES]
    category_index = category_ids.index(current_category) if current_category in category_ids else 0
    selected_category = st.selectbox(
        "Нова класифікація",
        options=category_ids,
        index=category_index,
        format_func=lambda category_id: f"{category_id} - {CATEGORY_LABELS.get(category_id, category_id)}",
    )
    if st.button("Зберегти класифікацію", type="primary"):
        save_funnel_fix(firestore_store, video_id, video, selected_category, reason, current_decision)
        st.success(f"Класифікацію оновлено: {selected_category}")
        st.cache_data.clear()

    st.subheader("2. Текст на кадрах")
    frame_rows = make_frame_editor_rows(video_id, frames, text_annotations, hf_data["ocr_by_key"])
    if frame_rows:
        edited_frames = st.data_editor(
            pd.DataFrame(frame_rows),
            hide_index=True,
            use_container_width=True,
            height=min(760, 64 + 48 * len(frame_rows)),
            disabled=["frame_id", "time", "ocr_text"],
            column_config={
                "time": st.column_config.NumberColumn("time", width="small", format="%.3f"),
                "frame_id": st.column_config.TextColumn("frame_id", width="medium"),
                "subtitle_text": st.column_config.TextColumn("subtitle_text", width="large"),
                "static_text": st.column_config.TextColumn("static_text", width="large"),
                "other_text": st.column_config.TextColumn("other_text", width="medium"),
                "status": st.column_config.SelectboxColumn("status", options=TEXT_STATUSES, width="small"),
                "ocr_text": st.column_config.TextColumn("ocr_text", width="large"),
            },
            key=f"frame_fix_editor_{video_id}",
        )
        if st.button("Зберегти текст на кадрах"):
            changed = save_frame_fixes(firestore_store, video_id, edited_frames, text_annotations, reason)
            st.success(f"Оновлено кадрів: {changed}")
            st.cache_data.clear()
    else:
        st.warning("Для цього відео немає кадрів у frames_manifest.")

    st.subheader("3. Фінальний текст")
    transcript_status = clean_text(current_transcript.get("status")) or "accepted"
    transcript_status_index = (
        TRANSCRIPT_STATUSES.index(transcript_status)
        if transcript_status in TRANSCRIPT_STATUSES
        else 0
    )
    edited_transcript = st.text_area(
        "transcript_text",
        value=clean_text(current_transcript.get("transcript_text")),
        height=260,
    )
    selected_transcript_status = st.selectbox(
        "Статус фінального тексту",
        options=TRANSCRIPT_STATUSES,
        index=transcript_status_index,
    )
    if st.button("Зберегти фінальний текст"):
        save_transcript_fix(
            firestore_store,
            video_id,
            edited_transcript,
            selected_transcript_status,
            reason,
            current_transcript,
        )
        st.success("Фінальний текст оновлено.")
        st.cache_data.clear()


if __name__ == "__main__":
    main()
