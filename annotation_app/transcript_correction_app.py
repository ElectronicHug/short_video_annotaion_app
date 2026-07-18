from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from annotation_app.common.auth import current_annotator_id, require_login
from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore


TARGET_FUNNEL_CATEGORIES = {"matched", "title_matched"}
HISTORY_LIMIT = 5
DEFAULT_VIDEO_WIDTH = 520


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def video_url(store: HfDatasetStore, video: dict[str, Any]) -> str:
    gcs_url = video.get("video_gcs_url")
    if isinstance(gcs_url, str) and gcs_url:
        return gcs_url
    return store.video_url(video)


def frame_key(video_id: str, frame_id: str) -> str:
    return f"{video_id}/{frame_id}"


@st.cache_data(show_spinner=False, ttl=120)
def load_transcript_rows() -> dict[str, Any]:
    store = HfDatasetStore.from_config(root=Path(__file__).resolve().parents[1])
    manifest = {str(row.get("video_id")): row for row in store.load_manifest()}
    funnel_state = store.load_funnel_state({"decisions": {}})
    selected_videos = {
        str(video_id)
        for video_id, decision in (funnel_state.get("decisions") or {}).items()
        if isinstance(decision, dict) and decision.get("category") in TARGET_FUNNEL_CATEGORIES
    }
    frames = [row for row in store.load_frames_manifest() if str(row.get("video_id")) in selected_videos]
    ocr_by_key = {
        frame_key(str(row.get("video_id")), str(row.get("frame_id"))): row
        for row in store.load_qwen_frame_ocr()
    }
    text_by_key = {
        frame_key(str(row.get("video_id")), str(row.get("frame_id"))): row
        for row in store.load_text_frame_corrections()
    }

    manual_drafts_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in store.load_manual_subtitle_transcript_drafts():
        manual_drafts_by_video.setdefault(str(row.get("video_id")), []).append(row)

    ocr_llm_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in store.load_ocr_llm_transcripts():
        ocr_llm_by_video.setdefault(str(row.get("video_id")), []).append(row)

    corrected_by_video = {
        str(row.get("video_id")): row
        for row in store.load_corrected_transcripts()
    }
    return {
        "manifest": manifest,
        "selected_videos": sorted(selected_videos),
        "frames": frames,
        "ocr_by_key": ocr_by_key,
        "text_by_key": text_by_key,
        "manual_drafts_by_video": manual_drafts_by_video,
        "ocr_llm_by_video": ocr_llm_by_video,
        "corrected_by_video": corrected_by_video,
    }


def rows_for_video(data: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    rows = []
    for frame in data["frames"]:
        if str(frame.get("video_id")) != video_id:
            continue
        key = frame_key(str(frame.get("video_id")), str(frame.get("frame_id")))
        ocr = data["ocr_by_key"].get(key, {})
        manual = data["text_by_key"].get(key, {})
        rows.append(
            {
                "time": frame.get("timestamp_seconds"),
                "frame_id": frame.get("frame_id"),
                "ocr_text": clean_text(ocr.get("prediction_text") or ocr.get("raw_text")),
                "manual_subtitle": clean_text(manual.get("subtitle_text")),
            }
        )
    return sorted(rows, key=lambda row: (float(row.get("time") or 0), str(row.get("frame_id") or "")))


def candidate_map(video_id: str, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    corrected = data["corrected_by_video"].get(video_id)
    if corrected:
        candidates["ground_truth_if_exists"] = corrected

    for row in data["manual_drafts_by_video"].get(video_id, []):
        endpoint = str(row.get("endpoint") or "manual_draft")
        candidates[endpoint] = row

    for row in data["ocr_llm_by_video"].get(video_id, []):
        endpoint = f"ocr_{row.get('endpoint') or 'llm'}"
        candidates.setdefault(endpoint, row)
    return candidates


def preferred_candidate(candidates: dict[str, dict[str, Any]]) -> tuple[str, str]:
    for label in ("ground_truth_if_exists", "openai", "local_gemma4"):
        row = candidates.get(label)
        if row and clean_text(row.get("transcript_text")):
            return label, clean_text(row.get("transcript_text"))
    for label, row in candidates.items():
        text = clean_text(row.get("transcript_text"))
        if text:
            return label, text
    return "", ""


def choose_next_video(video_ids: list[str], annotations: dict[str, dict[str, Any]]) -> str | None:
    current = st.session_state.get("transcript_current_video_id")
    if current in video_ids:
        return str(current)
    skipped = set(st.session_state.get("transcript_skipped_video_ids", []))
    for video_id in video_ids:
        if video_id not in annotations and video_id not in skipped:
            return video_id
    for video_id in video_ids:
        if video_id not in annotations:
            return video_id
    return None


def remember_video(video_id: str) -> None:
    history = list(st.session_state.get("transcript_history", []))
    if not history or history[-1] != video_id:
        history.append(video_id)
    st.session_state["transcript_history"] = history[-HISTORY_LIMIT:]


def go_back() -> bool:
    history = list(st.session_state.get("transcript_history", []))
    if not history:
        return False
    previous = history.pop()
    st.session_state["transcript_history"] = history
    st.session_state["transcript_current_video_id"] = previous
    st.session_state.pop("active_transcript_video_id", None)
    return True


def reset_current() -> None:
    st.session_state.pop("transcript_current_video_id", None)
    st.session_state.pop("active_transcript_video_id", None)


def save_annotation(
    store: FirestoreDecisionStore,
    video_id: str,
    transcript_text: str,
    status: str,
    base_candidate_label: str,
) -> None:
    annotation = {
        "dataset_id": DATASET_ID,
        "task": "video_transcript_correction",
        "video_id": video_id,
        "transcript_text": clean_text(transcript_text),
        "status": status,
        "base_candidate_label": base_candidate_label,
        "annotator_id": current_annotator_id(),
        "annotated_at": now_iso(),
    }
    store.upsert_video_transcript_annotation(
        dataset_id=DATASET_ID,
        video_id=video_id,
        annotation=annotation,
        annotator_id=current_annotator_id(),
    )


def show_candidate(label: str, row: dict[str, Any] | None) -> None:
    text = clean_text((row or {}).get("transcript_text"))
    confidence = clean_text((row or {}).get("confidence"))
    st.caption(f"{label}" + (f" / {confidence}" if confidence else ""))
    st.text_area(
        label=f"candidate_{label}",
        value=text or "Нема кандидата",
        height=240,
        disabled=True,
        label_visibility="collapsed",
    )


def render_page_css() -> None:
    st.markdown(
        """
        <style>
        .stApp h1 {
            font-size: 2.15rem;
            margin-bottom: 0.65rem;
        }
        .stApp h2 {
            font-size: 1.65rem;
        }
        .stApp h3 {
            font-size: 1.3rem;
        }
        .stButton > button {
            min-height: 3.1rem;
            font-size: 1rem;
            white-space: normal;
        }
        textarea {
            font-size: 1.02rem !important;
            line-height: 1.45 !important;
        }
        div[data-testid="stDataFrame"] {
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    active_user = require_login(form_key="transcript_login_form")
    if active_user is None:
        st.stop()

    render_page_css()
    st.header("Корекція фінального тексту відео")
    st.caption("Один фінальний текст на відео. За основу беремо OpenAI-кандидат, якщо він є.")

    hf_store = HfDatasetStore.from_config(root=Path(__file__).resolve().parents[1])
    firestore_store = FirestoreDecisionStore.from_config()
    data = load_transcript_rows()
    annotations = firestore_store.load_video_transcript_annotations(DATASET_ID)

    with st.sidebar:
        st.metric("Відео для перевірки", len(data["selected_videos"]))
        st.metric("Вже розмічено", len(annotations))
        st.metric("Історія назад", len(st.session_state.get("transcript_history", [])))
        video_width = st.slider("Ширина відео", min_value=420, max_value=820, value=DEFAULT_VIDEO_WIDTH, step=20)
        if st.button("Оновити дані", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    video_id = choose_next_video(data["selected_videos"], annotations)
    if not video_id:
        st.success("Усі доступні відео вже мають фінальну transcript-розмітку.")
        return
    st.session_state["transcript_current_video_id"] = video_id
    video = data["manifest"].get(video_id)
    if not video:
        st.error(f"Не знайдено відео в manifest: {video_id}")
        return

    candidates = candidate_map(video_id, data)
    base_label, base_text = preferred_candidate(candidates)
    text_key = f"transcript_text_{video_id}"
    if st.session_state.get("active_transcript_video_id") != video_id:
        st.session_state["active_transcript_video_id"] = video_id
        st.session_state[text_key] = base_text

    left, right = st.columns([0.95, 1.45], gap="large")
    with left:
        st.video(video_url(hf_store, video), width=video_width)
        st.caption(f"{video_id} | база: {base_label or 'нема кандидата'}")

    with right:
        st.text_area("Фінальний текст", key=text_key, height=360)

        st.subheader("Кандидати")
        show_candidate("openai", candidates.get("openai"))
        show_candidate("local_gemma4", candidates.get("local_gemma4"))
        if candidates.get("ground_truth_if_exists"):
            show_candidate("ground_truth_if_exists", candidates.get("ground_truth_if_exists"))

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Зберегти", type="primary", use_container_width=True):
            remember_video(video_id)
            save_annotation(firestore_store, video_id, st.session_state[text_key], "accepted", base_label)
            reset_current()
            st.rerun()
        if b2.button("Нема мовлення", use_container_width=True):
            remember_video(video_id)
            save_annotation(firestore_store, video_id, "", "empty", base_label)
            reset_current()
            st.rerun()
        if b3.button("Проблема", use_container_width=True):
            remember_video(video_id)
            save_annotation(firestore_store, video_id, st.session_state[text_key], "problem", base_label)
            reset_current()
            st.rerun()
        if b4.button("Назад", use_container_width=True):
            if go_back():
                st.rerun()
            st.info("Немає попередніх відео в історії.")

        if st.button("Наступне відео", use_container_width=True):
            remember_video(video_id)
            skipped = set(st.session_state.get("transcript_skipped_video_ids", []))
            skipped.add(video_id)
            st.session_state["transcript_skipped_video_ids"] = sorted(skipped)
            reset_current()
            st.rerun()

    with st.expander("Debug: кадри, OCR і ручні субтитри", expanded=False):
        debug_df = pd.DataFrame(rows_for_video(data, video_id))
        if not debug_df.empty:
            st.dataframe(
                debug_df,
                hide_index=True,
                use_container_width=True,
                height=min(760, 64 + 54 * len(debug_df)),
                column_config={
                    "time": st.column_config.NumberColumn("time", width="small", format="%.3f"),
                    "frame_id": st.column_config.TextColumn("frame_id", width="medium"),
                    "ocr_text": st.column_config.TextColumn("ocr_text", width="large"),
                    "manual_subtitle": st.column_config.TextColumn("manual subtitle", width="large"),
                },
            )
        else:
            st.info("Для цього відео немає debug-рядків.")


if __name__ == "__main__":
    main()
