from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
DATASET_ID = "manual_seed_v2"
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "datasets" / DATASET_ID))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT / "results"))
PSEUDO_LABELS_PATH = (
    RESULTS_DIR
    / "ocr_predictions"
    / DATASET_ID
    / "chatgpt_corrector_final_df.jsonl"
)
ANNOTATION_DIR = DATASET_DIR / "03_text_annotation" / "annotations"
STATE_PATH = ANNOTATION_DIR / "state.json"
EXPORT_PATH = ANNOTATION_DIR / "export.jsonl"
HISTORY_LIMIT = 2


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def frame_key(video_id: str, frame_id: str) -> str:
    return f"{video_id}/{frame_id}"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    if text.lower() == "none":
        return ""
    return text.strip()


def image_data_uri(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.is_absolute():
        image_path = ROOT / image_path
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_pseudo_labels() -> list[dict[str, Any]]:
    if not PSEUDO_LABELS_PATH.exists():
        raise FileNotFoundError(f"Pseudo-label file not found: {PSEUDO_LABELS_PATH}")

    df = pd.read_json(PSEUDO_LABELS_PATH, lines=True)
    required = ["video_id", "frame_id", "frame_path"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Pseudo-label file is missing columns: {missing}")

    df = df.sort_values(["video_id", "timestamp_seconds", "frame_id"], na_position="last")
    rows = []
    seen: set[str] = set()
    for row in df.to_dict("records"):
        key = frame_key(str(row["video_id"]), str(row["frame_id"]))
        if key in seen:
            continue
        seen.add(key)
        frame_path = Path(str(row["frame_path"]))
        abs_frame_path = frame_path if frame_path.is_absolute() else ROOT / frame_path
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "video_id": str(row["video_id"]),
                "frame_id": str(row["frame_id"]),
                "timestamp_seconds": row.get("timestamp_seconds"),
                "source_category": row.get("source_category"),
                "frame_path": str(row["frame_path"]),
                "abs_frame_path": str(abs_frame_path),
                "pseudo_subtitles": clean_text(row.get("corrected_text")),
                "pseudo_confidence": clean_text(row.get("confidence")),
                "easyocr": clean_text(row.get("easyocr")),
                "florence_hf": clean_text(row.get("florence_hf")),
                "qwen2_vl_frame_ocr": clean_text(row.get("qwen2_vl_frame_ocr")),
            }
        )
    return rows


def default_state() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "stage": "03_text_annotation",
        "source_predictions": str(PSEUDO_LABELS_PATH.relative_to(ROOT)),
        "updated_at": None,
        "current_key": None,
        "annotations": {},
        "recent_history": [],
    }


def load_state() -> dict[str, Any]:
    state = read_json(STATE_PATH, default_state())
    if not isinstance(state, dict):
        return default_state()
    state.setdefault("dataset_id", DATASET_ID)
    state.setdefault("stage", "03_text_annotation")
    state.setdefault("source_predictions", str(PSEUDO_LABELS_PATH.relative_to(ROOT)))
    state.setdefault("updated_at", None)
    state.setdefault("current_key", None)
    state.setdefault("annotations", {})
    state.setdefault("recent_history", [])
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(STATE_PATH, state)


def build_export_rows(rows: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = state.get("annotations", {})
    export_rows = []
    for row in rows:
        key = frame_key(row["video_id"], row["frame_id"])
        annotation = annotations.get(key, {})
        export_rows.append(
            {
                "dataset_id": DATASET_ID,
                "stage": "03_text_annotation",
                "video_id": row["video_id"],
                "frame_id": row["frame_id"],
                "timestamp_seconds": row.get("timestamp_seconds"),
                "source_category": row.get("source_category"),
                "frame_path": row.get("frame_path"),
                "pseudo_subtitles": row.get("pseudo_subtitles", ""),
                "pseudo_confidence": row.get("pseudo_confidence", ""),
                "subtitle_text": annotation.get("subtitle_text", ""),
                "static_text": annotation.get("static_text", ""),
                "notes": annotation.get("notes", ""),
                "status": annotation.get("status"),
                "annotated": key in annotations,
                "annotated_at": annotation.get("annotated_at"),
            }
        )
    return export_rows


def write_export(rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    export_rows = build_export_rows(rows, state)
    write_text(EXPORT_PATH, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in export_rows))


def next_unfinished_index(
    rows: list[dict[str, Any]],
    annotations: dict[str, Any],
    current_index: int,
) -> int | None:
    for offset in range(1, len(rows) + 1):
        index = (current_index + offset) % len(rows)
        key = frame_key(rows[index]["video_id"], rows[index]["frame_id"])
        if key not in annotations:
            return index
    return None


def resolve_current_index(rows: list[dict[str, Any]], state: dict[str, Any]) -> int | None:
    annotations = state.get("annotations", {})
    current_key = state.get("current_key")
    if current_key:
        for index, row in enumerate(rows):
            if frame_key(row["video_id"], row["frame_id"]) == current_key:
                return index

    for index, row in enumerate(rows):
        key = frame_key(row["video_id"], row["frame_id"])
        if key not in annotations:
            return index
    return None


def previous_static_text(rows: list[dict[str, Any]], state: dict[str, Any], index: int) -> str:
    current_video_id = rows[index]["video_id"]
    annotations = state.get("annotations", {})
    for previous_index in range(index - 1, -1, -1):
        previous_row = rows[previous_index]
        if previous_row["video_id"] != current_video_id:
            break
        previous_key = frame_key(previous_row["video_id"], previous_row["frame_id"])
        static_text = clean_text(annotations.get(previous_key, {}).get("static_text"))
        if static_text:
            return static_text
    return ""


def save_annotation(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    index: int,
    *,
    subtitle_text: str,
    static_text: str,
    notes: str,
    status: str,
) -> None:
    row = rows[index]
    key = frame_key(row["video_id"], row["frame_id"])
    previous_annotation = state.get("annotations", {}).get(key)
    state.setdefault("annotations", {})[key] = {
        "video_id": row["video_id"],
        "frame_id": row["frame_id"],
        "frame_path": row.get("frame_path"),
        "timestamp_seconds": row.get("timestamp_seconds"),
        "source_category": row.get("source_category"),
        "pseudo_subtitles": row.get("pseudo_subtitles", ""),
        "pseudo_confidence": row.get("pseudo_confidence", ""),
        "subtitle_text": subtitle_text.strip(),
        "static_text": static_text.strip(),
        "notes": notes.strip(),
        "status": status,
        "annotated_at": now_iso(),
    }
    history = state.setdefault("recent_history", [])
    history.append({"key": key, "index": index, "previous_annotation": previous_annotation})
    state["recent_history"] = history[-HISTORY_LIMIT:]

    next_index = next_unfinished_index(rows, state.get("annotations", {}), index)
    state["current_key"] = None if next_index is None else frame_key(rows[next_index]["video_id"], rows[next_index]["frame_id"])
    save_state(state)
    write_export(rows, state)


def undo_last(rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    history = state.get("recent_history", [])
    if not history:
        return
    item = history.pop()
    key = item["key"]
    previous_annotation = item.get("previous_annotation")
    if previous_annotation:
        state.setdefault("annotations", {})[key] = previous_annotation
    else:
        state.setdefault("annotations", {}).pop(key, None)
    state["recent_history"] = history[-HISTORY_LIMIT:]
    state["current_key"] = key
    save_state(state)
    write_export(rows, state)


def render_metrics(rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    total = len(rows)
    done = len(state.get("annotations", {}))
    remaining = max(0, total - done)
    videos = len({row["video_id"] for row in rows})
    cols = st.columns(4)
    cols[0].metric("Annotated", done)
    cols[1].metric("Remaining", remaining)
    cols[2].metric("Frames", total)
    cols[3].metric("Videos", videos)
    if total:
        st.progress(done / total)


def render_frame(row: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="frame-card">
          <div class="frame-label">
            {row['video_id']} | {row['frame_id']} | {row.get('timestamp_seconds')}s | {row.get('source_category') or '-'}
          </div>
          <img src="{image_data_uri(row['frame_path'])}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_subtitle_text(widget_key: str, text: str) -> None:
    st.session_state[widget_key] = text


def render_ocr_candidate(label: str, text: str, subtitle_key: str, button_key: str) -> None:
    label_col, button_col = st.columns([4, 1])
    label_col.caption(label)
    button_col.button(
        "Insert",
        key=button_key,
        on_click=set_subtitle_text,
        args=(subtitle_key, text),
        use_container_width=True,
    )
    st.code(text, language=None)


def main() -> None:
    st.set_page_config(page_title="Text Annotation", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1500px;
        }
        .frame-card {
            background: #f8fafc;
            border: 1px solid #d7dde5;
            overflow: hidden;
            width: 100%;
        }
        .frame-card img {
            display: block;
            height: calc(100vh - 250px);
            min-height: 430px;
            object-fit: contain;
            width: 100%;
        }
        .frame-label {
            color: #475569;
            font-size: 0.9rem;
            padding: 0.45rem 0.6rem;
        }
        .pseudo-box {
            background: #f8fafc;
            border: 1px solid #d7dde5;
            color: #111827;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
            padding: 0.65rem 0.75rem;
            white-space: pre-wrap;
        }
        div.stButton > button {
            min-height: 2.8rem;
            font-weight: 650;
        }
        div[data-testid="stHorizontalBlock"] div.stButton > button {
            min-height: 2rem;
            padding-bottom: 0.15rem;
            padding-top: 0.15rem;
        }
        textarea {
            font-size: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Text Annotation")

    try:
        rows = load_pseudo_labels()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.info("Run the ChatGPT corrector and save chatgpt_corrector_final_df.jsonl first.")
        return

    if not rows:
        st.error("No pseudo-labeled frames found.")
        return

    state = load_state()
    render_metrics(rows, state)

    with st.sidebar:
        st.caption(f"Source: {PSEUDO_LABELS_PATH.relative_to(ROOT)}")
        st.caption(f"State: {STATE_PATH.relative_to(ROOT)}")
        st.caption(f"Export: {EXPORT_PATH.relative_to(ROOT)}")
        if st.button("Reload pseudo labels", use_container_width=True):
            load_pseudo_labels.clear()
            st.rerun()
        if st.button("Back", use_container_width=True, disabled=not state.get("recent_history")):
            undo_last(rows, state)
            st.rerun()
        if st.button("Export JSONL", use_container_width=True):
            write_export(rows, state)
            st.success("Exported")

        st.divider()
        status_counts: dict[str, int] = {}
        for annotation in state.get("annotations", {}).values():
            status = annotation.get("status", "accepted")
            status_counts[status] = status_counts.get(status, 0) + 1
        st.subheader("Status")
        for status in ["accepted", "empty_subtitles", "skipped"]:
            st.caption(f"{status}: {status_counts.get(status, 0)}")

    index = resolve_current_index(rows, state)
    if index is None:
        st.success("All unique frames are annotated.")
        return

    row = rows[index]
    key = frame_key(row["video_id"], row["frame_id"])
    state["current_key"] = key
    save_state(state)

    existing = state.get("annotations", {}).get(key, {})
    pseudo_subtitles = row.get("pseudo_subtitles", "")
    default_subtitles = existing.get("subtitle_text", pseudo_subtitles)
    default_static = existing.get("static_text", "")
    default_notes = existing.get("notes", "")
    subtitle_key = f"subtitle_text::{key}"
    static_key = f"static_text::{key}"
    notes_key = f"notes::{key}"

    if st.session_state.get("active_text_annotation_key") != key:
        st.session_state["active_text_annotation_key"] = key
        st.session_state[subtitle_key] = default_subtitles
        st.session_state[static_key] = default_static
        st.session_state[notes_key] = default_notes

    image_col, form_col = st.columns([1.45, 1], gap="large")
    with image_col:
        render_frame(row)

    with form_col:
        confidence = row.get("pseudo_confidence") or "-"
        st.caption(f"Pseudo confidence: {confidence}")
        st.markdown(f"<div class='pseudo-box'>{pseudo_subtitles or ' '}</div>", unsafe_allow_html=True)

        previous_static = previous_static_text(rows, state, index)
        copy_previous = st.checkbox(
            "Copy previous static text",
            value=False,
            disabled=not previous_static or bool(default_static),
        )
        if copy_previous and previous_static and not clean_text(st.session_state.get(static_key)):
            st.session_state[static_key] = previous_static

        with st.form(key=f"text_form_{key}"):
            subtitle_text = st.text_area(
                "Subtitles",
                height=160,
                help="Dynamic speech subtitles. Pseudo-label is prefilled here.",
                key=subtitle_key,
            )
            static_text = st.text_area(
                "Static text",
                height=120,
                help="Titles, fixed overlays, watermarks, CTA, author text.",
                key=static_key,
            )
            notes = st.text_area("Notes", height=80, key=notes_key)

            save_col, empty_col, skip_col = st.columns(3)
            save_clicked = save_col.form_submit_button("Save", type="primary", use_container_width=True)
            empty_clicked = empty_col.form_submit_button("Empty", use_container_width=True)
            skip_clicked = skip_col.form_submit_button("Skip", use_container_width=True)

        st.markdown("**OCR candidates**")
        render_ocr_candidate(
            "Qwen2-VL frame OCR",
            row.get("qwen2_vl_frame_ocr", ""),
            subtitle_key,
            f"insert_qwen::{key}",
        )
        render_ocr_candidate(
            "EasyOCR",
            row.get("easyocr", ""),
            subtitle_key,
            f"insert_easyocr::{key}",
        )
        render_ocr_candidate(
            "Florence HF",
            row.get("florence_hf", ""),
            subtitle_key,
            f"insert_florence::{key}",
        )

        if save_clicked:
            save_annotation(
                rows,
                state,
                index,
                subtitle_text=subtitle_text,
                static_text=static_text,
                notes=notes,
                status="accepted",
            )
            st.rerun()
        if empty_clicked:
            save_annotation(
                rows,
                state,
                index,
                subtitle_text="",
                static_text=static_text,
                notes=notes,
                status="empty_subtitles",
            )
            st.rerun()
        if skip_clicked:
            save_annotation(
                rows,
                state,
                index,
                subtitle_text=subtitle_text,
                static_text=static_text,
                notes=notes,
                status="skipped",
            )
            st.rerun()

    updated_at = state.get("updated_at")
    if updated_at:
        st.caption(f"Last saved: {updated_at}")


if __name__ == "__main__":
    main()
