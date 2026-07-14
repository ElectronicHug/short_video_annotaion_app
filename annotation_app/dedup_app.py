from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "datasets" / "manual_seed_v2"))
CLASSIFICATION_STATE_PATH = DATASET_DIR / "annotations" / "state.json"
FRAMES_MANIFEST_PATH = DATASET_DIR / "01_frames" / "manifest.json"
DEDUP_DIR = DATASET_DIR / "02_deduplication"
STATE_PATH = DEDUP_DIR / "annotations" / "state.json"
EXPORT_PATH = DEDUP_DIR / "annotations" / "export.jsonl"
HISTORY_LIMIT = 2

CLASSIFICATION_CATEGORIES = [
    ("matched", "Matched"),
    ("title_matched", "Title + Matched"),
    ("partially_matched", "Partly Matched"),
    ("unmatched", "Unmatched"),
    ("ignore", "Ignore"),
]

DECISIONS = [
    {
        "id": "not_duplicate",
        "label": "Not Duplicate",
        "help": "New subtitle/text state worth keeping.",
    },
    {
        "id": "duplicate",
        "label": "Duplicate",
        "help": "Same visible subtitle/text as the previous frame.",
    },
    {
        "id": "unclear",
        "label": "Unclear",
        "help": "Hard to decide; keep for later review.",
    },
]
DECISION_BY_ID = {decision["id"]: decision for decision in DECISIONS}


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


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def pair_key(video_id: str, frame_id: str) -> str:
    return f"{video_id}/{frame_id}"


def image_data_uri(path: str | Path) -> str:
    image_path = ROOT / path if not Path(path).is_absolute() else Path(path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def load_frames_dataset() -> dict[str, Any]:
    manifest = read_json(FRAMES_MANIFEST_PATH)
    if manifest is None:
        raise FileNotFoundError(f"Frame manifest not found: {FRAMES_MANIFEST_PATH}")

    videos = []
    for video in manifest.get("videos", []):
        frames_manifest_path = ROOT / video["frames_manifest_path"]
        frames_manifest = read_json(frames_manifest_path)
        if not frames_manifest:
            continue
        frames = frames_manifest.get("frames", [])
        if len(frames) < 2:
            continue
        videos.append(
            {
                **video,
                "frames_manifest_path": relative_path(frames_manifest_path),
                "frames": frames,
                "frame_count": len(frames),
            }
        )
    return {**manifest, "videos": videos}


def default_state() -> dict[str, Any]:
    return {
        "dataset_id": "manual_seed_v2",
        "stage": "02_deduplication",
        "updated_at": None,
        "current_video_id": None,
        "current_pair_index": 1,
        "decisions": {},
        "recent_history": [],
    }


def load_state() -> dict[str, Any]:
    state = read_json(STATE_PATH, default_state())
    if not isinstance(state, dict):
        return default_state()
    state.setdefault("dataset_id", "manual_seed_v2")
    state.setdefault("stage", "02_deduplication")
    state.setdefault("current_video_id", None)
    state.setdefault("current_pair_index", 1)
    state.setdefault("decisions", {})
    state.setdefault("recent_history", [])
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(STATE_PATH, state)


def classification_counts() -> dict[str, int]:
    state = read_json(CLASSIFICATION_STATE_PATH, {})
    counts = {category_id: 0 for category_id, _ in CLASSIFICATION_CATEGORIES}
    for decision in state.get("decisions", {}).values():
        category = decision.get("category")
        if category in counts:
            counts[category] += 1
    return counts


def total_pair_count(videos: list[dict[str, Any]]) -> int:
    return sum(max(0, len(video["frames"]) - 1) for video in videos)


def video_pair_progress(video: dict[str, Any], decisions: dict[str, Any]) -> tuple[int, int]:
    total = max(0, len(video["frames"]) - 1)
    done = sum(
        pair_key(video["video_id"], frame["frame_id"]) in decisions
        for frame in video["frames"][1:]
    )
    return done, total


def next_unfinished_video_index(
    videos: list[dict[str, Any]],
    current_index: int,
    decisions: dict[str, Any],
) -> int | None:
    for offset in range(1, len(videos) + 1):
        index = (current_index + offset) % len(videos)
        done, total = video_pair_progress(videos[index], decisions)
        if done < total:
            return index
    return None


def first_unfinished_pair_index(video: dict[str, Any], decisions: dict[str, Any]) -> int:
    for index in range(1, len(video["frames"])):
        frame_id = video["frames"][index]["frame_id"]
        if pair_key(video["video_id"], frame_id) not in decisions:
            return index
    return max(1, len(video["frames"]) - 1)


def resolve_current_position(
    videos: list[dict[str, Any]],
    state: dict[str, Any],
) -> tuple[int, int] | None:
    decisions = state.get("decisions", {})
    current_id = state.get("current_video_id")
    if current_id:
        for index, video in enumerate(videos):
            if video["video_id"] == current_id:
                pair_index = int(state.get("current_pair_index", 1))
                pair_index = max(1, min(pair_index, len(video["frames"]) - 1))
                return index, pair_index

    for index, video in enumerate(videos):
        done, total = video_pair_progress(video, decisions)
        if done < total:
            return index, first_unfinished_pair_index(video, decisions)
    return None


def build_export_rows(dataset: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    decisions = state.get("decisions", {})
    for video in dataset["videos"]:
        frames = video["frames"]
        for index in range(1, len(frames)):
            frame = frames[index]
            previous_frame = frames[index - 1]
            key = pair_key(video["video_id"], frame["frame_id"])
            decision = decisions.get(key, {})
            rows.append(
                {
                    "dataset_id": "manual_seed_v2",
                    "stage": "02_deduplication",
                    "video_id": video["video_id"],
                    "source_category": video.get("source_category"),
                    "frame_id": frame["frame_id"],
                    "previous_frame_id": previous_frame["frame_id"],
                    "frame_path": frame["path"],
                    "previous_frame_path": previous_frame["path"],
                    "timestamp_seconds": frame.get("timestamp_seconds"),
                    "previous_timestamp_seconds": previous_frame.get("timestamp_seconds"),
                    "decision": decision.get("decision"),
                    "decision_label": DECISION_BY_ID.get(decision.get("decision"), {}).get("label"),
                    "annotated_at": decision.get("annotated_at"),
                    "annotated": key in decisions,
                }
            )
    return rows


def write_export(dataset: dict[str, Any], state: dict[str, Any]) -> None:
    rows = build_export_rows(dataset, state)
    write_text(EXPORT_PATH, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def save_decision(
    dataset: dict[str, Any],
    state: dict[str, Any],
    video: dict[str, Any],
    pair_index: int,
    decision_id: str,
) -> None:
    frame = video["frames"][pair_index]
    previous_frame = video["frames"][pair_index - 1]
    key = pair_key(video["video_id"], frame["frame_id"])
    previous_decision = state.get("decisions", {}).get(key)
    state.setdefault("decisions", {})[key] = {
        "decision": decision_id,
        "video_id": video["video_id"],
        "frame_id": frame["frame_id"],
        "previous_frame_id": previous_frame["frame_id"],
        "frame_path": frame["path"],
        "previous_frame_path": previous_frame["path"],
        "timestamp_seconds": frame.get("timestamp_seconds"),
        "previous_timestamp_seconds": previous_frame.get("timestamp_seconds"),
        "annotated_at": now_iso(),
    }
    history = state.setdefault("recent_history", [])
    history.append(
        {
            "video_id": video["video_id"],
            "pair_index": pair_index,
            "key": key,
            "previous_decision": previous_decision,
        }
    )
    state["recent_history"] = history[-HISTORY_LIMIT:]
    advance_after_save(dataset["videos"], state, video, pair_index)
    save_state(state)
    write_export(dataset, state)


def advance_after_save(
    videos: list[dict[str, Any]],
    state: dict[str, Any],
    video: dict[str, Any],
    pair_index: int,
) -> None:
    if pair_index < len(video["frames"]) - 1:
        state["current_video_id"] = video["video_id"]
        state["current_pair_index"] = pair_index + 1
        return

    current_index = next(
        index for index, candidate_video in enumerate(videos) if candidate_video["video_id"] == video["video_id"]
    )
    next_index = next_unfinished_video_index(videos, current_index, state.get("decisions", {}))
    if next_index is None:
        state["current_video_id"] = None
        state["current_pair_index"] = 1
    else:
        next_video = videos[next_index]
        state["current_video_id"] = next_video["video_id"]
        state["current_pair_index"] = first_unfinished_pair_index(next_video, state.get("decisions", {}))


def undo_last(dataset: dict[str, Any], state: dict[str, Any]) -> None:
    history = state.get("recent_history", [])
    if not history:
        return
    item = history.pop()
    key = item["key"]
    previous_decision = item.get("previous_decision")
    if previous_decision:
        state.setdefault("decisions", {})[key] = previous_decision
    else:
        state.setdefault("decisions", {}).pop(key, None)
    state["recent_history"] = history[-HISTORY_LIMIT:]
    state["current_video_id"] = item["video_id"]
    state["current_pair_index"] = item["pair_index"]
    save_state(state)
    write_export(dataset, state)


def render_frame(path: str, label: str, frame_id: str, timestamp: Any) -> None:
    st.markdown(
        f"""
        <div class="frame-card">
          <div class="frame-label">{label} | {frame_id} | {timestamp}s</div>
          <img src="{image_data_uri(path)}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(dataset: dict[str, Any], state: dict[str, Any]) -> None:
    total_pairs = total_pair_count(dataset["videos"])
    done = len(state.get("decisions", {}))
    remaining = max(0, total_pairs - done)
    cols = st.columns(4)
    cols[0].metric("Deduped", done)
    cols[1].metric("Remaining", remaining)
    cols[2].metric("Videos", len(dataset["videos"]))
    cols[3].metric("FPS", dataset.get("fps", "-"))
    if total_pairs:
        st.progress(done / total_pairs)


def main() -> None:
    st.set_page_config(page_title="Frame Deduplication", layout="wide")
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
            height: calc(100vh - 310px);
            min-height: 360px;
            object-fit: contain;
            width: 100%;
        }
        .frame-label {
            color: #475569;
            font-size: 0.9rem;
            padding: 0.45rem 0.6rem;
        }
        div.stButton > button {
            min-height: 3.4rem;
            font-weight: 650;
        }
        .decision-help {
            color: #64748b;
            font-size: 0.88rem;
            margin-top: -0.4rem;
            min-height: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Frame Deduplication")

    try:
        dataset = load_frames_dataset()
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Run notebooks/manual_seed_v2_extract_frames.ipynb first.")
        return

    state = load_state()
    videos = dataset["videos"]
    if not videos:
        st.error("No videos with at least two frames were found.")
        return

    with st.sidebar:
        st.caption(f"Frames: {FRAMES_MANIFEST_PATH}")
        st.caption(f"Output: {DEDUP_DIR}")
        if st.button("Reload frames", use_container_width=True):
            load_frames_dataset.clear()
            st.rerun()
        if st.button("Back", use_container_width=True, disabled=not state.get("recent_history")):
            undo_last(dataset, state)
            st.rerun()
        if st.button("Export JSONL", use_container_width=True):
            write_export(dataset, state)
            st.success("Exported")

        st.divider()
        st.subheader("Buckets")
        counts = classification_counts()
        for category_id, label in CLASSIFICATION_CATEGORIES:
            st.caption(f"{label}: {counts.get(category_id, 0)}")

        st.divider()
        st.subheader("Dedup")
        decision_counts = {decision["id"]: 0 for decision in DECISIONS}
        for decision in state.get("decisions", {}).values():
            decision_id = decision.get("decision")
            if decision_id in decision_counts:
                decision_counts[decision_id] += 1
        for decision in DECISIONS:
            st.caption(f"{decision['label']}: {decision_counts[decision['id']]}")

    render_metrics(dataset, state)

    position = resolve_current_position(videos, state)
    if position is None:
        st.success("All frame pairs are annotated.")
        return

    video_index, pair_index = position
    video = videos[video_index]
    state["current_video_id"] = video["video_id"]
    state["current_pair_index"] = pair_index
    save_state(state)

    done_for_video, total_for_video = video_pair_progress(video, state.get("decisions", {}))
    st.caption(
        f"{video['video_id']} | {video.get('source_category')} | "
        f"pair {pair_index}/{len(video['frames']) - 1} | video progress {done_for_video}/{total_for_video}"
    )

    previous_frame = video["frames"][pair_index - 1]
    current_frame = video["frames"][pair_index]

    frames_col, actions_col = st.columns([3.3, 1], gap="large")
    with frames_col:
        previous_col, current_col = st.columns(2)
        with previous_col:
            render_frame(
                previous_frame["path"],
                "Old",
                previous_frame["frame_id"],
                previous_frame.get("timestamp_seconds"),
            )
        with current_col:
            render_frame(
                current_frame["path"],
                "New",
                current_frame["frame_id"],
                current_frame.get("timestamp_seconds"),
            )

    with actions_col:
        st.subheader("Decision")
        for decision in DECISIONS:
            if st.button(decision["label"], use_container_width=True):
                save_decision(dataset, state, video, pair_index, decision["id"])
                st.rerun()
            st.markdown(
                f"<div class='decision-help'>{decision['help']}</div>",
                unsafe_allow_html=True,
            )

    updated_at = state.get("updated_at")
    if updated_at:
        st.caption(f"Last saved: {updated_at}")


if __name__ == "__main__":
    main()
