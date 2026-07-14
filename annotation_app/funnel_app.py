from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
    from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore
    from annotation_app.common.hf_tokens import get_config_value, get_storage_backend
except ModuleNotFoundError:
    from common.firestore_decision_store import FirestoreDecisionStore
    from common.hf_dataset_store import DATASET_ID, HfDatasetStore
    from common.hf_tokens import get_config_value, get_storage_backend


ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
RAW_DATASET_DIR = Path(os.getenv("RAW_DATASET_DIR", ROOT / "raw_dataset"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "datasets" / "manual_seed_v2"))
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
STATE_PATH = ANNOTATIONS_DIR / "state.json"
EXPORT_PATH = ANNOTATIONS_DIR / "export.jsonl"
BUCKETS_DIR = DATASET_DIR / "buckets"
MAX_DURATION_SECONDS = 60
HISTORY_LIMIT = 2
HF_PREFETCH_AHEAD = 2
HF_DOWNLOAD_TIMEOUT_SECONDS = 20

CATEGORIES = [
    {
        "id": "matched",
        "label": "Matched",
        "help": "Speech matches visible subtitles/text.",
    },
    {
        "id": "title_matched",
        "label": "Title matched",
        "help": "Title/static text matches; subtitles may be absent.",
    },
    {
        "id": "partially_matched",
        "label": "Partly matched",
        "help": "Only part of the visible text matches.",
    },
    {
        "id": "unmatched",
        "label": "Unmatched",
        "help": "Visible text does not match the speech/title.",
    },
    {
        "id": "annotation_problem",
        "label": "Annotation problem",
        "help": "Technical/load issue or impossible to annotate reliably.",
    },
    {
        "id": "ignore",
        "label": "Ignore",
        "help": "Not useful for the dataset.",
    },
]
CATEGORY_BY_ID = {category["id"]: category for category in CATEGORIES}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_info(path: Path) -> dict[str, Any]:
    return read_json(path, {}) or {}


def extract_json_string(text: str, field_name: str) -> str:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*("(?:\\.|[^"\\])*")', text)
    if not match:
        return ""
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ""
    return value if isinstance(value, str) else ""


def extract_json_number(text: str, field_name: str) -> float | int | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if not match:
        return None
    raw_value = match.group(1)
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return int(value) if value.is_integer() else value


def load_fast_info(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = extract_json_string(text, "title") or extract_json_string(text, "fulltitle")
    uploader = extract_json_string(text, "uploader") or extract_json_string(text, "channel")
    return {
        "duration": extract_json_number(text, "duration"),
        "title": title,
        "uploader": uploader,
        "webpage_url": extract_json_string(text, "webpage_url"),
        "width": extract_json_number(text, "width"),
        "height": extract_json_number(text, "height"),
        "filesize": extract_json_number(text, "filesize"),
    }


def find_mp4(video_dir: Path) -> Path | None:
    mp4s = sorted(video_dir.glob("*.mp4"))
    return mp4s[0] if mp4s else None


@st.cache_data(show_spinner=False)
def scan_raw_videos() -> list[dict[str, Any]]:
    videos = []
    if not RAW_DATASET_DIR.exists():
        return videos

    for video_dir in sorted(path for path in RAW_DATASET_DIR.iterdir() if path.is_dir()):
        expected_video_path = video_dir / f"{video_dir.name}.mp4"
        video_path = expected_video_path if expected_video_path.exists() else find_mp4(video_dir)
        info_path = video_dir / f"{video_dir.name}.info.json"
        if video_path is None or not info_path.exists():
            continue

        videos.append(
            {
                "video_id": video_dir.name,
                "video_path": str(video_path),
                "video_relpath": relative_path(video_path),
                "info_path": str(info_path),
                "info_relpath": relative_path(info_path),
            }
        )
    return videos


def enrich_video(video: dict[str, Any]) -> dict[str, Any] | None:
    info = load_fast_info(Path(video["info_path"]))
    duration = info.get("duration")
    if duration is None:
        return None
    try:
        duration_seconds = float(duration)
    except (TypeError, ValueError):
        return None
    if duration_seconds > MAX_DURATION_SECONDS:
        return None

    return {
        **video,
        "duration_seconds": duration_seconds,
        "title": info.get("title") or "",
        "uploader": info.get("uploader") or "",
        "webpage_url": info.get("webpage_url") or "",
        "width": info.get("width"),
        "height": info.get("height"),
        "filesize": info.get("filesize"),
    }


def default_state() -> dict[str, Any]:
    return {
        "dataset_id": "manual_seed_v2",
        "source_dataset": "raw_dataset",
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "updated_at": None,
        "current_video_id": None,
        "decisions": {},
        "recent_history": [],
    }


def default_hf_state() -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "task": "funnel",
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "updated_at": None,
        "current_video_id": None,
        "decisions": {},
        "recent_history": [],
    }


def get_decision_backend() -> str:
    backend = get_config_value("DECISION_BACKEND").lower()
    if backend:
        return backend
    if get_config_value("FIRESTORE_PROJECT_ID") or get_config_value("GCP_PROJECT_ID"):
        return "firestore"
    return "hf"


def load_state() -> dict[str, Any]:
    state = read_json(STATE_PATH, default_state())
    if not isinstance(state, dict):
        return default_state()
    state.setdefault("dataset_id", "manual_seed_v2")
    state.setdefault("source_dataset", "raw_dataset")
    state.setdefault("max_duration_seconds", MAX_DURATION_SECONDS)
    state.setdefault("current_video_id", None)
    state.setdefault("decisions", {})
    state.setdefault("recent_history", [])
    return state


def normalize_state(state: dict[str, Any], *, dataset_id: str) -> dict[str, Any]:
    if not isinstance(state, dict):
        state = default_hf_state() if dataset_id == DATASET_ID else default_state()
    state.setdefault("dataset_id", dataset_id)
    state.setdefault("max_duration_seconds", MAX_DURATION_SECONDS)
    state.setdefault("current_video_id", None)
    state.setdefault("decisions", {})
    state.setdefault("recent_history", [])
    return state


def keep_manifest_decisions_only(state: dict[str, Any], videos_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid_ids = set(videos_by_id)
    decisions = state.get("decisions", {})
    if isinstance(decisions, dict):
        state["decisions"] = {video_id: decision for video_id, decision in decisions.items() if video_id in valid_ids}

    history = state.get("recent_history", [])
    if isinstance(history, list):
        state["recent_history"] = [item for item in history if item.get("video_id") in valid_ids]

    current_id = state.get("current_video_id")
    if current_id and current_id not in valid_ids:
        state["current_video_id"] = None
    return state


def current_manifest_decisions(
    state: dict[str, Any],
    videos_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    decisions = state.get("decisions", {})
    if not isinstance(decisions, dict):
        return {}
    return {
        video_id: decision
        for video_id, decision in decisions.items()
        if video_id in videos_by_id and isinstance(decision, dict)
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    write_json(STATE_PATH, state)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_rows(state: dict[str, Any], videos_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    dataset_id = state.get("dataset_id") or "manual_seed_v2"
    for video_id, decision in sorted(state.get("decisions", {}).items()):
        video = videos_by_id.get(video_id, {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "video_id": video_id,
                "category": decision.get("category"),
                "category_label": CATEGORY_BY_ID.get(decision.get("category"), {}).get("label"),
                "video_path": video.get("video_relpath") or decision.get("video_path"),
                "info_path": video.get("info_relpath") or decision.get("info_path"),
                "duration_seconds": video.get("duration_seconds") or decision.get("duration_seconds"),
                "title": video.get("title") or decision.get("title"),
                "uploader": video.get("uploader") or decision.get("uploader"),
                "webpage_url": video.get("webpage_url") or decision.get("webpage_url"),
                "classified_at": decision.get("classified_at"),
            }
        )
    return rows


def group_rows_by_category(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_category: dict[str, list[dict[str, Any]]] = {category["id"]: [] for category in CATEGORIES}
    for row in rows:
        category = row.get("category")
        if category in rows_by_category:
            rows_by_category[category].append(row)
    return rows_by_category


def write_exports(state: dict[str, Any], videos_by_id: dict[str, dict[str, Any]]) -> None:
    rows = export_rows(state, videos_by_id)
    write_text(EXPORT_PATH, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    rows_by_category = group_rows_by_category(rows)

    for category in CATEGORIES:
        bucket_dir = BUCKETS_DIR / category["id"]
        bucket_rows = rows_by_category[category["id"]]
        write_json(
            bucket_dir / "videos.json",
            {
                "dataset_id": "manual_seed_v2",
                "category": category["id"],
                "category_label": category["label"],
                "count": len(bucket_rows),
                "videos": bucket_rows,
            },
        )
        write_text(
            bucket_dir / "videos.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bucket_rows),
        )


def select_next_video(
    videos: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    randomize: bool,
    enrich: bool = True,
) -> dict[str, Any] | None:
    decided_ids = set(state.get("decisions", {}).keys())
    candidates = [video for video in videos if video["video_id"] not in decided_ids]
    if not candidates:
        return None

    current_id = state.get("current_video_id")
    if current_id:
        for video in candidates:
            if video["video_id"] == current_id:
                enriched_video = enrich_video(video) if enrich else video
                if enriched_video:
                    return enriched_video
                break

    if randomize:
        candidates = random.sample(candidates, len(candidates))

    for video in candidates:
        enriched_video = enrich_video(video) if enrich else video
        if enriched_video:
            return enriched_video
    return None


def select_prefetch_videos(
    videos: list[dict[str, Any]],
    state: dict[str, Any],
    current_video_id: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    decided_ids = set(state.get("decisions", {}).keys())
    candidates = [video for video in videos if video["video_id"] not in decided_ids]
    if not candidates:
        return []

    start_index = 0
    for index, video in enumerate(candidates):
        if video["video_id"] == current_video_id:
            start_index = index + 1
            break
    return candidates[start_index : start_index + limit]


def set_current_video(state: dict[str, Any], video_id: str | None) -> None:
    state["current_video_id"] = video_id
    save_state(state)


def classify_video(
    state: dict[str, Any],
    video: dict[str, Any],
    category_id: str,
    videos_by_id: dict[str, dict[str, Any]],
) -> None:
    video_id = video["video_id"]
    previous_decision = state.get("decisions", {}).get(video_id)
    state.setdefault("decisions", {})[video_id] = {
        "category": category_id,
        "video_path": video["video_relpath"],
        "info_path": video["info_relpath"],
        "duration_seconds": video["duration_seconds"],
        "title": video.get("title", ""),
        "uploader": video.get("uploader", ""),
        "webpage_url": video.get("webpage_url", ""),
        "classified_at": now_iso(),
    }
    history = state.setdefault("recent_history", [])
    history.append({"video_id": video_id, "previous_decision": previous_decision})
    state["recent_history"] = history[-HISTORY_LIMIT:]
    state["current_video_id"] = None
    save_state(state)
    write_exports(state, videos_by_id)


def write_hf_exports(store: HfDatasetStore, state: dict[str, Any], videos_by_id: dict[str, dict[str, Any]]) -> None:
    rows = export_rows(state, videos_by_id)
    store.upload_funnel_outputs(
        state=state,
        export_rows=rows,
        rows_by_category=group_rows_by_category(rows),
        categories=CATEGORIES,
    )


def classify_hf_video(
    store: HfDatasetStore,
    decision_store: FirestoreDecisionStore | None,
    state: dict[str, Any],
    video: dict[str, Any],
    category_id: str,
    videos_by_id: dict[str, dict[str, Any]],
    extra_fields: dict[str, Any] | None = None,
) -> None:
    video_id = video["video_id"]
    previous_decision = state.get("decisions", {}).get(video_id)
    decision = {
        "category": category_id,
        "video_path": video["video_path"],
        "info_path": video.get("info_path"),
        "duration_seconds": video["duration_seconds"],
        "title": video.get("title", ""),
        "uploader": video.get("uploader", ""),
        "webpage_url": video.get("webpage_url", ""),
        "classified_at": now_iso(),
    }
    if extra_fields:
        decision.update(extra_fields)
    state.setdefault("decisions", {})[video_id] = decision
    history = state.setdefault("recent_history", [])
    history.append({"video_id": video_id, "previous_decision": previous_decision})
    state["recent_history"] = history[-HISTORY_LIMIT:]
    state["current_video_id"] = None
    state["updated_at"] = now_iso()
    st.session_state.pop("hf_current_video_id", None)
    if decision_store is not None:
        decision_store.upsert_funnel_decision(
            dataset_id=state.get("dataset_id") or DATASET_ID,
            video_id=video_id,
            decision=decision,
            annotator_id=get_config_value("ANNOTATOR_ID", "default"),
        )
    else:
        write_hf_exports(store, state, videos_by_id)


def undo_last(state: dict[str, Any], videos_by_id: dict[str, dict[str, Any]]) -> str | None:
    history = state.get("recent_history", [])
    if not history:
        return None
    item = history.pop()
    video_id = item["video_id"]
    previous_decision = item.get("previous_decision")
    if previous_decision:
        state.setdefault("decisions", {})[video_id] = previous_decision
    else:
        state.setdefault("decisions", {}).pop(video_id, None)
    state["recent_history"] = history[-HISTORY_LIMIT:]
    state["current_video_id"] = video_id
    save_state(state)
    write_exports(state, videos_by_id)
    return video_id


def undo_hf_last(
    store: HfDatasetStore,
    decision_store: FirestoreDecisionStore | None,
    state: dict[str, Any],
    videos_by_id: dict[str, dict[str, Any]],
) -> str | None:
    history = state.get("recent_history", [])
    if not history:
        return None
    item = history.pop()
    video_id = item["video_id"]
    previous_decision = item.get("previous_decision")
    if previous_decision:
        state.setdefault("decisions", {})[video_id] = previous_decision
        if decision_store is not None:
            decision_store.upsert_funnel_decision(
                dataset_id=state.get("dataset_id") or DATASET_ID,
                video_id=video_id,
                decision=previous_decision,
                annotator_id=get_config_value("ANNOTATOR_ID", "default"),
            )
    else:
        state.setdefault("decisions", {}).pop(video_id, None)
        if decision_store is not None:
            decision_store.delete_funnel_decision(
                dataset_id=state.get("dataset_id") or DATASET_ID,
                video_id=video_id,
            )
    state["recent_history"] = history[-HISTORY_LIMIT:]
    state["current_video_id"] = video_id
    state["updated_at"] = now_iso()
    st.session_state["hf_current_video_id"] = video_id
    if decision_store is None:
        write_hf_exports(store, state, videos_by_id)
    return video_id


def render_metric_row(videos: list[dict[str, Any]], state: dict[str, Any]) -> None:
    total = len(videos)
    video_ids = {video["video_id"] for video in videos}
    all_decisions = state.get("decisions", {})
    decisions = {
        video_id: decision
        for video_id, decision in all_decisions.items()
        if video_id in video_ids
    } if isinstance(all_decisions, dict) else {}
    done = len(decisions)
    remaining = max(0, total - done)
    cols = st.columns(4)
    cols[0].metric("Classified", done)
    cols[1].metric("Remaining", remaining)
    cols[2].metric("Raw videos", total)
    cols[3].metric("Max duration", f"{MAX_DURATION_SECONDS}s")

    if total:
        st.progress(done / total)


def main() -> None:
    st.set_page_config(page_title="Data Annotation Funnel", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1400px;
        }
        div[data-testid="stVideo"] {
            border: 1px solid #d7dde5;
            margin-left: auto;
            margin-right: auto;
            width: 70%;
            overflow: hidden;
        }
        div.stButton > button {
            min-height: 3.3rem;
            font-weight: 650;
        }
        .decision-help {
            color: #64748b;
            font-size: 0.88rem;
            margin-top: -0.4rem;
            min-height: 1.6rem;
        }
        .meta-box {
            border-top: 1px solid #e5e7eb;
            color: #475569;
            font-size: 0.92rem;
            margin-top: 1rem;
            padding-top: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Data Annotation Funnel")

    storage_backend = get_storage_backend()
    decision_backend = get_decision_backend()
    hf_store = None
    decision_store: FirestoreDecisionStore | None = None
    if storage_backend == "hf":
        hf_store = HfDatasetStore.from_config(root=ROOT)
        if decision_backend == "firestore":
            decision_store = FirestoreDecisionStore.from_config()
        videos = [
            video
            for video in hf_store.load_manifest()
            if float(video.get("duration_seconds") or 0) <= MAX_DURATION_SECONDS
        ]
        videos_by_id = {video["video_id"]: video for video in videos}
        state = normalize_state(hf_store.load_funnel_state(default_hf_state()), dataset_id=DATASET_ID)
        if decision_store is not None:
            state.setdefault("decisions", {}).update(decision_store.load_funnel_decisions(DATASET_ID))
        session_current_id = st.session_state.get("hf_current_video_id")
        if session_current_id and session_current_id not in state.get("decisions", {}):
            state["current_video_id"] = session_current_id
    else:
        videos = scan_raw_videos()
        videos_by_id = {video["video_id"]: video for video in videos}
        state = load_state()

    with st.sidebar:
        if hf_store is not None:
            st.caption("Backend: HF Dataset")
            st.caption(f"Repo: {hf_store.repo_id}")
            st.caption(f"Decision log: {decision_backend}")
        else:
            st.caption(f"Source: {RAW_DATASET_DIR}")
            st.caption(f"Target: {DATASET_DIR}")
        randomize = st.toggle("Random order", value=False)
        rescan_label = "Reload HF manifest" if hf_store is not None else "Rescan raw_dataset"
        if st.button(rescan_label, use_container_width=True):
            if hf_store is None:
                scan_raw_videos.clear()
            st.rerun()
        if st.button(
            "Back",
            use_container_width=True,
            disabled=not state.get("recent_history"),
        ):
            if hf_store is not None:
                undo_hf_last(hf_store, decision_store, state, videos_by_id)
            else:
                undo_last(state, videos_by_id)
            st.rerun()
        if st.button("Export indexes", use_container_width=True):
            if hf_store is not None:
                if decision_store is not None:
                    st.info("Firestore decisions are synced to HF by the Cloud Run Job.")
                else:
                    write_hf_exports(hf_store, state, videos_by_id)
            else:
                write_exports(state, videos_by_id)
                st.success("Exported")

        st.divider()
        st.subheader("Buckets")
        counts = {category["id"]: 0 for category in CATEGORIES}
        decisions_for_counts = (
            current_manifest_decisions(state, videos_by_id)
            if hf_store is not None
            else state.get("decisions", {})
        )
        for decision in decisions_for_counts.values():
            category = decision.get("category")
            if category in counts:
                counts[category] += 1
        for category in CATEGORIES:
            st.caption(f"{category['label']}: {counts[category['id']]}")

    if not videos:
        st.error("No raw videos under 60 seconds were found.")
        return

    render_metric_row(videos, state)

    top_left, top_right = st.columns([1, 2])
    with top_left:
        if st.button("Choose Video", type="primary", use_container_width=True):
            state["current_video_id"] = None
            if hf_store is not None:
                st.session_state.pop("hf_current_video_id", None)
            else:
                save_state(state)
            st.rerun()
    with top_right:
        st.caption("Videos already classified are skipped automatically.")

    video = select_next_video(videos, state, randomize=randomize, enrich=hf_store is None)
    if video is None:
        st.success("All available short videos are classified.")
        return

    if state.get("current_video_id") != video["video_id"]:
        if hf_store is not None:
            state["current_video_id"] = video["video_id"]
            st.session_state["hf_current_video_id"] = video["video_id"]
        else:
            set_current_video(state, video["video_id"])

    if hf_store is not None:
        current_cached = hf_store.is_video_cached(video)
        try:
            if current_cached:
                video_path = hf_store.download_video(video)
            else:
                with st.spinner(
                    f"Downloading video from HF Dataset... ({HF_DOWNLOAD_TIMEOUT_SECONDS}s timeout)"
                ):
                    video_path = hf_store.download_video_async(video).result(
                        timeout=HF_DOWNLOAD_TIMEOUT_SECONDS
                    )
        except TimeoutError:
            classify_hf_video(
                hf_store,
                decision_store,
                state,
                video,
                "annotation_problem",
                videos_by_id,
                {
                    "annotation_problem_reason": "hf_download_timeout",
                    "annotation_problem_seconds": HF_DOWNLOAD_TIMEOUT_SECONDS,
                },
            )
            st.warning("Video download took too long. Marked as annotation problem.")
            st.rerun()
        except Exception as exc:
            classify_hf_video(
                hf_store,
                decision_store,
                state,
                video,
                "annotation_problem",
                videos_by_id,
                {
                    "annotation_problem_reason": "hf_download_error",
                    "annotation_problem_error_type": type(exc).__name__,
                },
            )
            st.warning("Video download failed. Marked as annotation problem.")
            st.rerun()
        prefetched_ids = st.session_state.setdefault("hf_prefetched_video_ids", set())
        prefetch_videos = [
            item
            for item in select_prefetch_videos(
                videos,
                state,
                video["video_id"],
                limit=HF_PREFETCH_AHEAD,
            )
            if item["video_id"] not in prefetched_ids
        ]
        if prefetch_videos:
            hf_store.prefetch_videos(prefetch_videos)
            prefetched_ids.update(item["video_id"] for item in prefetch_videos)
        st.sidebar.divider()
        st.sidebar.subheader("HF Cache")
        st.sidebar.caption(f"Current video: {'cached' if current_cached else 'downloading'}")
        if prefetch_videos:
            st.sidebar.caption("Prefetch queued:")
            for item in prefetch_videos:
                st.sidebar.caption(f"- {item['video_id']}")
        else:
            st.sidebar.caption("Prefetch queue: already warm or empty")
    else:
        video_path = Path(video["video_path"])
    main_col, action_col = st.columns([3, 1], gap="large")

    with main_col:
        st.video(str(video_path))
        st.markdown(
            f"""
            <div class="meta-box">
              <strong>{video['video_id']}</strong><br>
              Duration: {video['duration_seconds']:.1f}s<br>
              Uploader: {video.get('uploader') or '-'}<br>
              Title: {video.get('title') or '-'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if video.get("webpage_url"):
            st.link_button("Open Source URL", video["webpage_url"])

    with action_col:
        st.subheader("Classify")
        for category in CATEGORIES:
            if st.button(category["label"], use_container_width=True):
                if hf_store is not None:
                    classify_hf_video(hf_store, decision_store, state, video, category["id"], videos_by_id)
                else:
                    classify_video(state, video, category["id"], videos_by_id)
                st.rerun()
            st.markdown(
                f"<div class='decision-help'>{category['help']}</div>",
                unsafe_allow_html=True,
            )

    updated_at = state.get("updated_at")
    if updated_at:
        st.caption(f"Last saved: {updated_at}")


if __name__ == "__main__":
    main()
