from __future__ import annotations

import json
import os
import random
import re
import sys
import uuid
from concurrent.futures import TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
from huggingface_hub import hf_hub_url

ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_app.common.auth import current_annotator_id, require_login
from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore
from annotation_app.common.hf_tokens import get_config_value, get_storage_backend

RAW_DATASET_DIR = Path(os.getenv("RAW_DATASET_DIR", ROOT / "raw_dataset"))
DATASET_DIR = Path(os.getenv("DATASET_DIR", ROOT / "datasets" / "manual_seed_v2"))
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
STATE_PATH = ANNOTATIONS_DIR / "state.json"
EXPORT_PATH = ANNOTATIONS_DIR / "export.jsonl"
BUCKETS_DIR = DATASET_DIR / "buckets"
MAX_DURATION_SECONDS = 60
HISTORY_LIMIT = 5
HF_PREFETCH_AHEAD = 2
HF_DOWNLOAD_TIMEOUT_SECONDS = 20
HF_VIDEO_MODE_DIRECT_URL = "url"
HF_VIDEO_MODE_DOWNLOAD = "download"
DEFAULT_VIDEO_WIDTH_PERCENT = 88
DEFAULT_CLAIM_TTL_MINUTES = 30

CATEGORIES = [
    {
        "id": "no_usable_speech",
        "label": "Без корисного мовлення",
        "help": "Немає мовлення, придатного для transcript dataset: тиша, музика, спів, шум, фонова мова або дуже короткі вигуки.",
    },
    {
        "id": "speech_no_text",
        "label": "Мовлення без тексту",
        "help": "Є корисне мовлення, але на екрані немає видимого тексту для OCR/subtitle порівняння.",
    },
    {
        "id": "text_without_subtitles",
        "label": "Текст без субтитрів",
        "help": "Є корисне мовлення і видимий текст, але це не субтитри до мовлення: заголовок, банер, список, текстова картка, UI тощо.",
    },
    {
        "id": "insufficient_subtitle_alignment",
        "label": "Недостатній збіг субтитрів",
        "help": "Є subtitle-like текст, але він покриває замало мовлення або помітно відрізняється, тому як pseudo label йому не довіряємо.",
    },
    {
        "id": "partially_matched",
        "label": "Субтитри частково збігаються",
        "help": "Субтитри загалом правильні, але покривають не все корисне мовлення: приблизно 80–95%.",
    },
    {
        "id": "title_matched",
        "label": "Субтитри + додатковий текст",
        "help": "Субтитри майже повністю збігаються з мовленням, але є значущий додатковий текст, який треба відділяти.",
    },
    {
        "id": "matched",
        "label": "Субтитри збігаються",
        "help": "Видимі субтитри майже повністю і буквально відповідають корисному мовленню; додаткового значущого тексту немає.",
    },
    {
        "id": "problem",
        "label": "Проблема",
        "help": "Технічна проблема: відео не відкривається, погана якість, неможливо надійно розмітити через збій або іншу технічну причину.",
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


def get_hf_video_mode() -> str:
    mode = get_config_value("HF_VIDEO_MODE", HF_VIDEO_MODE_DIRECT_URL).lower()
    if mode in {HF_VIDEO_MODE_DIRECT_URL, HF_VIDEO_MODE_DOWNLOAD}:
        return mode
    return HF_VIDEO_MODE_DIRECT_URL


def get_claim_ttl_minutes() -> int:
    raw_value = get_config_value("CLAIM_TTL_MINUTES", str(DEFAULT_CLAIM_TTL_MINUTES))
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return DEFAULT_CLAIM_TTL_MINUTES


def current_session_id() -> str:
    session_id = st.session_state.get("annotation_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        st.session_state["annotation_session_id"] = session_id
    return str(session_id)


def is_own_claim(claim: dict[str, Any]) -> bool:
    return (
        str(claim.get("annotator_id") or "") == current_annotator_id()
        or str(claim.get("session_id") or "") == current_session_id()
    )


def locked_video_ids(active_claims: dict[str, dict[str, Any]]) -> set[str]:
    return {
        video_id
        for video_id, claim in active_claims.items()
        if not is_own_claim(claim)
    }


def hf_video_url(store: HfDatasetStore, video: dict[str, Any]) -> str:
    video_url_method = getattr(store, "video_url", None)
    if callable(video_url_method):
        return str(video_url_method(video))
    return hf_hub_url(
        repo_id=store.repo_id,
        filename=str(video["video_path"]),
        repo_type="dataset",
    )


def resolve_video_source(store: HfDatasetStore | None, video: dict[str, Any]) -> tuple[str | None, str]:
    gcs_url = video.get("video_gcs_url")
    if isinstance(gcs_url, str) and gcs_url:
        return gcs_url, "gcs"
    mirror_url = video.get("video_mirror_url") or video.get("video_cdn_url")
    if isinstance(mirror_url, str) and mirror_url:
        return mirror_url, "mirror"
    if store is not None and video.get("video_path"):
        return hf_video_url(store, video), "hf"
    source_url = video.get("webpage_url")
    if isinstance(source_url, str) and source_url:
        return source_url, "source_link"
    return None, "missing"


def hf_download_futures() -> dict[str, Any]:
    futures = st.session_state.setdefault("hf_download_futures", {})
    return futures if isinstance(futures, dict) else {}


def describe_video_size(video: dict[str, Any]) -> str:
    filesize = video.get("filesize")
    if not filesize:
        return "unknown size"
    try:
        return f"{float(filesize) / 1024 / 1024:.1f} MB"
    except (TypeError, ValueError):
        return "unknown size"


def remember_download_problem(video: dict[str, Any], reason: str, detail: str) -> None:
    st.session_state["last_download_problem"] = {
        "video_id": video.get("video_id"),
        "video_path": video.get("video_path"),
        "size": describe_video_size(video),
        "reason": reason,
        "detail": detail,
        "time": now_iso(),
    }


def video_column_weights(width_percent: int) -> list[float]:
    center = max(0.55, min(1.0, width_percent / 100))
    side = max(0.0, (1.0 - center) / 2)
    return [side, center, side]


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
                "annotator_id": decision.get("annotator_id"),
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
    locked_ids: set[str] | None = None,
    skipped_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    decided_ids = set(state.get("decisions", {}).keys())
    locked_ids = locked_ids or set()
    skipped_ids = skipped_ids or set()
    candidates = [
        video
        for video in videos
        if (
            video["video_id"] not in decided_ids
            and video["video_id"] not in locked_ids
            and video["video_id"] not in skipped_ids
        )
    ]
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
    locked_ids: set[str] | None = None,
    skipped_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    decided_ids = set(state.get("decisions", {}).keys())
    locked_ids = locked_ids or set()
    skipped_ids = skipped_ids or set()
    candidates = [
        video
        for video in videos
        if (
            video["video_id"] not in decided_ids
            and video["video_id"] not in locked_ids
            and video["video_id"] not in skipped_ids
        )
    ]
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
        "annotator_id": current_annotator_id(),
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
        "annotator_id": current_annotator_id(),
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
            annotator_id=decision.get("annotator_id") or current_annotator_id(),
        )
    else:
        write_hf_exports(store, state, videos_by_id)


def skip_current_video(
    decision_store: FirestoreDecisionStore | None,
    state: dict[str, Any],
    video: dict[str, Any],
) -> None:
    video_id = video["video_id"]
    skipped_ids = st.session_state.setdefault("funnel_skipped_video_ids", [])
    if video_id not in skipped_ids:
        skipped_ids.append(video_id)
    state["current_video_id"] = None
    st.session_state.pop("hf_current_video_id", None)
    if decision_store is not None:
        decision_store.release_funnel_claim(
            dataset_id=state.get("dataset_id") or DATASET_ID,
            video_id=video_id,
        )
    else:
        save_state(state)


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
                annotator_id=previous_decision.get("annotator_id") or current_annotator_id(),
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
    cols[0].metric("Розмічено", done)
    cols[1].metric("Залишилось", remaining)
    cols[2].metric("Відео", total)
    cols[3].metric("Макс. тривалість", f"{MAX_DURATION_SECONDS}s")

    if total:
        st.progress(done / total)


def main() -> None:
    st.set_page_config(page_title="Відбір відео", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1680px;
        }
        div[data-testid="stVideo"] {
            border: 1px solid #d7dde5;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            max-height: 58vh;
            overflow: hidden;
        }
        div.stButton > button {
            min-height: 3.3rem;
            font-weight: 650;
        }
        .decision-help {
            color: #64748b;
            font-size: 0.88rem;
            line-height: 1.35;
            min-height: 3.3rem;
            padding-top: 0.15rem;
        }
        .decision-separator {
            border-top: 1px solid #e5e7eb;
            margin: 0.45rem 0 0.7rem;
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
    active_user = require_login()
    if active_user is None:
        return

    storage_backend = get_storage_backend()
    decision_backend = get_decision_backend()
    hf_video_mode = get_hf_video_mode()
    claim_ttl_minutes = get_claim_ttl_minutes()
    hf_store = None
    decision_store: FirestoreDecisionStore | None = None
    active_claims: dict[str, dict[str, Any]] = {}
    claim_locked_ids: set[str] = set()
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
            state["decisions"] = decision_store.load_funnel_decisions(DATASET_ID)
            active_claims = decision_store.load_active_funnel_claims(DATASET_ID)
            claim_locked_ids = locked_video_ids(active_claims)
        session_current_id = st.session_state.get("hf_current_video_id")
        if (
            session_current_id
            and session_current_id not in state.get("decisions", {})
            and session_current_id not in claim_locked_ids
        ):
            state["current_video_id"] = session_current_id
        elif session_current_id in claim_locked_ids:
            st.session_state.pop("hf_current_video_id", None)
    else:
        videos = scan_raw_videos()
        videos_by_id = {video["video_id"]: video for video in videos}
        state = load_state()

    with st.sidebar:
        if hf_store is not None:
            st.caption("Сховище: HF Dataset")
            st.caption(f"Repo: {hf_store.repo_id}")
            st.caption(f"Журнал рішень: {decision_backend}")
            st.caption(f"Режим відео: {hf_video_mode}")
            if decision_store is not None:
                st.caption(f"Активні блокування: {len(claim_locked_ids)}")
                st.caption(f"Час блокування: {claim_ttl_minutes} хв")
        else:
            st.caption(f"Джерело: {RAW_DATASET_DIR}")
            st.caption(f"Ціль: {DATASET_DIR}")
        video_width_percent = st.slider(
            "Ширина відео",
            min_value=55,
            max_value=100,
            value=int(st.session_state.get("video_width_percent", DEFAULT_VIDEO_WIDTH_PERCENT)),
            step=5,
            help="Налаштувати розмір відео для цієї сесії браузера.",
        )
        st.session_state["video_width_percent"] = video_width_percent
        randomize = st.toggle("Випадковий порядок", value=False)
        rescan_label = "Оновити HF manifest" if hf_store is not None else "Пересканувати raw_dataset"
        if st.button(rescan_label, use_container_width=True):
            if hf_store is None:
                scan_raw_videos.clear()
            st.rerun()
        if st.button("Експортувати індекси", use_container_width=True):
            if hf_store is not None:
                if decision_store is not None:
                    st.info("Рішення з Firestore синхронізуються в HF через Cloud Run Job.")
                else:
                    write_hf_exports(hf_store, state, videos_by_id)
            else:
                write_exports(state, videos_by_id)
                st.success("Експортовано")

        last_problem = st.session_state.get("last_download_problem")
        if isinstance(last_problem, dict):
            st.divider()
            st.subheader("Остання проблема із завантаженням")
            st.caption(f"Відео: {last_problem.get('video_id')}")
            st.caption(f"Розмір: {last_problem.get('size')}")
            st.caption(f"Причина: {last_problem.get('reason')}")
            st.caption(f"Деталі: {last_problem.get('detail')}")

        st.divider()
        st.subheader("Категорії")
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
        st.error("Не знайдено відео до 60 секунд.")
        return

    skipped_video_ids = set(st.session_state.get("funnel_skipped_video_ids", []))
    video = select_next_video(
        videos,
        state,
        randomize=randomize,
        enrich=hf_store is None,
        locked_ids=claim_locked_ids,
        skipped_ids=skipped_video_ids,
    )
    if video is None:
        st.success("Усі доступні короткі відео вже розмічені або тимчасово заблоковані.")
        return

    if hf_store is not None and decision_store is not None:
        claim_result = decision_store.claim_funnel_video(
            dataset_id=state.get("dataset_id") or DATASET_ID,
            video_id=video["video_id"],
            annotator_id=current_annotator_id(),
            session_id=current_session_id(),
            ttl_minutes=claim_ttl_minutes,
        )
        if not claim_result.get("claimed"):
            state["current_video_id"] = None
            st.session_state.pop("hf_current_video_id", None)
            st.info("Це відео щойно взяв інший анотатор. Вибираю інше.")
            st.rerun()

    if state.get("current_video_id") != video["video_id"]:
        if hf_store is not None:
            state["current_video_id"] = video["video_id"]
            st.session_state["hf_current_video_id"] = video["video_id"]
        else:
            set_current_video(state, video["video_id"])

    if hf_store is not None and hf_video_mode == HF_VIDEO_MODE_DIRECT_URL:
        video_path, video_source = resolve_video_source(hf_store, video)
        st.sidebar.divider()
        st.sidebar.subheader("HF Відео")
        st.sidebar.caption(f"Режим: прямий URL ({video_source})")
        st.sidebar.caption(f"Розмір: {describe_video_size(video)}")
    elif hf_store is not None:
        current_cached = hf_store.is_video_cached(video)
        futures = hf_download_futures()
        current_future = futures.get(video["video_id"])
        try:
            if current_cached:
                video_path = hf_store.download_video(video)
            else:
                if current_future is None:
                    current_future = hf_store.download_video_async(video)
                    futures[video["video_id"]] = current_future
                with st.spinner(
                    f"Завантажую відео з HF Dataset... (таймаут {HF_DOWNLOAD_TIMEOUT_SECONDS}s)"
                ):
                    video_path = current_future.result(timeout=HF_DOWNLOAD_TIMEOUT_SECONDS)
                futures.pop(video["video_id"], None)
        except TimeoutError:
            if current_future is not None:
                current_future.cancel()
            futures.pop(video["video_id"], None)
            remember_download_problem(
                video,
                "hf_download_timeout",
                f"Timed out after {HF_DOWNLOAD_TIMEOUT_SECONDS}s; file size {describe_video_size(video)}",
            )
            classify_hf_video(
                hf_store,
                decision_store,
                state,
                video,
                "problem",
                videos_by_id,
                {
                    "problem_reason": "hf_download_timeout",
                    "problem_seconds": HF_DOWNLOAD_TIMEOUT_SECONDS,
                },
            )
            st.warning("Відео завантажувалось занадто довго. Позначено як проблему з розміткою.")
            st.rerun()
        except Exception as exc:
            futures.pop(video["video_id"], None)
            remember_download_problem(
                video,
                "hf_download_error",
                f"{type(exc).__name__}: {str(exc)[:240]}",
            )
            classify_hf_video(
                hf_store,
                decision_store,
                state,
                video,
                "problem",
                videos_by_id,
                {
                    "problem_reason": "hf_download_error",
                    "problem_error_type": type(exc).__name__,
                },
            )
            st.warning("Не вдалося завантажити відео. Позначено як проблему з розміткою.")
            st.rerun()
        prefetch_videos = [
            item
            for item in select_prefetch_videos(
                videos,
                state,
                video["video_id"],
                limit=HF_PREFETCH_AHEAD,
                locked_ids=claim_locked_ids,
                skipped_ids=skipped_video_ids,
            )
            if item["video_id"] not in futures and not hf_store.is_video_cached(item)
        ]
        if prefetch_videos:
            for item in prefetch_videos:
                futures[item["video_id"]] = hf_store.download_video_async(item)
        st.sidebar.divider()
        st.sidebar.subheader("HF Cache")
        st.sidebar.caption(f"Поточне відео: {'у кеші' if current_cached else 'завантажено зараз'}")
        st.sidebar.caption(f"Поточний розмір: {describe_video_size(video)}")
        video_source = "hf_download"
        if prefetch_videos:
            st.sidebar.caption("Черга попереднього завантаження:")
            for item in prefetch_videos:
                st.sidebar.caption(f"- {item['video_id']} ({describe_video_size(item)})")
        else:
            st.sidebar.caption("Черга попереднього завантаження: вже прогріта або порожня")
    else:
        video_path = Path(video["video_path"])
        video_source = "local"
    main_col, action_col = st.columns([1.45, 1.15], gap="large")

    with main_col:
        video_left, video_center, video_right = st.columns(video_column_weights(video_width_percent))
        with video_center:
            if video_path and video_source != "source_link":
                st.video(str(video_path))
            elif video_path:
                st.warning("Прямий файл відео недоступний. Відкрий посилання на джерело вручну.")
                st.link_button("Відкрити джерело", str(video_path))
            else:
                st.error("Для цього елемента немає URL відео.")
        st.markdown(
            f"""
            <div class="meta-box">
              <strong>{video['video_id']}</strong><br>
              Тривалість: {video['duration_seconds']:.1f}s<br>
              Автор: {video.get('uploader') or '-'}<br>
              Назва: {video.get('title') or '-'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if video.get("webpage_url"):
            st.link_button("Відкрити джерело", video["webpage_url"])

    with action_col:
        st.subheader("Класифікація")
        for category in [category for category in CATEGORIES if category["id"] != "problem"]:
            button_col, help_col = st.columns([0.82, 2.05], gap="medium")
            with button_col:
                clicked = st.button(category["label"], use_container_width=True, key=f"funnel_category_{category['id']}")
            with help_col:
                st.markdown(
                    f"<div class='decision-help'>{category['help']}</div>",
                    unsafe_allow_html=True,
                )
            if clicked:
                if hf_store is not None:
                    classify_hf_video(hf_store, decision_store, state, video, category["id"], videos_by_id)
                else:
                    classify_video(state, video, category["id"], videos_by_id)
                st.rerun()
            st.markdown("<div class='decision-separator'></div>", unsafe_allow_html=True)
        st.divider()
        problem_category = CATEGORY_BY_ID["problem"]
        problem_col, problem_help_col = st.columns([0.82, 2.05], gap="medium")
        with problem_col:
            problem_clicked = st.button(
                problem_category["label"],
                use_container_width=True,
                key="funnel_category_problem",
            )
        with problem_help_col:
            st.markdown(
                f"<div class='decision-help'>{problem_category['help']}</div>",
                unsafe_allow_html=True,
            )
        if problem_clicked:
            if hf_store is not None:
                classify_hf_video(hf_store, decision_store, state, video, "problem", videos_by_id)
            else:
                classify_video(state, video, "problem", videos_by_id)
            st.rerun()
        st.markdown("<div class='decision-separator'></div>", unsafe_allow_html=True)
        if st.button("Наступне відео", use_container_width=True, key="funnel_skip_video"):
            skip_current_video(decision_store, state, video)
            st.rerun()
        if st.button("Назад", use_container_width=True, disabled=not state.get("recent_history"), key="funnel_bottom_back"):
            if hf_store is not None:
                undo_hf_last(hf_store, decision_store, state, videos_by_id)
            else:
                undo_last(state, videos_by_id)
            st.rerun()

    updated_at = state.get("updated_at")
    if updated_at:
        st.caption(f"Останнє збереження: {updated_at}")


if __name__ == "__main__":
    main()
