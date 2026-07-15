from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sort_key(row: dict[str, Any]) -> tuple[str, float, str]:
    timestamp = row.get("timestamp_seconds")
    try:
        timestamp_value = float(timestamp)
    except (TypeError, ValueError):
        timestamp_value = 0.0
    return (
        str(row.get("video_id") or ""),
        timestamp_value,
        str(row.get("frame_id") or ""),
    )


def normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": annotation.get("dataset_id") or DATASET_ID,
        "task": "text_frame_correction",
        "video_id": annotation.get("video_id"),
        "frame_id": annotation.get("frame_id"),
        "timestamp_seconds": annotation.get("timestamp_seconds"),
        "source_category": annotation.get("source_category"),
        "frame_path": annotation.get("frame_path"),
        "frame_gcs_url": annotation.get("frame_gcs_url"),
        "ocr_model": annotation.get("ocr_model"),
        "ocr_text": clean_text(annotation.get("ocr_text")),
        "subtitle_text": clean_text(annotation.get("subtitle_text")),
        "static_text": clean_text(annotation.get("static_text")),
        "other_text": clean_text(annotation.get("other_text")),
        "status": annotation.get("status") or "accepted",
        "annotator_id": annotation.get("annotator_id") or "unknown",
        "annotated_at": annotation.get("annotated_at"),
    }


def build_video_state(export_rows: list[dict[str, Any]]) -> dict[str, Any]:
    videos: dict[str, dict[str, Any]] = {}
    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in export_rows:
        video_id = str(row.get("video_id") or "")
        if video_id:
            rows_by_video[video_id].append(row)

    for video_id, rows in sorted(rows_by_video.items()):
        statuses = Counter(str(row.get("status") or "unknown") for row in rows)
        annotators = sorted({str(row.get("annotator_id") or "unknown") for row in rows})
        annotated_at_values = sorted(
            str(row.get("annotated_at"))
            for row in rows
            if row.get("annotated_at")
        )
        videos[video_id] = {
            "video_id": video_id,
            "annotated_frames": len(rows),
            "status_counts": dict(sorted(statuses.items())),
            "annotators": annotators,
            "first_annotated_at": annotated_at_values[0] if annotated_at_values else None,
            "last_annotated_at": annotated_at_values[-1] if annotated_at_values else None,
        }

    return {
        "dataset_id": DATASET_ID,
        "task": "text_frame_correction",
        "generated_at": now_iso(),
        "frame_count": len(export_rows),
        "video_count": len(videos),
        "videos": videos,
    }


def main() -> None:
    hf_store = HfDatasetStore.from_config(root=ROOT)
    decision_store = FirestoreDecisionStore.from_config()

    annotations = decision_store.load_text_frame_annotations(DATASET_ID)
    export_rows = [
        normalize_annotation(annotation)
        for annotation in annotations.values()
        if isinstance(annotation, dict)
    ]
    export_rows = sorted(export_rows, key=sort_key)
    video_state = build_video_state(export_rows)

    hf_store.upload_text_frame_outputs(
        export_rows=export_rows,
        video_state=video_state,
    )
    print(
        "synced "
        f"{len(export_rows)} Firestore text frame annotations "
        f"across {video_state['video_count']} videos to HF Dataset"
    )


if __name__ == "__main__":
    main()
