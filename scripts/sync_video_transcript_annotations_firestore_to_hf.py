from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import (
    CORRECTED_TRANSCRIPTS_PATH,
    DATASET_ID,
    HfDatasetStore,
    TRANSCRIPT_VIDEO_STATE_PATH,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def sort_key(row: dict[str, Any]) -> str:
    return str(row.get("video_id") or "")


def normalize_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": annotation.get("dataset_id") or DATASET_ID,
        "task": "video_transcript_correction",
        "video_id": annotation.get("video_id"),
        "transcript_text": clean_text(annotation.get("transcript_text")),
        "status": annotation.get("status") or "accepted",
        "candidate_label": annotation.get("candidate_label"),
        "annotator_id": annotation.get("annotator_id") or "unknown",
        "annotated_at": annotation.get("annotated_at"),
    }


def build_video_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(row.get("status") or "unknown") for row in rows)
    annotators = Counter(str(row.get("annotator_id") or "unknown") for row in rows)
    return {
        "dataset_id": DATASET_ID,
        "task": "video_transcript_correction",
        "generated_at": now_iso(),
        "video_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "annotator_counts": dict(sorted(annotators.items())),
    }


def upload_outputs(store: HfDatasetStore, rows: list[dict[str, Any]], video_state: dict[str, Any]) -> None:
    out_dir = store.cache_dir / "corrected_transcript_outbox"
    transcripts_dir = out_dir / "transcripts"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / "corrected_transcripts.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (transcripts_dir / "transcript_video_state.json").write_text(
        json.dumps(video_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store.api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo="",
        repo_id=store.repo_id,
        repo_type="dataset",
        commit_message="Update corrected video transcripts",
    )


def main() -> None:
    hf_store = HfDatasetStore.from_config(root=ROOT)
    decision_store = FirestoreDecisionStore.from_config()
    annotations = decision_store.load_video_transcript_annotations(DATASET_ID)
    rows = sorted(
        [normalize_annotation(annotation) for annotation in annotations.values() if isinstance(annotation, dict)],
        key=sort_key,
    )
    video_state = build_video_state(rows)
    upload_outputs(hf_store, rows, video_state)
    print(
        f"synced {len(rows)} Firestore video transcript annotations to HF "
        f"({CORRECTED_TRANSCRIPTS_PATH}, {TRANSCRIPT_VIDEO_STATE_PATH})"
    )


if __name__ == "__main__":
    main()
