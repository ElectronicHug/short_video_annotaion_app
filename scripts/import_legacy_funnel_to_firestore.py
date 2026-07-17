from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCAL_SERVICE_ACCOUNT_PATH = ROOT / ".secrets" / "external-annotation-app-writer.service-account.json"
if (
    not any(
        name in os.environ
        for name in ("GCP_SERVICE_ACCOUNT_JSON", "GOOGLE_APPLICATION_CREDENTIALS")
    )
    and LOCAL_SERVICE_ACCOUNT_PATH.exists()
):
    os.environ["GCP_SERVICE_ACCOUNT_JSON"] = LOCAL_SERVICE_ACCOUNT_PATH.read_text(encoding="utf-8")

from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore


DEFAULT_LEGACY_EXPORT = (
    WORKSPACE_ROOT
    / "legacy_local_pipeline"
    / "datasets"
    / "manual_seed_v2"
    / "annotations"
    / "export.jsonl"
)
DEFAULT_ANNOTATOR_ID = "legacy_zhenya"
IMPORT_SOURCE = "legacy_manual_seed_v2_funnel"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def normalize_decision(row: dict[str, Any], *, imported_at: str, annotator_id: str) -> dict[str, Any]:
    decision = dict(row)
    decision["import_source"] = IMPORT_SOURCE
    decision["imported_at"] = imported_at
    decision["annotator_id"] = annotator_id
    decision.setdefault("dataset_id", DATASET_ID)
    return decision


def print_counter(title: str, values: list[str]) -> None:
    print(title)
    for key, count in Counter(values).most_common():
        print(f"  {key}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import legacy manual_seed_v2 funnel classifications into Firestore.",
    )
    parser.add_argument(
        "--legacy-export",
        type=Path,
        default=DEFAULT_LEGACY_EXPORT,
        help=f"Path to legacy funnel export JSONL. Default: {DEFAULT_LEGACY_EXPORT}",
    )
    parser.add_argument(
        "--annotator-id",
        default=DEFAULT_ANNOTATOR_ID,
        help=f"Annotator id to use for imported records. Default: {DEFAULT_ANNOTATOR_ID}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write missing records to Firestore. Without this flag the script is dry-run only.",
    )
    parser.add_argument(
        "--overwrite-conflicts",
        action="store_true",
        help="Overwrite existing Firestore records when categories differ. Not recommended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_export = args.legacy_export.resolve()
    if not legacy_export.exists():
        raise FileNotFoundError(f"Legacy export not found: {legacy_export}")

    legacy_rows = load_jsonl(legacy_export)
    legacy_by_id = {str(row["video_id"]): row for row in legacy_rows if row.get("video_id")}
    if len(legacy_by_id) != len(legacy_rows):
        print(f"WARNING: loaded {len(legacy_rows)} rows but only {len(legacy_by_id)} unique video_id values")

    hf_store = HfDatasetStore.from_config(root=ROOT)
    manifest = hf_store.load_manifest()
    manifest_by_id = {str(row["video_id"]): row for row in manifest if row.get("video_id")}
    short_manifest_ids = {
        str(row["video_id"])
        for row in manifest
        if row.get("video_id") and float(row.get("duration_seconds") or 0) <= 60
    }

    store = FirestoreDecisionStore.from_config()
    firestore_records = store.load_funnel_decision_records(DATASET_ID)
    firestore_by_id = {
        str(row["video_id"]): row
        for row in firestore_records
        if row.get("video_id")
    }

    missing_from_manifest = sorted(set(legacy_by_id) - set(manifest_by_id))
    missing_from_short_manifest = sorted(set(legacy_by_id) - short_manifest_ids)

    to_insert: list[tuple[str, dict[str, Any]]] = []
    same_existing: list[str] = []
    conflicts: list[tuple[str, str, str]] = []
    to_overwrite: list[tuple[str, dict[str, Any]]] = []

    for video_id, legacy_row in sorted(legacy_by_id.items()):
        if video_id not in manifest_by_id:
            continue
        current = firestore_by_id.get(video_id)
        if current is None:
            to_insert.append((video_id, legacy_row))
            continue
        legacy_category = str(legacy_row.get("category") or "")
        current_category = str(current.get("category") or "")
        if legacy_category == current_category:
            same_existing.append(video_id)
        else:
            conflicts.append((video_id, legacy_category, current_category))
            if args.overwrite_conflicts:
                to_overwrite.append((video_id, legacy_row))

    print("=== Legacy funnel import audit ===")
    print(f"legacy export: {legacy_export}")
    print(f"legacy rows: {len(legacy_rows)}")
    print(f"legacy unique videos: {len(legacy_by_id)}")
    print(f"present in HF manifest: {len(set(legacy_by_id) & set(manifest_by_id))}")
    print(f"present in HF short <=60 manifest: {len(set(legacy_by_id) & short_manifest_ids)}")
    print(f"missing from HF manifest: {len(missing_from_manifest)}")
    print(f"missing from short <=60 manifest: {len(missing_from_short_manifest)}")
    print(f"already same in Firestore: {len(same_existing)}")
    print(f"category conflicts: {len(conflicts)}")
    print(f"missing in Firestore, ready to insert: {len(to_insert)}")
    if args.overwrite_conflicts:
        print(f"conflicts requested for overwrite: {len(to_overwrite)}")
    print_counter("legacy categories:", [str(row.get("category") or "unknown") for row in legacy_rows])
    print_counter("insert categories:", [str(row.get("category") or "unknown") for _, row in to_insert])

    if conflicts:
        print("conflicts:")
        for video_id, legacy_category, current_category in conflicts[:50]:
            print(f"  {video_id}: legacy={legacy_category}, firestore={current_category}")

    if missing_from_manifest:
        print("missing from HF manifest:")
        for video_id in missing_from_manifest[:50]:
            print(f"  {video_id}")

    if not args.apply:
        print("DRY RUN ONLY. Re-run with --apply to write missing records.")
        return

    imported_at = datetime.now(timezone.utc).isoformat()
    written = 0
    for video_id, legacy_row in to_insert + to_overwrite:
        decision = normalize_decision(legacy_row, imported_at=imported_at, annotator_id=args.annotator_id)
        store.upsert_funnel_decision(
            dataset_id=DATASET_ID,
            video_id=video_id,
            decision=decision,
            annotator_id=args.annotator_id,
        )
        written += 1

    print(f"wrote {written} Firestore funnel decisions")
    print("HF sync is handled by Cloud Run Scheduler, or run scripts/sync_funnel_firestore_to_hf.py manually.")


if __name__ == "__main__":
    main()
