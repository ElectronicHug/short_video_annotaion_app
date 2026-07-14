from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID, HfDatasetStore
from annotation_app.funnel_app import (
    MAX_DURATION_SECONDS,
    default_hf_state,
    normalize_state,
    write_hf_exports,
)


def main() -> None:
    hf_store = HfDatasetStore.from_config(root=ROOT)
    decision_store = FirestoreDecisionStore.from_config()

    videos = [
        video
        for video in hf_store.load_manifest()
        if float(video.get("duration_seconds") or 0) <= MAX_DURATION_SECONDS
    ]
    videos_by_id = {video["video_id"]: video for video in videos}

    state = normalize_state(hf_store.load_funnel_state(default_hf_state()), dataset_id=DATASET_ID)
    firestore_decisions = decision_store.load_funnel_decisions(DATASET_ID)
    state.setdefault("decisions", {}).update(firestore_decisions)
    state["current_video_id"] = None

    write_hf_exports(hf_store, state, videos_by_id)
    print(f"synced {len(firestore_decisions)} Firestore funnel decisions to HF Dataset")


if __name__ == "__main__":
    main()
