from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, try_to_load_from_cache
from huggingface_hub.errors import EntryNotFoundError

from .hf_tokens import get_hf_dataset_repo, get_hf_token


DATASET_ID = "short_video_ocr_dataset"
FUNNEL_STATE_PATH = "annotations/funnel_state.json"
FUNNEL_EXPORT_PATH = "annotations/funnel_export.jsonl"
TEXT_FRAME_CORRECTIONS_PATH = "annotations/text_frame_corrections.jsonl"
TEXT_VIDEO_STATE_PATH = "annotations/text_video_state.json"
FRAMES_MANIFEST_PATH = "frames_manifest.jsonl"
QWEN_FRAME_OCR_PATH = "ocr_predictions/qwen2_vl_2b_frame_ocr/qwen2_vl_2b_frame_ocr.jsonl"
MANUAL_SUBTITLE_TRANSCRIPT_DRAFTS_PATH = "transcripts/manual_frame_subtitle_transcript_drafts.jsonl"
OCR_LLM_TRANSCRIPTS_PATH = "transcripts/ocr_llm_transcripts.jsonl"
CORRECTED_TRANSCRIPTS_PATH = "transcripts/corrected_transcripts.jsonl"
TRANSCRIPT_VIDEO_STATE_PATH = "transcripts/transcript_video_state.json"
PREFETCH_WORKERS = 4


_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS)


class HfDatasetStore:
    def __init__(self, repo_id: str, read_token: str, write_token: str, cache_dir: Path) -> None:
        self.repo_id = repo_id
        self.read_token = read_token
        self.write_token = write_token or read_token
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api = HfApi(token=self.write_token)

    @classmethod
    def from_config(cls, *, root: Path) -> "HfDatasetStore":
        token_file = root.parent / ".hf_token"
        read_token = get_hf_token("HF_TOKEN_READ", token_file=token_file)
        write_token = get_hf_token("HF_TOKEN_WRITE", token_file=token_file)
        return cls(
            repo_id=get_hf_dataset_repo(),
            read_token=read_token,
            write_token=write_token,
            cache_dir=root / ".cache" / "hf_dataset",
        )

    def _download(self, path_in_repo: str, *, force_download: bool = False) -> Path:
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=self.read_token or None,
            cache_dir=str(self.cache_dir),
            force_download=force_download,
        )
        return Path(local_path)

    def load_manifest(self) -> list[dict[str, Any]]:
        path = self._download("manifest.jsonl")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def load_funnel_state(self, default_state: dict[str, Any]) -> dict[str, Any]:
        try:
            path = self._download(FUNNEL_STATE_PATH)
        except EntryNotFoundError:
            return default_state
        state = json.loads(path.read_text(encoding="utf-8"))
        return state if isinstance(state, dict) else default_state

    def load_frames_manifest(self) -> list[dict[str, Any]]:
        path = self._download(FRAMES_MANIFEST_PATH)
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def load_qwen_frame_ocr(self) -> list[dict[str, Any]]:
        try:
            path = self._download(QWEN_FRAME_OCR_PATH)
        except EntryNotFoundError:
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def load_text_frame_corrections(self) -> list[dict[str, Any]]:
        try:
            path = self._download(TEXT_FRAME_CORRECTIONS_PATH, force_download=True)
        except EntryNotFoundError:
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def load_manual_subtitle_transcript_drafts(self) -> list[dict[str, Any]]:
        try:
            path = self._download(MANUAL_SUBTITLE_TRANSCRIPT_DRAFTS_PATH, force_download=True)
        except EntryNotFoundError:
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def load_ocr_llm_transcripts(self) -> list[dict[str, Any]]:
        try:
            path = self._download(OCR_LLM_TRANSCRIPTS_PATH, force_download=True)
        except EntryNotFoundError:
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def load_corrected_transcripts(self) -> list[dict[str, Any]]:
        try:
            path = self._download(CORRECTED_TRANSCRIPTS_PATH, force_download=True)
        except EntryNotFoundError:
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def download_video(self, video: dict[str, Any]) -> Path:
        return self._download(str(video["video_path"]))

    def video_url(self, video: dict[str, Any]) -> str:
        return hf_hub_url(
            repo_id=self.repo_id,
            filename=str(video["video_path"]),
            repo_type="dataset",
        )

    def download_video_async(self, video: dict[str, Any]) -> Future[Path]:
        return _PREFETCH_EXECUTOR.submit(self.download_video, video)

    def is_video_cached(self, video: dict[str, Any]) -> bool:
        cached = try_to_load_from_cache(
            repo_id=self.repo_id,
            filename=str(video["video_path"]),
            repo_type="dataset",
            cache_dir=str(self.cache_dir),
        )
        return isinstance(cached, str) and Path(cached).exists()

    def prefetch_videos(self, videos: list[dict[str, Any]]) -> None:
        for video in videos:
            _PREFETCH_EXECUTOR.submit(self.download_video, video)

    def upload_funnel_outputs(
        self,
        *,
        state: dict[str, Any],
        export_rows: list[dict[str, Any]],
        rows_by_category: dict[str, list[dict[str, Any]]],
        categories: list[dict[str, str]],
    ) -> None:
        out_dir = self.cache_dir / "outbox"
        annotations_dir = out_dir / "annotations"
        buckets_dir = out_dir / "buckets"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        state_path = annotations_dir / "funnel_state.json"
        export_path = annotations_dir / "funnel_export.jsonl"
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        export_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in export_rows),
            encoding="utf-8",
        )

        for category in categories:
            category_id = category["id"]
            bucket_rows = rows_by_category.get(category_id, [])
            bucket_dir = buckets_dir / category_id
            bucket_dir.mkdir(parents=True, exist_ok=True)
            (bucket_dir / "videos.json").write_text(
                json.dumps(
                    {
                        "dataset_id": DATASET_ID,
                        "category": category_id,
                        "category_label": category["label"],
                        "count": len(bucket_rows),
                        "videos": bucket_rows,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (bucket_dir / "videos.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bucket_rows),
                encoding="utf-8",
            )

        self.api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo="",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Update funnel annotations",
        )

    def upload_text_frame_outputs(
        self,
        *,
        export_rows: list[dict[str, Any]],
        video_state: dict[str, Any],
    ) -> None:
        out_dir = self.cache_dir / "text_outbox"
        annotations_dir = out_dir / "annotations"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        export_path = annotations_dir / "text_frame_corrections.jsonl"
        state_path = annotations_dir / "text_video_state.json"
        export_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in export_rows),
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps(video_state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.api.upload_folder(
            folder_path=str(out_dir),
            path_in_repo="",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message="Update text frame annotations",
        )
