# Agent Notes

This repository contains Streamlit annotation apps for the short-video OCR dataset.

Guidelines:

- Keep only app code, lightweight configs, docs, and tests in git.
- Do not commit raw videos, extracted frames, crops, generated results, annotation state, `.env`, tokens, or Streamlit secrets.
- The source of truth for data and annotations is the Hugging Face Dataset repo `electronichug/short_video_ocr_dataset`.
- Prefer UTF-8 JSON/JSONL for all state and export files.
- Streamlit Cloud deployments use different main file paths under `annotation_app/`.
