# Agent Notes

This repository contains Streamlit annotation apps for the short-video OCR dataset.

Start by reading:

- `docs/agent_runbook.md` for architecture, launch commands, secrets, and troubleshooting.
- `README.md` for longer historical deployment notes.
- `docs/streamlit_secrets_template.toml` for Streamlit Cloud secrets.
- `docs/private_vpn_streamlit_secrets.example.toml` for private VPN/Linux server secrets.

Guidelines:

- Keep only app code, lightweight configs, docs, and tests in git.
- Do not commit raw videos, extracted frames, crops, generated results, annotation state, `.env`, tokens, or Streamlit secrets.
- The source of truth for source videos/frames/OCR artifacts is the Hugging Face Dataset repo `ElectronicHug/short_video_ocr_dataset`.
- The live source of truth for annotation clicks is GCP Firestore:
  - `funnel_decisions`
  - `funnel_claims`
  - `text_frame_annotations`
- HF is updated from Firestore by scheduled Cloud Run sync jobs.
- Prefer UTF-8 JSON/JSONL for all state and export files.
- The active deployment entry point is the multipage Streamlit app `app.py`.
- Do not print, copy, or commit service-account JSON, HF tokens, Streamlit secrets, or `.streamlit/secrets.toml`.
- If the app fails on startup, first check whether Streamlit/GCP/HF secrets are configured.
