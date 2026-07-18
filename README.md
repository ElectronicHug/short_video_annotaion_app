# Short Video Annotation App

Streamlit annotation tools for the short-video OCR/ASR dataset.

This repository contains the app and sync code only. Videos, manifests, OCR
outputs, annotation exports, and transcript candidates live in the Hugging Face
Dataset repo:

```text
ElectronicHug/short_video_ocr_dataset
```

Firestore is the live write log. HF is the shared batch artifact store.

## Pages

The app is a multipage Streamlit app with `app.py` as the entrypoint.

```text
pages/1_Funnel.py                  video classification
pages/2_Text_Frame_Correction.py   frame-level subtitle/static text correction
pages/3_Transcript_Correction.py   final video transcript correction
pages/4_Problem_Fixes.py           owner/reviewer repair tools
pages/5_Stats.py                   live Firestore/HF statistics
```

The sidebar labels are Ukrainian for annotators.

## Data Flow

Live writes:

```text
Streamlit -> Firestore
```

Batch sync:

```text
Firestore -> Hugging Face Dataset
```

Main Firestore collections:

```text
funnel_decisions
funnel_claims
text_frame_annotations
video_transcript_annotations
```

Important HF outputs:

```text
annotations/funnel_state.json
annotations/funnel_export.jsonl
annotations/text_frame_corrections.jsonl
annotations/text_video_state.json
transcripts/corrected_transcripts.jsonl
transcripts/transcript_video_state.json
```

## Setup

From workspace root:

```powershell
.\vlm-env\python.exe -m pip install -r short_video_annotaion_app\requirements.txt
```

From this repository:

```powershell
..\vlm-env\python.exe -m pip install -r requirements.txt
```

Run locally:

```powershell
..\vlm-env\python.exe -m streamlit run app.py
```

Direct page debugging:

```powershell
..\vlm-env\python.exe -m streamlit run annotation_app\funnel_app.py
..\vlm-env\python.exe -m streamlit run annotation_app\text_frame_correction_app.py
..\vlm-env\python.exe -m streamlit run annotation_app\transcript_correction_app.py
```

## Secrets

Do not commit `.secrets/`, `.streamlit/secrets.toml`, `.env`, token files, or
service account JSON files.

Minimum Streamlit secrets shape:

```toml
STORAGE_BACKEND = "hf"
DECISION_BACKEND = "firestore"
HF_DATASET_REPO = "ElectronicHug/short_video_ocr_dataset"
HF_TOKEN_READ = "hf_..."
GCP_PROJECT_ID = "short-video-dataset-ocr"
FIRESTORE_COLLECTION = "funnel_decisions"
FIRESTORE_CLAIMS_COLLECTION = "funnel_claims"
FIRESTORE_TEXT_COLLECTION = "text_frame_annotations"
FIRESTORE_TRANSCRIPT_COLLECTION = "video_transcript_annotations"
HF_VIDEO_MODE = "url"
AUTH_COOKIE_SECRET = "random-long-stable-secret"
TEXT_CLAIM_TTL_MINUTES = 20160

[auth_users.zhenya]
display_name = "Zhenya"
role = "owner"
password = "..."

[auth_users.annotator_1]
display_name = "Annotator 1"
role = "annotator"
password = "..."

[gcp_service_account]
type = "service_account"
project_id = "short-video-dataset-ocr"
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----
"""
client_email = "external-annotation-app-writer@short-video-dataset-ocr.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

If the app fails on startup in Streamlit Cloud or on a server, first check that
these secrets exist and that the service account key is valid TOML/JSON with
real newlines inside `private_key`.

The app service account should be limited to the permissions it needs, normally
Firestore read/write for the configured project.

## Local Linux / VPN Server

Known server:

```text
ssh emrozek@100.97.153.110
```

Known app path:

```text
/home/emrozek/apps/short_video_annotaion_app
```

Known URL:

```text
http://100.97.153.111:8502/
```

Typical update/restart:

```bash
cd /home/emrozek/apps/short_video_annotaion_app
git pull --ff-only
pkill -f "streamlit.*app.py" || true
nohup .venv/bin/streamlit run app.py --server.headless true --server.address 100.97.153.111 --server.port 8502 > streamlit.out.log 2> streamlit.err.log &
```

This is not the project owner's personal server. Avoid destructive commands and
do not change unrelated services.

## Streamlit Cloud

Deploy this repo with:

```text
Main file path: app.py
```

Copy secrets through the Streamlit Cloud UI. Do not commit local secrets.

## Sync Scripts

Text-frame annotations Firestore -> HF:

```powershell
$env:GCP_SERVICE_ACCOUNT_JSON = Get-Content -Raw .secrets\external-annotation-app-writer.service-account.json
..\vlm-env\python.exe scripts\sync_text_frame_annotations_firestore_to_hf.py
```

Video transcript annotations Firestore -> HF:

```powershell
$env:GCP_SERVICE_ACCOUNT_JSON = Get-Content -Raw .secrets\external-annotation-app-writer.service-account.json
..\vlm-env\python.exe scripts\sync_video_transcript_annotations_firestore_to_hf.py
```

Funnel sync is already deployed as a Cloud Run Job in:

```text
project: short-video-dataset-ocr
region: europe-central2
job: funnel-sync-to-hf
```

Check current GCP state before creating or replacing Cloud Run jobs.

## Annotation Notes

Funnel labels:

```text
matched
title_matched
partially_matched
unmatched
annotation_problem
ignore
```

Frame text correction stores:

- `subtitle_text`: what is spoken;
- `static_text`: visible but not spoken;
- `other_text`: optional notes/other visible text.

Final transcript correction stores one final `transcript_text` per video.

The problem fixes page is the preferred way to repair old or wrong records
because it writes audit metadata to Firestore.

## Agent Notes

For broader project state and commands, read the workspace-level:

```text
../WORK_CONTEXT.md
../RUNBOOK.md
```
