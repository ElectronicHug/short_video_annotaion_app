# Annotation App Agent Runbook

This document is the quick handoff for humans and AI agents working on
`short_video_annotaion_app`.

## What This Repo Is

This repo contains the Streamlit annotation UI for the short-video OCR dataset.
It does not store videos, frames, OCR results, or real annotation exports in git.

Active entry point:

```text
app.py
```

Active pages:

```text
pages/1_Funnel.py                 video-level classification
pages/2_Text_Frame_Correction.py  frame-level text correction
pages/3_Stats.py                  Firestore annotation statistics
```

Core modules:

```text
annotation_app/funnel_app.py
annotation_app/text_frame_correction_app.py
annotation_app/stats_app.py
annotation_app/common/auth.py
annotation_app/common/firestore_decision_store.py
annotation_app/common/hf_dataset_store.py
annotation_app/common/hf_tokens.py
```

## Data Sources

Hugging Face Dataset repo:

```text
ElectronicHug/short_video_ocr_dataset
```

HF stores source artifacts and synced annotation exports, including videos,
frames, OCR predictions, and JSONL exports.

GCP Firestore is the live annotation log and the source of truth while people
are annotating:

```text
funnel_decisions          video classification decisions
funnel_claims             temporary video locks for funnel annotation
text_frame_annotations    frame text corrections
```

Cloud Run Jobs sync Firestore back to HF in batch commits:

```text
funnel-sync-to-hf
text-frame-sync-to-hf
```

## What Was Built

Funnel page:

- Shows one short video at a time.
- Writes every click to Firestore.
- Uses video-level locks to reduce duplicated work.
- Supports categories such as matched, title matched, unmatched, ignore, and
  annotation problem.

Text frame correction page:

- Shows one OCR frame at a time.
- Lets annotators fill:
  - subtitles
  - static text
  - other text
- Prefills subtitles from Qwen OCR minus previous static text when possible.
- Prefills static text from the previous frame.
- Shows previous frame text on the right as read-only context with copy buttons.
- Writes every saved frame to Firestore.

Stats page:

- Reads directly from Firestore.
- Shows video classification totals and per-annotator breakdown.
- Shows text correction totals and per-annotator breakdown.

Auth:

- `AUTH_MODE = "password"` for public Streamlit Cloud.
- `AUTH_MODE = "profile_only"` for private VPN/Linux server deployments.
- Profile-only mode assumes VPN is the outer access layer and only asks users to
  choose their profile so annotations are attributed correctly.

## Required Secrets

Never commit real secrets.

Local ignored paths:

```text
.streamlit/secrets.toml
.secrets/
```

Streamlit Cloud template:

```text
docs/streamlit_secrets_template.toml
```

Private VPN/Linux server template:

```text
docs/private_vpn_streamlit_secrets.example.toml
```

Important settings:

```toml
STORAGE_BACKEND = "hf"
DECISION_BACKEND = "firestore"
HF_DATASET_REPO = "ElectronicHug/short_video_ocr_dataset"
HF_TOKEN_READ = "hf_..."
GCP_PROJECT_ID = "short-video-dataset-ocr"
FIRESTORE_COLLECTION = "funnel_decisions"
FIRESTORE_CLAIMS_COLLECTION = "funnel_claims"
FIRESTORE_TEXT_COLLECTION = "text_frame_annotations"
AUTH_COOKIE_SECRET = "long-random-stable-string"
```

For Streamlit Cloud:

```toml
AUTH_MODE = "password"
```

For private VPN server:

```toml
AUTH_MODE = "profile_only"
```

GCP service account:

- Use the minimal writer account for annotation apps:

```text
external-annotation-app-writer@short-video-dataset-ocr.iam.gserviceaccount.com
```

- Expected role:

```text
roles/datastore.user
```

- Do not put owner/editor keys, HF write tokens, or Secret Manager accessor
  credentials into the annotation app.

If the app fails with Firestore credential or PEM errors, the most common cause
is missing or malformed `GCP_SERVICE_ACCOUNT_JSON` / `[gcp_service_account]`.

## Run Locally on Windows

From repo root:

```powershell
..\vlm-env\python.exe -m pip install -r requirements.txt
..\vlm-env\python.exe -m streamlit run app.py
```

For local secrets, create:

```text
short_video_annotaion_app/.streamlit/secrets.toml
```

Use one of the templates from `docs/`.

## Deploy on Streamlit Community Cloud

Create a Streamlit app from the GitHub repo:

```text
ElectronicHug/short_video_annotaion_app
```

Main file path:

```text
app.py
```

Copy secrets from:

```text
docs/streamlit_secrets_template.toml
```

Fill:

- `HF_TOKEN_READ`
- `AUTH_COOKIE_SECRET`
- `[auth_users.*]`
- full GCP service-account credentials

If the site starts but cannot load data or write annotations, check app secrets
first. In Streamlit Cloud, error messages are often redacted, so use
`Manage app -> Logs`.

## Deploy on Private Linux VPN Server

Target used in current deployment:

```text
Tailscale 1.98.4
Linux 6.8.0-101-generic
Python 3.12
uv
```

Current server layout:

```text
/home/emrozek/apps/short_video_annotaion_app
/home/emrozek/apps/short_video_annotaion_app/.venv
/home/emrozek/apps/short_video_annotaion_app/.streamlit/secrets.toml
```

Install:

```bash
mkdir -p ~/apps
cd ~/apps
git clone https://github.com/ElectronicHug/short_video_annotaion_app.git
cd short_video_annotaion_app
wget -qO- https://astral.sh/uv/install.sh | sh
~/.local/bin/uv venv --python 3.12
~/.local/bin/uv pip install -r requirements.txt
```

Secrets:

```bash
mkdir -p .streamlit
nano .streamlit/secrets.toml
chmod 700 .streamlit
chmod 600 .streamlit/secrets.toml
```

Use:

```text
docs/private_vpn_streamlit_secrets.example.toml
```

Run:

```bash
cd ~/apps/short_video_annotaion_app
nohup .venv/bin/streamlit run app.py \
  --server.headless true \
  --server.address 100.97.153.111 \
  --server.port 8502 \
  > streamlit-8502.log 2>&1 < /dev/null &
```

Open:

```text
http://100.97.153.111:8502
```

Update:

```bash
cd ~/apps/short_video_annotaion_app
git pull --ff-only
.venv/bin/python -m py_compile app.py annotation_app/*.py annotation_app/common/*.py
pkill -f 'streamlit run app.py --server.headless true --server.address 100.97.153.111 --server.port 8502'
nohup .venv/bin/streamlit run app.py \
  --server.headless true \
  --server.address 100.97.153.111 \
  --server.port 8502 \
  > streamlit-8502.log 2>&1 < /dev/null &
```

Logs:

```bash
tail -n 100 streamlit-8502.log
```

## Common Problems

`ModuleNotFoundError`

- The venv is not active or dependencies were not installed.
- Run `uv pip install -r requirements.txt` inside the repo.

Firestore credential / PEM error

- Secrets are missing or the private key is malformed.
- Make sure the full service-account JSON is present.
- Do not leave placeholder `...` in `private_key`.

No videos or frames

- Check `HF_TOKEN_READ`.
- Check `HF_DATASET_REPO`.
- Check whether the HF Dataset contains the expected manifest/OCR files.

Writes do not appear in Firestore

- Check `DECISION_BACKEND = "firestore"`.
- Check the service account has `roles/datastore.user`.
- Check `GCP_PROJECT_ID`.

Stats page is empty

- Stats reads Firestore, not HF.
- If HF has synced exports but Firestore is empty, the page will be empty.

Streamlit Cloud error is redacted

- Open app logs in Streamlit Cloud.
- Most likely missing `st.secrets` values.

## Safety Rules

- Do not commit `.streamlit/secrets.toml`.
- Do not commit `.secrets/`.
- Do not print service-account private keys or HF tokens.
- If a service-account key is pasted into chat or otherwise exposed, create a
  new key and delete the old key in GCP IAM.
- Keep generated data, frame caches, and annotation exports out of this app repo.
