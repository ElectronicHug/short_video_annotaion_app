# Short Video Annotation App

Streamlit annotation tools for the short-video OCR dataset.

This repository contains app code only. Videos, manifests, annotation state, and exports are stored in the Hugging Face Dataset repo:

```text
ElectronicHug/short_video_ocr_dataset
```

For the fastest operational handoff, read:

```text
docs/agent_runbook.md
```

## Streamlit Deployment

Deploy this GitHub repo as one multipage Streamlit Community Cloud app:

```text
app.py
```

Pages:

```text
Funnel
Text Frame Correction
Stats
```

Funnel categories:

```text
matched
title_matched
partially_matched
unmatched
annotation_problem
ignore
```

In HF mode, video download is capped at 20 seconds. A timeout or download
error is automatically saved as `annotation_problem` in Firestore, so the app
can move on instead of hanging.

## Local Setup

Base environment: `vlm-env` with Python `3.13.14`.

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
```

## Data Layout

Local filesystem defaults:

```text
raw_dataset/
datasets/manual_seed_v2/
results/
```

Path overrides:

```text
APP_ROOT
RAW_DATASET_DIR
DATASET_DIR
RESULTS_DIR
```

Future hosted mode should read/write through:

```text
STORAGE_BACKEND=hf
DECISION_BACKEND=firestore
HF_DATASET_REPO=ElectronicHug/short_video_ocr_dataset
HF_TOKEN_READ=<read token>
HF_TOKEN_WRITE=<write token>
GCP_PROJECT_ID=short-video-dataset-ocr
FIRESTORE_COLLECTION=funnel_decisions
FIRESTORE_CLAIMS_COLLECTION=funnel_claims
FIRESTORE_TEXT_COLLECTION=text_frame_annotations
CLAIM_TTL_MINUTES=30
TEXT_CLAIM_TTL_MINUTES=60
HF_VIDEO_MODE=url
AUTH_COOKIE_SECRET=<random long string>
```

`AUTH_COOKIE_SECRET` signs the 24-hour login cookie. It should be stable across
app restarts and must stay in Streamlit secrets, not in git.

For Streamlit Community Cloud, add the same values to app secrets. Prefer the
`[gcp_service_account]` table shown below for Firestore writes. For local HF
testing, the app can also read `../.hf_token` with `HF_TOKEN_READ = "..."` and
`HF_TOKEN_WRITE = "..."`.

## GCP Firestore Decision Log

Target project and region:

```text
GCP_PROJECT_ID=short-video-dataset-ocr
GCP_REGION=europe-central2
```

Enable APIs:

```powershell
gcloud config set project short-video-dataset-ocr
gcloud services enable firestore.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

Create Firestore database:

```powershell
gcloud firestore databases create `
  --database="(default)" `
  --location=europe-central2
```

Local auth for Firestore Python clients:

```powershell
gcloud auth application-default login
```

Firestore collection:

```text
funnel_decisions/{dataset_id}__funnel__{video_id}
```

Document shape:

```json
{
  "dataset_id": "short_video_ocr_dataset",
  "task": "funnel",
  "video_id": "...",
  "annotator_id": "default",
  "category": "matched",
  "decision": {
    "category": "matched",
    "video_path": "videos/...mp4",
    "info_path": "videos/...info.json",
    "duration_seconds": 12.3,
    "title": "...",
    "uploader": "...",
    "webpage_url": "...",
    "classified_at": "..."
  },
  "updated_at": "<server timestamp>",
  "synced_to_hf_at": null
}
```

## Streamlit Community Cloud Secrets

```toml
STORAGE_BACKEND = "hf"
DECISION_BACKEND = "firestore"
HF_DATASET_REPO = "ElectronicHug/short_video_ocr_dataset"
HF_TOKEN_READ = "hf_..."
GCP_PROJECT_ID = "short-video-dataset-ocr"
FIRESTORE_COLLECTION = "funnel_decisions"
HF_VIDEO_MODE = "url"

[auth_users.zhenya]
display_name = "Zhenya"
role = "owner"
password = "..."

[auth_users.colleague]
display_name = "Колега"
role = "reviewer"
password = "..."

[auth_users.annotator_1]
display_name = "Анотатор 1"
role = "annotator"
password = "..."

[auth_users.annotator_2]
display_name = "Анотатор 2"
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
client_email = "streamlit-firestore-writer@short-video-dataset-ocr.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

The Streamlit service account needs Firestore read/write permissions, for example `roles/datastore.user`.

## Cloud Run Sync Job

The sync job reads Firestore decisions and writes one HF Dataset commit containing:

```text
annotations/funnel_state.json
annotations/funnel_export.jsonl
buckets/*/videos.json
buckets/*/videos.jsonl
```

Text frame correction sync reads Firestore frame annotations and writes one HF
Dataset commit containing:

```text
annotations/text_frame_corrections.jsonl
annotations/text_video_state.json
```

Run text sync locally:

```powershell
..\vlm-env\python.exe scripts\sync_text_frame_annotations_firestore_to_hf.py
```

For local runs, either configure Application Default Credentials:

```powershell
gcloud auth application-default login
```

or point `GOOGLE_APPLICATION_CREDENTIALS` to a local service-account JSON file
that is not committed to git:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"
..\vlm-env\python.exe scripts\sync_text_frame_annotations_firestore_to_hf.py
```

Create a secret for the HF write token:

```powershell
gcloud secrets create hf-token-write --replication-policy="automatic"
gcloud secrets versions add hf-token-write --data-file=hf_token_write.txt
```

Create Artifact Registry repository:

```powershell
gcloud artifacts repositories create short-video-ocr `
  --repository-format=docker `
  --location=europe-central2
```

Build and push the sync image:

```powershell
gcloud builds submit `
  --config cloudbuild.sync.yaml `
  --substitutions _IMAGE=europe-central2-docker.pkg.dev/short-video-dataset-ocr/short-video-ocr/funnel-sync:latest
```

Build and push the text correction sync image:

```powershell
gcloud builds submit `
  --config cloudbuild.sync.yaml `
  --substitutions _DOCKERFILE=Dockerfile.text-sync,_IMAGE=europe-central2-docker.pkg.dev/short-video-dataset-ocr/short-video-ocr/text-frame-sync:latest
```

Grant the Cloud Run job service account access to Firestore and the HF token secret. For the default Compute service account:

```powershell
$PROJECT_NUMBER = gcloud projects describe short-video-dataset-ocr --format="value(projectNumber)"
$RUN_SA = "$PROJECT_NUMBER-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding short-video-dataset-ocr `
  --member="serviceAccount:$RUN_SA" `
  --role="roles/datastore.user"

gcloud secrets add-iam-policy-binding hf-token-write `
  --member="serviceAccount:$RUN_SA" `
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding short-video-dataset-ocr `
  --member="serviceAccount:$RUN_SA" `
  --role="roles/run.developer"
```

Create the Cloud Run Job:

```powershell
gcloud run jobs create funnel-sync-to-hf `
  --image europe-central2-docker.pkg.dev/short-video-dataset-ocr/short-video-ocr/funnel-sync:latest `
  --region europe-central2 `
  --set-env-vars STORAGE_BACKEND=hf,DECISION_BACKEND=firestore,GCP_PROJECT_ID=short-video-dataset-ocr,HF_DATASET_REPO=ElectronicHug/short_video_ocr_dataset,FIRESTORE_COLLECTION=funnel_decisions `
  --set-secrets HF_TOKEN_WRITE=hf-token-write:latest,HF_TOKEN_READ=hf-token-write:latest
```

Create the text correction Cloud Run Job:

```powershell
gcloud run jobs create text-frame-sync-to-hf `
  --image europe-central2-docker.pkg.dev/short-video-dataset-ocr/short-video-ocr/text-frame-sync:latest `
  --region europe-central2 `
  --set-env-vars STORAGE_BACKEND=hf,DECISION_BACKEND=firestore,GCP_PROJECT_ID=short-video-dataset-ocr,HF_DATASET_REPO=ElectronicHug/short_video_ocr_dataset,FIRESTORE_TEXT_COLLECTION=text_frame_annotations `
  --set-secrets HF_TOKEN_WRITE=hf-token-write:latest,HF_TOKEN_READ=hf-token-write:latest
```

Run manually:

```powershell
gcloud run jobs execute funnel-sync-to-hf --region europe-central2 --wait
```

Run text sync manually:

```powershell
gcloud run jobs execute text-frame-sync-to-hf --region europe-central2 --wait
```

Schedule every 15 minutes:

```powershell
gcloud scheduler jobs create http funnel-sync-to-hf-every-15m `
  --location=europe-central2 `
  --schedule="*/15 * * * *" `
  --uri="https://run.googleapis.com/v2/projects/short-video-dataset-ocr/locations/europe-central2/jobs/funnel-sync-to-hf:run" `
  --http-method=POST `
  --oauth-service-account-email="$RUN_SA"
```

## Rules

- Do not commit videos, frames, results, tokens, or `.env`.
- Store annotation state and exports in the HF Dataset repo, not in Streamlit Cloud local disk.
- Keep app code separate from dataset artifacts.
