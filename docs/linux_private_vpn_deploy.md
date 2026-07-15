# Linux Private VPN Deploy

Target environment used for planning:

```text
Tailscale: 1.98.4
Kernel: Linux 6.8.0-101-generic
App: Streamlit multipage app
Repo: https://github.com/ElectronicHug/short_video_annotaion_app
```

This deployment is intended for a private VPN/Tailscale server. The VPN is the
outer access-control layer. Inside the app, users choose a required profile, so
Firestore annotations are written under the correct `annotator_id` instead of
`default`.

## Security Model

Use:

```text
AUTH_MODE = "profile_only"
```

Use the minimal service account:

```text
external-annotation-app-writer@short-video-dataset-ocr.iam.gserviceaccount.com
```

Expected IAM role:

```text
roles/datastore.user
```

Do not put these on the private server:

```text
HF_TOKEN_WRITE
Secret Manager accessor credentials
Owner / Editor service-account keys
```

Firestore -> HF sync is still handled by Cloud Run Jobs in the GCP project.

## Install

```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv
```

Clone:

```bash
mkdir -p ~/apps
cd ~/apps
git clone https://github.com/ElectronicHug/short_video_annotaion_app.git
cd short_video_annotaion_app
```

Create venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Secrets

Create the Streamlit secrets folder:

```bash
mkdir -p .streamlit
nano .streamlit/secrets.toml
```

Use this template:

```text
docs/private_vpn_streamlit_secrets.example.toml
```

For the GCP service account, paste the full JSON key for:

```text
external-annotation-app-writer@short-video-dataset-ocr.iam.gserviceaccount.com
```

The current local key file is expected to be stored outside git at:

```text
../short_video_annotaion_app/.secrets/external-annotation-app-writer.service-account.json
```

## Run Manually

```bash
source .venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open through Tailscale:

```text
http://<tailscale-ip-or-hostname>:8501
```

## systemd Service

Create service file:

```bash
sudo nano /etc/systemd/system/short-video-annotation.service
```

Example:

```ini
[Unit]
Description=Short Video OCR Annotation Streamlit App
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/<linux-user>/apps/short_video_annotaion_app
ExecStart=/home/<linux-user>/apps/short_video_annotaion_app/.venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable short-video-annotation
sudo systemctl start short-video-annotation
sudo systemctl status short-video-annotation
```

Logs:

```bash
journalctl -u short-video-annotation -f
```

## Update

```bash
cd ~/apps/short_video_annotaion_app
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart short-video-annotation
```

## Notes

- Tailscale reconnects can create a new Streamlit session. The app stores the
  selected profile in a signed 24-hour cookie/query token, so users should not
  need to choose their profile on every rerun.
- If a user chooses the wrong profile, annotations will be attributed to that
  profile. This mode assumes VPN-level trust.
- If the service-account key is exposed, delete it in GCP IAM and create a new
  key.
