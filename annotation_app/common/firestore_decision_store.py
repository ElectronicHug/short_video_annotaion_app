from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from google.cloud import firestore
from google.oauth2 import service_account

from .hf_tokens import get_config_value


DEFAULT_COLLECTION = "funnel_decisions"
TASK = "funnel"


def _streamlit_secret_value(name: str) -> Any:
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


def _mapping_to_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def _service_account_info() -> dict[str, Any] | None:
    section_value = _streamlit_secret_value("gcp_service_account")
    if isinstance(section_value, Mapping):
        return _mapping_to_dict(section_value)

    raw_value = _streamlit_secret_value("GCP_SERVICE_ACCOUNT_JSON") or get_config_value(
        "GCP_SERVICE_ACCOUNT_JSON"
    )
    if not raw_value:
        return None
    if isinstance(raw_value, Mapping):
        return _mapping_to_dict(raw_value)

    text_value = str(raw_value).strip()
    try:
        return json.loads(text_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "GCP_SERVICE_ACCOUNT_JSON must be valid JSON. In Streamlit secrets, "
            "prefer a [gcp_service_account] table or use TOML literal triple quotes: "
            "GCP_SERVICE_ACCOUNT_JSON = '''{...}'''."
        ) from exc


class FirestoreDecisionStore:
    def __init__(
        self,
        *,
        project_id: str,
        collection: str = DEFAULT_COLLECTION,
        database: str | None = None,
    ) -> None:
        credentials = None
        info = _service_account_info()
        if info:
            credentials = service_account.Credentials.from_service_account_info(info)
            project_id = project_id or info.get("project_id", "")

        kwargs: dict[str, Any] = {"project": project_id or None, "credentials": credentials}
        if database:
            kwargs["database"] = database
        self.client = firestore.Client(**kwargs)
        self.collection = self.client.collection(collection)

    @classmethod
    def from_config(cls) -> "FirestoreDecisionStore":
        return cls(
            project_id=get_config_value("FIRESTORE_PROJECT_ID")
            or get_config_value("GCP_PROJECT_ID", "short-video-dataset-ocr"),
            collection=get_config_value("FIRESTORE_COLLECTION", DEFAULT_COLLECTION),
            database=get_config_value("FIRESTORE_DATABASE"),
        )

    @staticmethod
    def document_id(dataset_id: str, video_id: str, task: str = TASK) -> str:
        return f"{dataset_id}__{task}__{video_id}"

    def load_funnel_decisions(self, dataset_id: str) -> dict[str, dict[str, Any]]:
        query = (
            self.collection.where("dataset_id", "==", dataset_id)
            .where("task", "==", TASK)
        )
        decisions: dict[str, dict[str, Any]] = {}
        for document in query.stream():
            data = document.to_dict() or {}
            video_id = data.get("video_id")
            decision = data.get("decision")
            if isinstance(video_id, str) and isinstance(decision, dict):
                decisions[video_id] = decision
        return decisions

    def upsert_funnel_decision(
        self,
        *,
        dataset_id: str,
        video_id: str,
        decision: dict[str, Any],
        annotator_id: str = "default",
    ) -> None:
        document = self.collection.document(self.document_id(dataset_id, video_id))
        document.set(
            {
                "dataset_id": dataset_id,
                "task": TASK,
                "video_id": video_id,
                "annotator_id": annotator_id,
                "decision": decision,
                "category": decision.get("category"),
                "classified_at": decision.get("classified_at"),
                "updated_at": firestore.SERVER_TIMESTAMP,
                "synced_to_hf_at": None,
            },
            merge=True,
        )

    def delete_funnel_decision(self, *, dataset_id: str, video_id: str) -> None:
        self.collection.document(self.document_id(dataset_id, video_id)).delete()
