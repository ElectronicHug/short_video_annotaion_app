from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from google.cloud import firestore
from google.oauth2 import service_account

from .hf_tokens import get_config_value


DEFAULT_COLLECTION = "funnel_decisions"
DEFAULT_CLAIMS_COLLECTION = "funnel_claims"
DEFAULT_TEXT_COLLECTION = "text_frame_annotations"
TASK = "funnel"
TEXT_TASK = "text_frame_correction"


def _streamlit_secret_value(name: str) -> Any:
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


def _mapping_to_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _is_active_claim(data: Mapping[str, Any], now: datetime) -> bool:
    expires_at = _parse_datetime(data.get("expires_at"))
    return bool(expires_at and expires_at > now)


def _service_account_info() -> dict[str, Any] | None:
    section_value = _streamlit_secret_value("gcp_service_account")
    if isinstance(section_value, Mapping):
        info = _mapping_to_dict(section_value)
        private_key = info.get("private_key")
        if isinstance(private_key, str):
            info["private_key"] = private_key.replace("\\n", "\n")
        return info

    raw_value = _streamlit_secret_value("GCP_SERVICE_ACCOUNT_JSON") or get_config_value(
        "GCP_SERVICE_ACCOUNT_JSON"
    )
    if not raw_value:
        return None
    if isinstance(raw_value, Mapping):
        info = _mapping_to_dict(raw_value)
        private_key = info.get("private_key")
        if isinstance(private_key, str):
            info["private_key"] = private_key.replace("\\n", "\n")
        return info

    text_value = str(raw_value).strip()
    try:
        info = json.loads(text_value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "GCP_SERVICE_ACCOUNT_JSON must be valid JSON. In Streamlit secrets, "
            "prefer a [gcp_service_account] table or use TOML literal triple quotes: "
            "GCP_SERVICE_ACCOUNT_JSON = '''{...}'''."
        ) from exc
    if isinstance(info, dict):
        private_key = info.get("private_key")
        if isinstance(private_key, str):
            info["private_key"] = private_key.replace("\\n", "\n")
    return info


class FirestoreDecisionStore:
    def __init__(
        self,
        *,
        project_id: str,
        collection: str = DEFAULT_COLLECTION,
        claims_collection: str = DEFAULT_CLAIMS_COLLECTION,
        text_collection: str = DEFAULT_TEXT_COLLECTION,
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
        self.claims_collection = self.client.collection(claims_collection)
        self.text_collection = self.client.collection(text_collection)

    @classmethod
    def from_config(cls) -> "FirestoreDecisionStore":
        return cls(
            project_id=get_config_value("FIRESTORE_PROJECT_ID")
            or get_config_value("GCP_PROJECT_ID", "short-video-dataset-ocr"),
            collection=get_config_value("FIRESTORE_COLLECTION", DEFAULT_COLLECTION),
            claims_collection=get_config_value("FIRESTORE_CLAIMS_COLLECTION", DEFAULT_CLAIMS_COLLECTION),
            text_collection=get_config_value("FIRESTORE_TEXT_COLLECTION", DEFAULT_TEXT_COLLECTION),
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

    def load_funnel_decision_records(self, dataset_id: str) -> list[dict[str, Any]]:
        query = (
            self.collection.where("dataset_id", "==", dataset_id)
            .where("task", "==", TASK)
        )
        records: list[dict[str, Any]] = []
        for document in query.stream():
            data = document.to_dict() or {}
            decision = data.get("decision")
            if not isinstance(decision, dict):
                decision = {}
            records.append(
                {
                    **data,
                    "document_id": document.id,
                    "category": data.get("category") or decision.get("category"),
                    "annotator_id": data.get("annotator_id") or decision.get("annotator_id") or "unknown",
                    "classified_at": data.get("classified_at") or decision.get("classified_at"),
                }
            )
        return records

    def load_active_funnel_claims(self, dataset_id: str) -> dict[str, dict[str, Any]]:
        now = datetime.now(timezone.utc)
        query = (
            self.claims_collection.where("dataset_id", "==", dataset_id)
            .where("task", "==", TASK)
        )
        claims: dict[str, dict[str, Any]] = {}
        for document in query.stream():
            data = document.to_dict() or {}
            video_id = data.get("video_id")
            if isinstance(video_id, str) and _is_active_claim(data, now):
                claims[video_id] = data
        return claims

    def claim_funnel_video(
        self,
        *,
        dataset_id: str,
        video_id: str,
        annotator_id: str,
        session_id: str,
        ttl_minutes: int = 30,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=max(1, ttl_minutes))
        document_id = self.document_id(dataset_id, video_id)
        decision_ref = self.collection.document(document_id)
        claim_ref = self.claims_collection.document(document_id)
        transaction = self.client.transaction()

        @firestore.transactional
        def _claim(transaction: firestore.Transaction) -> dict[str, Any]:
            decision_snapshot = decision_ref.get(transaction=transaction)
            if decision_snapshot.exists:
                decision_data = decision_snapshot.to_dict() or {}
                if isinstance(decision_data.get("decision"), dict):
                    return {"claimed": False, "reason": "already_decided"}

            claim_snapshot = claim_ref.get(transaction=transaction)
            if claim_snapshot.exists:
                claim_data = claim_snapshot.to_dict() or {}
                claim_owner = str(claim_data.get("annotator_id") or "")
                claim_session = str(claim_data.get("session_id") or "")
                same_owner = claim_owner == annotator_id or claim_session == session_id
                if _is_active_claim(claim_data, now) and not same_owner:
                    return {
                        "claimed": False,
                        "reason": "already_claimed",
                        "claim": claim_data,
                    }

            claim = {
                "dataset_id": dataset_id,
                "task": TASK,
                "video_id": video_id,
                "annotator_id": annotator_id,
                "session_id": session_id,
                "claimed_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
            transaction.set(claim_ref, claim, merge=True)
            return {"claimed": True, "claim": claim}

        return _claim(transaction)

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
        self.claims_collection.document(self.document_id(dataset_id, video_id)).delete()

    def load_text_frame_annotations(self, dataset_id: str) -> dict[str, dict[str, Any]]:
        query = (
            self.text_collection.where("dataset_id", "==", dataset_id)
            .where("task", "==", TEXT_TASK)
        )
        annotations: dict[str, dict[str, Any]] = {}
        for document in query.stream():
            data = document.to_dict() or {}
            video_id = data.get("video_id")
            frame_id = data.get("frame_id")
            annotation = data.get("annotation")
            if isinstance(video_id, str) and isinstance(frame_id, str) and isinstance(annotation, dict):
                annotations[f"{video_id}/{frame_id}"] = annotation
        return annotations

    def load_text_frame_annotation_records(self, dataset_id: str) -> list[dict[str, Any]]:
        query = (
            self.text_collection.where("dataset_id", "==", dataset_id)
            .where("task", "==", TEXT_TASK)
        )
        records: list[dict[str, Any]] = []
        for document in query.stream():
            data = document.to_dict() or {}
            annotation = data.get("annotation")
            if not isinstance(annotation, dict):
                annotation = {}
            records.append(
                {
                    **data,
                    "document_id": document.id,
                    "annotator_id": data.get("annotator_id") or annotation.get("annotator_id") or "unknown",
                    "status": data.get("status") or annotation.get("status"),
                    "annotated_at": data.get("annotated_at") or annotation.get("annotated_at"),
                    "subtitle_text": annotation.get("subtitle_text", ""),
                    "static_text": annotation.get("static_text", ""),
                    "other_text": annotation.get("other_text", ""),
                }
            )
        return records

    def upsert_text_frame_annotation(
        self,
        *,
        dataset_id: str,
        video_id: str,
        frame_id: str,
        annotation: dict[str, Any],
        annotator_id: str = "default",
    ) -> None:
        document_id = f"{dataset_id}__{TEXT_TASK}__{video_id}__{frame_id}"
        self.text_collection.document(document_id).set(
            {
                "dataset_id": dataset_id,
                "task": TEXT_TASK,
                "video_id": video_id,
                "frame_id": frame_id,
                "annotator_id": annotator_id,
                "annotation": annotation,
                "status": annotation.get("status"),
                "annotated_at": annotation.get("annotated_at"),
                "updated_at": firestore.SERVER_TIMESTAMP,
                "synced_to_hf_at": None,
            },
            merge=True,
        )

    def load_active_text_video_claims(self, dataset_id: str) -> dict[str, dict[str, Any]]:
        now = datetime.now(timezone.utc)
        query = (
            self.claims_collection.where("dataset_id", "==", dataset_id)
            .where("task", "==", TEXT_TASK)
        )
        claims: dict[str, dict[str, Any]] = {}
        for document in query.stream():
            data = document.to_dict() or {}
            video_id = data.get("video_id")
            if isinstance(video_id, str) and _is_active_claim(data, now):
                claims[video_id] = data
        return claims

    def claim_text_video(
        self,
        *,
        dataset_id: str,
        video_id: str,
        annotator_id: str,
        session_id: str,
        ttl_minutes: int = 60,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=max(1, ttl_minutes))
        document_id = self.document_id(dataset_id, video_id, task=TEXT_TASK)
        claim_ref = self.claims_collection.document(document_id)
        transaction = self.client.transaction()

        @firestore.transactional
        def _claim(transaction: firestore.Transaction) -> dict[str, Any]:
            claim_snapshot = claim_ref.get(transaction=transaction)
            if claim_snapshot.exists:
                claim_data = claim_snapshot.to_dict() or {}
                claim_owner = str(claim_data.get("annotator_id") or "")
                claim_session = str(claim_data.get("session_id") or "")
                same_owner = claim_owner == annotator_id or claim_session == session_id
                if _is_active_claim(claim_data, now) and not same_owner:
                    return {
                        "claimed": False,
                        "reason": "already_claimed",
                        "claim": claim_data,
                    }

            claim = {
                "dataset_id": dataset_id,
                "task": TEXT_TASK,
                "video_id": video_id,
                "annotator_id": annotator_id,
                "session_id": session_id,
                "claimed_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "updated_at": firestore.SERVER_TIMESTAMP,
            }
            transaction.set(claim_ref, claim, merge=True)
            return {"claimed": True, "claim": claim}

        return _claim(transaction)

    def delete_funnel_decision(self, *, dataset_id: str, video_id: str) -> None:
        self.collection.document(self.document_id(dataset_id, video_id)).delete()
        self.claims_collection.document(self.document_id(dataset_id, video_id)).delete()
