from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd
import streamlit as st

from annotation_app.common.auth import require_login
from annotation_app.common.firestore_decision_store import FirestoreDecisionStore
from annotation_app.common.hf_dataset_store import DATASET_ID


FUNNEL_CATEGORY_LABELS = {
    "no_usable_speech": "Без корисного мовлення",
    "speech_no_text": "Мовлення без тексту",
    "text_without_subtitles": "Текст без субтитрів",
    "insufficient_subtitle_alignment": "Недостатній збіг субтитрів",
    "partially_matched": "Субтитри частково збігаються",
    "title_matched": "Субтитри + додатковий текст",
    "matched": "Субтитри збігаються",
    "problem": "Проблема",
    # legacy labels for historical Firestore records
    "unmatched": "Legacy: Не збігається",
    "ignore": "Legacy: Ігнорувати",
    "annotation_problem": "Legacy: Проблема розмітки",
    "partly_matched": "Legacy: Частково збігаються",
}


def percent(value: int, total: int) -> float:
    return round((value / total) * 100, 1) if total else 0.0


def display_counter_table(counter: Counter[str], total: int, *, label_column: str) -> None:
    rows = [
        {
            label_column: key,
            "count": count,
            "%": percent(count, total),
        }
        for key, count in counter.most_common()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def funnel_by_user(records: list[dict[str, Any]]) -> pd.DataFrame:
    user_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        user = str(record.get("annotator_id") or "unknown")
        category = str(record.get("category") or "unknown")
        user_category_counts[user][category] += 1

    rows: list[dict[str, Any]] = []
    for user, counter in sorted(user_category_counts.items()):
        total = sum(counter.values())
        row: dict[str, Any] = {"annotator": user, "total": total}
        for category_id, label in FUNNEL_CATEGORY_LABELS.items():
            count = counter.get(category_id, 0)
            row[label] = count
            row[f"{label} %"] = percent(count, total)
        rows.append(row)
    return pd.DataFrame(rows)


def text_by_user(records: list[dict[str, Any]]) -> pd.DataFrame:
    user_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        user_records[str(record.get("annotator_id") or "unknown")].append(record)

    rows: list[dict[str, Any]] = []
    for user, items in sorted(user_records.items()):
        status_counts = Counter(str(item.get("status") or "unknown") for item in items)
        unique_videos = {item.get("video_id") for item in items if item.get("video_id")}
        rows.append(
            {
                "annotator": user,
                "frames": len(items),
                "videos": len(unique_videos),
                "accepted": status_counts.get("accepted", 0),
                "empty": status_counts.get("empty", 0),
                "problem": status_counts.get("problem", 0),
                "accepted %": percent(status_counts.get("accepted", 0), len(items)),
                "with subtitles": sum(1 for item in items if str(item.get("subtitle_text") or "").strip()),
                "with static text": sum(1 for item in items if str(item.get("static_text") or "").strip()),
                "with other": sum(1 for item in items if str(item.get("other_text") or "").strip()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    active_user = require_login(form_key="stats_login_form")
    if active_user is None:
        st.stop()

    st.header("Статистика")
    st.caption("Точка правди для цієї сторінки: GCP Firestore.")

    store = FirestoreDecisionStore.from_config()
    if st.button("Оновити статистику", type="primary"):
        st.cache_data.clear()
        st.rerun()

    funnel_records = store.load_funnel_decision_records(DATASET_ID)
    text_records = store.load_text_frame_annotation_records(DATASET_ID)

    st.subheader("Класифікація відео")
    funnel_total = len(funnel_records)
    category_counter = Counter(str(record.get("category") or "unknown") for record in funnel_records)
    c1, c2, c3 = st.columns(3)
    c1.metric("Усього рішень", funnel_total)
    c2.metric("Анотатори", len({record.get("annotator_id") for record in funnel_records}))
    c3.metric("Matched + static", category_counter.get("matched", 0) + category_counter.get("title_matched", 0))

    if funnel_total:
        display_counter_table(
            Counter(FUNNEL_CATEGORY_LABELS.get(key, key) for key in category_counter.elements()),
            funnel_total,
            label_column="category",
        )
        st.caption("По користувачах")
        st.dataframe(funnel_by_user(funnel_records), hide_index=True, use_container_width=True)
    else:
        st.info("У Firestore ще немає funnel-рішень.")

    st.subheader("Виправлення тексту")
    text_total = len(text_records)
    status_counter = Counter(str(record.get("status") or "unknown") for record in text_records)
    unique_text_videos = {record.get("video_id") for record in text_records if record.get("video_id")}
    t1, t2, t3 = st.columns(3)
    t1.metric("Розмічені кадри", text_total)
    t2.metric("Відео з розміткою", len(unique_text_videos))
    t3.metric("Анотатори", len({record.get("annotator_id") for record in text_records}))

    if text_total:
        display_counter_table(status_counter, text_total, label_column="status")
        st.caption("По користувачах")
        st.dataframe(text_by_user(text_records), hide_index=True, use_container_width=True)
    else:
        st.info("У Firestore ще немає виправлень тексту.")


if __name__ == "__main__":
    main()
