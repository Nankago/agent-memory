from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from memory_collapse.io_utils import write_jsonl


SESSION_KEY_PATTERN = re.compile(r"^session_(\d+)$")


def convert_raw_external(benchmark: str, input_path: str | Path, output_path: str | Path) -> dict[str, str]:
    benchmark_name = benchmark.lower()
    source_path = Path(input_path)
    if benchmark_name == "longmemeval":
        rows = _convert_longmemeval(source_path)
    elif benchmark_name == "locomo":
        rows = _convert_locomo(source_path)
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    output = write_jsonl(output_path, rows)
    return {"converted_jsonl": str(output)}


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    rows.append(json.loads(stripped))
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _convert_longmemeval(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_or_jsonl(path)
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    converted: list[dict[str, Any]] = []
    for item in rows:
        question_id = str(item.get("question_id") or item.get("id"))
        answer_session_ids = {str(value) for value in item.get("answer_session_ids", [])}
        session_ids = item.get("haystack_session_ids", [])
        session_dates = item.get("haystack_dates", [])
        session_rows = item.get("haystack_sessions", [])
        context: list[dict[str, Any]] = []
        for idx, session in enumerate(session_rows):
            session_id = str(session_ids[idx]) if idx < len(session_ids) else f"{question_id}_session_{idx:04d}"
            session_date = session_dates[idx] if idx < len(session_dates) else None
            turn_support = any(bool(turn.get("has_answer")) for turn in session if isinstance(turn, dict))
            context.append(
                {
                    "text": _render_longmemeval_session(session),
                    "metadata": {
                        "session_id": session_id,
                        "session_date": session_date,
                        "is_answer_support": bool(session_id in answer_session_ids or turn_support),
                        "turn_support_count": sum(int(bool(turn.get("has_answer"))) for turn in session if isinstance(turn, dict)),
                    },
                }
            )
        converted.append(
            {
                "id": question_id,
                "question": str(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "context": context,
                "metadata": {
                    "benchmark": "longmemeval",
                    "question_type": item.get("question_type"),
                    "question_date": item.get("question_date"),
                    "answer_session_ids": sorted(answer_session_ids),
                },
            }
        )
    return converted


def _render_longmemeval_session(session: Any) -> str:
    if not isinstance(session, list):
        return str(session)
    rendered_turns: list[str] = []
    for turn in session:
        if not isinstance(turn, dict):
            rendered_turns.append(str(turn))
            continue
        role = str(turn.get("role", "unknown"))
        content = str(turn.get("content", ""))
        if turn.get("has_answer"):
            rendered_turns.append(f"{role}: {content} [answer_support]")
        else:
            rendered_turns.append(f"{role}: {content}")
    return "\n".join(rendered_turns)


def _convert_locomo(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_or_jsonl(path)
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    converted: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(rows):
        sample_id = str(sample.get("sample_id") or sample.get("id") or f"locomo_sample_{sample_index:04d}")
        conversation = sample.get("conversation", {}) or {}
        session_records = _extract_locomo_sessions(conversation)
        dia_to_session_id = _build_dia_to_session_id(session_records)
        qa_rows = sample.get("qa", []) or []
        for qa_index, qa in enumerate(qa_rows):
            question_id = str(
                qa.get("question_id")
                or qa.get("id")
                or f"{sample_id}_qa{qa_index:04d}"
            )
            evidence_dia_ids = [str(value) for value in qa.get("evidence", [])]
            evidence_sessions = {dia_to_session_id[dia_id] for dia_id in evidence_dia_ids if dia_id in dia_to_session_id}
            context = [
                {
                    "text": session_record["text"],
                    "metadata": {
                        "session_id": session_record["session_id"],
                        "session_date": session_record["session_date"],
                        "is_answer_support": bool(session_record["session_id"] in evidence_sessions),
                        "dialog_ids": session_record["dialog_ids"],
                    },
                }
                for session_record in session_records
            ]
            converted.append(
                {
                    "id": question_id,
                    "question": str(qa.get("question", "")),
                    "answer": str(qa.get("answer", "")),
                    "context": context,
                    "metadata": {
                        "benchmark": "locomo",
                        "sample_id": sample_id,
                        "category": qa.get("category"),
                        "evidence_dialog_ids": evidence_dia_ids,
                        "evidence_session_ids": sorted(evidence_sessions),
                    },
                }
            )
    return converted


def _extract_locomo_sessions(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    speaker_a = str(conversation.get("speaker_a", "speaker_a"))
    speaker_b = str(conversation.get("speaker_b", "speaker_b"))
    speaker_lookup = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
    }
    session_records: list[dict[str, Any]] = []
    for key, value in conversation.items():
        match = SESSION_KEY_PATTERN.match(str(key))
        if not match or not isinstance(value, list):
            continue
        session_num = match.group(1)
        session_id = f"session_{session_num}"
        session_date = conversation.get(f"{session_id}_date_time")
        dialog_ids: list[str] = []
        rendered_turns: list[str] = []
        for turn in value:
            if not isinstance(turn, dict):
                rendered_turns.append(str(turn))
                continue
            speaker = str(turn.get("speaker", "unknown"))
            speaker_name = speaker_lookup.get(speaker, speaker)
            dia_id = turn.get("dia_id")
            if dia_id is not None:
                dialog_ids.append(str(dia_id))
            text = str(turn.get("text", ""))
            if turn.get("blip_caption"):
                text = f"{text}\n[image_caption] {turn['blip_caption']}"
            rendered_turns.append(f"{speaker_name}: {text}")
        session_records.append(
            {
                "session_id": session_id,
                "session_num": int(session_num),
                "session_date": session_date,
                "dialog_ids": dialog_ids,
                "text": "\n".join(rendered_turns),
            }
        )
    session_records.sort(key=lambda row: row["session_num"])
    return session_records


def _build_dia_to_session_id(session_records: list[dict[str, Any]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for session_record in session_records:
        for dialog_id in session_record["dialog_ids"]:
            lookup[str(dialog_id)] = str(session_record["session_id"])
    return lookup
