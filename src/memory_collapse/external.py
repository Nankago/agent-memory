from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_collapse.io_utils import ensure_dir, read_jsonl, write_jsonl


SUPPORTED_BENCHMARKS = {"longmemeval", "locomo"}


@dataclass(frozen=True)
class ExternalMemoryItem:
    memory_id: str
    query_id: str
    text: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class ExternalQueryItem:
    query_id: str
    prompt: str
    gold_answer: str
    meta: dict[str, Any]


class JsonlExternalAdapter:
    def __init__(self, benchmark_name: str):
        benchmark = benchmark_name.lower()
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"Unsupported benchmark: {benchmark_name}")
        self.benchmark_name = benchmark

    def adapt(self, input_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        input_rows = read_jsonl(input_path)
        output_root = ensure_dir(output_dir)
        memories: list[dict[str, Any]] = []
        queries: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []

        for row_index, row in enumerate(input_rows):
            query_id = str(row.get("query_id") or row.get("id") or f"{self.benchmark_name}_{row_index:05d}")
            prompt = str(row.get("question") or row.get("query") or row.get("prompt") or "")
            gold_answer = str(row.get("answer") or row.get("gold_answer") or row.get("target") or "")
            context = row.get("context") or row.get("memories") or []

            normalized_query = ExternalQueryItem(
                query_id=query_id,
                prompt=prompt,
                gold_answer=gold_answer,
                meta={
                    "benchmark": self.benchmark_name,
                    "source_row": row_index,
                    "metadata": row.get("metadata", {}),
                },
            )
            queries.append(
                {
                    "query_id": normalized_query.query_id,
                    "prompt": normalized_query.prompt,
                    "gold_answer": normalized_query.gold_answer,
                    **normalized_query.meta,
                }
            )

            normalized_context = _normalize_context(context)
            for memory_index, item in enumerate(normalized_context):
                memory = ExternalMemoryItem(
                    memory_id=f"{query_id}_m{memory_index:04d}",
                    query_id=query_id,
                    text=item["text"],
                    meta={
                        "position": memory_index,
                        "benchmark": self.benchmark_name,
                        "metadata": item.get("metadata", {}),
                    },
                )
                memories.append(
                    {
                        "memory_id": memory.memory_id,
                        "query_id": memory.query_id,
                        "text": memory.text,
                        **memory.meta,
                    }
                )
            manifest_rows.append(
                {
                    "query_id": query_id,
                    "num_memories": len(normalized_context),
                }
            )

        queries_path = write_jsonl(output_root / "queries.jsonl", queries)
        memories_path = write_jsonl(output_root / "memories.jsonl", memories)
        manifest_path = write_jsonl(output_root / "manifest.jsonl", manifest_rows)
        summary_path = output_root / "adapter_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "benchmark": self.benchmark_name,
                    "num_queries": len(queries),
                    "num_memories": len(memories),
                },
                handle,
                indent=2,
            )
        return {
            "queries": str(queries_path),
            "memories": str(memories_path),
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        }


def _normalize_context(context: Any) -> list[dict[str, Any]]:
    if isinstance(context, list):
        items = context
    elif isinstance(context, str):
        items = [line.strip() for line in context.splitlines() if line.strip()]
    else:
        items = []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"text": item, "metadata": {}})
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("memory") or item.get("content") or "")
            normalized.append({"text": text, "metadata": item.get("metadata", {})})
    return normalized
