from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from memory_collapse.io_utils import ensure_dir, read_jsonl, write_jsonl


DEFAULT_RETRIEVE_TOP_K = 20
DEFAULT_FINAL_TOP_K = 10
DEFAULT_BATCH_SIZE = 16
RRF_K = 60
LOCOMO_MAX_CHUNK_TOKENS = 384
LOCOMO_DEFAULT_TURN_WINDOW = 8
LOCOMO_DEFAULT_TURN_STRIDE = 4


@dataclass(frozen=True)
class ExternalCase:
    query_id: str
    prompt: str
    gold_answer: str
    benchmark: str
    metadata: dict[str, Any]
    memories: list[dict[str, Any]]


@dataclass(frozen=True)
class ScoredMemory:
    memory: dict[str, Any]
    score: float
    stage: str


@dataclass(frozen=True)
class RetrievalUnit:
    memory: dict[str, Any]
    unit_id: str
    text: str


def load_external_cases(normalized_dir: str | Path) -> list[ExternalCase]:
    root = Path(normalized_dir)
    queries = read_jsonl(root / "queries.jsonl")
    memories = read_jsonl(root / "memories.jsonl")
    memories_by_query: dict[str, list[dict[str, Any]]] = {}
    for memory in memories:
        memories_by_query.setdefault(str(memory["query_id"]), []).append(memory)

    cases: list[ExternalCase] = []
    for query in queries:
        query_id = str(query["query_id"])
        metadata = {
            key: value
            for key, value in query.items()
            if key not in {"query_id", "prompt", "gold_answer", "benchmark"}
        }
        cases.append(
            ExternalCase(
                query_id=query_id,
                prompt=str(query.get("prompt", "")),
                gold_answer=str(query.get("gold_answer", "")),
                benchmark=str(query.get("benchmark", "external")),
                metadata=metadata,
                memories=sorted(
                    memories_by_query.get(query_id, []),
                    key=lambda row: (int(row.get("position", 0)), str(row.get("memory_id", ""))),
                ),
            )
        )
    return cases


def run_external_retrieval(
    normalized_dir: str | Path,
    output_dir: str | Path,
    retriever: str = "tfidf",
    retriever_model: str | None = None,
    reranker_model: str | None = None,
    device: str = "cpu",
    retrieve_top_k: int = DEFAULT_RETRIEVE_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, str]:
    cases = load_external_cases(normalized_dir)
    if not cases:
        raise RuntimeError("No normalized external cases found. Run prepare_external first.")

    retriever_name = retriever.lower()
    retriever_backend = _build_retriever_backend(
        retriever=retriever_name,
        retriever_model=retriever_model,
        device=device,
        batch_size=batch_size,
    )
    reranker_backend = _CrossEncoderReranker(
        model_name_or_path=reranker_model,
        device=device,
        batch_size=batch_size,
    ) if reranker_model else None

    output_root = ensure_dir(output_dir)
    diagnostics_rows: list[dict[str, Any]] = []
    per_query_metrics: list[dict[str, Any]] = []

    for case in cases:
        retrieved = retriever_backend.rank(case)[: max(int(retrieve_top_k), 1)]
        reranked = reranker_backend.rerank(case, retrieved)[: max(int(final_top_k), 1)] if reranker_backend else retrieved[: max(int(final_top_k), 1)]
        answer_support_ids = _answer_support_ids(case)
        query_metrics = _query_retrieval_metrics(
            answer_support_ids=answer_support_ids,
            retrieved=retrieved,
            final_ranked=reranked,
        )
        per_query_metrics.append(query_metrics)
        diagnostics_rows.append(
            {
                "query_id": case.query_id,
                "benchmark": case.benchmark,
                "prompt": case.prompt,
                "gold_answer": case.gold_answer,
                "num_memories": len(case.memories),
                "answer_support_ids": answer_support_ids,
                "retriever": retriever_name,
                "retriever_model": retriever_model,
                "reranker_model": reranker_model,
                "retrieve_top_k": int(retrieve_top_k),
                "final_top_k": int(final_top_k),
                "retrieved": [
                    {
                        "memory_id": row.memory["memory_id"],
                        "score": round(float(row.score), 6),
                        "stage": row.stage,
                        "is_answer_support": row.memory["memory_id"] in answer_support_ids,
                        "text": row.memory.get("text", ""),
                    }
                    for row in retrieved
                ],
                "final_ranked": [
                    {
                        "memory_id": row.memory["memory_id"],
                        "score": round(float(row.score), 6),
                        "stage": row.stage,
                        "is_answer_support": row.memory["memory_id"] in answer_support_ids,
                        "text": row.memory.get("text", ""),
                    }
                    for row in reranked
                ],
                **query_metrics,
            }
        )

    diagnostics_path = write_jsonl(output_root / "retrieval_diagnostics.jsonl", diagnostics_rows)
    summary = _summarize_external_metrics(
        per_query_metrics=per_query_metrics,
        benchmark_names=sorted({case.benchmark for case in cases}),
        retriever=retriever_name,
        retriever_model=retriever_model,
        reranker_model=reranker_model,
        device=device,
        retrieve_top_k=retrieve_top_k,
        final_top_k=final_top_k,
    )
    summary_path = output_root / "retrieval_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return {
        "retrieval_diagnostics": str(diagnostics_path),
        "retrieval_summary": str(summary_path),
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _answer_support_ids(case: ExternalCase) -> list[str]:
    explicit_support_ids = [
        str(memory["memory_id"])
        for memory in case.memories
        if bool((memory.get("metadata") or {}).get("is_answer_support"))
    ]
    if explicit_support_ids:
        return explicit_support_ids
    normalized_answer = _normalize_text(case.gold_answer)
    if not normalized_answer:
        return []
    support_ids = [
        str(memory["memory_id"])
        for memory in case.memories
        if normalized_answer in _normalize_text(str(memory.get("text", "")))
    ]
    return support_ids


def _query_retrieval_metrics(
    answer_support_ids: list[str],
    retrieved: list[ScoredMemory],
    final_ranked: list[ScoredMemory],
) -> dict[str, Any]:
    support_set = set(answer_support_ids)
    retrieve_ids = [str(row.memory["memory_id"]) for row in retrieved]
    final_ids = [str(row.memory["memory_id"]) for row in final_ranked]
    if not support_set:
        return {
            "has_answer_support": False,
            "support_recall_at_retrieve_k": None,
            "support_recall_at_final_k": None,
            "support_hit_at_1": None,
            "support_mrr": None,
        }
    support_positions = [idx + 1 for idx, memory_id in enumerate(final_ids) if memory_id in support_set]
    return {
        "has_answer_support": True,
        "support_recall_at_retrieve_k": float(len(support_set & set(retrieve_ids)) / len(support_set)),
        "support_recall_at_final_k": float(len(support_set & set(final_ids)) / len(support_set)),
        "support_hit_at_1": float(final_ids[0] in support_set) if final_ids else 0.0,
        "support_mrr": float(1.0 / min(support_positions)) if support_positions else 0.0,
    }


def _summarize_external_metrics(
    per_query_metrics: list[dict[str, Any]],
    benchmark_names: list[str],
    retriever: str,
    retriever_model: str | None,
    reranker_model: str | None,
    device: str,
    retrieve_top_k: int,
    final_top_k: int,
) -> dict[str, Any]:
    supported_rows = [row for row in per_query_metrics if row["has_answer_support"]]
    return {
        "benchmarks": benchmark_names,
        "num_queries": len(per_query_metrics),
        "num_queries_with_answer_support": len(supported_rows),
        "retriever": retriever,
        "retriever_model": retriever_model,
        "reranker_model": reranker_model,
        "device": device,
        "retrieve_top_k": int(retrieve_top_k),
        "final_top_k": int(final_top_k),
        "metrics": {
            "support_recall_at_retrieve_k": _mean_metric(supported_rows, "support_recall_at_retrieve_k"),
            "support_recall_at_final_k": _mean_metric(supported_rows, "support_recall_at_final_k"),
            "support_hit_at_1": _mean_metric(supported_rows, "support_hit_at_1"),
            "support_mrr": _mean_metric(supported_rows, "support_mrr"),
        },
    }


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _build_retriever_backend(
    retriever: str,
    retriever_model: str | None,
    device: str,
    batch_size: int,
) -> "_BaseRetriever":
    if retriever == "tfidf":
        return _TfidfRetriever()
    if retriever == "dense":
        if not retriever_model:
            raise ValueError("Dense retrieval requires --retriever-model.")
        return _DenseRetriever(retriever_model, device=device, batch_size=batch_size)
    if retriever == "hybrid":
        if not retriever_model:
            raise ValueError("Hybrid retrieval requires --retriever-model.")
        return _HybridRetriever(retriever_model, device=device, batch_size=batch_size)
    raise ValueError(f"Unsupported external retriever: {retriever}")


class _BaseRetriever:
    def rank(self, case: ExternalCase) -> list[ScoredMemory]:
        raise NotImplementedError


class _TfidfRetriever(_BaseRetriever):
    def rank(self, case: ExternalCase) -> list[ScoredMemory]:
        if not case.memories:
            return []
        memory_texts = [str(memory.get("text", "")) for memory in case.memories]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        all_matrix = vectorizer.fit_transform(memory_texts + [case.prompt])
        memory_matrix = all_matrix[: len(memory_texts)]
        query_vector = all_matrix[len(memory_texts)]
        scores = memory_matrix.dot(query_vector.T).toarray().ravel()
        ranked = sorted(
            zip(case.memories, scores, strict=False),
            key=lambda item: (float(item[1]), -int(item[0].get("position", 0))),
            reverse=True,
        )
        return [
            ScoredMemory(memory=memory, score=float(score), stage="retrieve")
            for memory, score in ranked
        ]


class _DenseRetriever(_BaseRetriever):
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        transformers, torch = _load_transformer_stack()
        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = transformers.AutoModel.from_pretrained(model_name_or_path)
        self._model.to(device)
        self._model.eval()
        self._prefix_queries = "e5" in model_name_or_path.lower()

    def rank(self, case: ExternalCase) -> list[ScoredMemory]:
        if not case.memories:
            return []
        retrieval_units = _collect_retrieval_units(
            case.memories,
            tokenizer=self._tokenizer,
            use_chunking=True,
            max_tokens=LOCOMO_MAX_CHUNK_TOKENS,
        )
        passages = [unit.text for unit in retrieval_units]
        query_text = case.prompt
        if self._prefix_queries:
            query_text = f"query: {query_text}"
            passages = [f"passage: {text}" for text in passages]
        query_embedding = self._encode_texts([query_text])[0]
        memory_embeddings = self._encode_texts(passages)
        scores = np.asarray(memory_embeddings @ query_embedding, dtype=float).ravel()
        return _aggregate_unit_scores(retrieval_units, scores, stage="retrieve")

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        batches: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                tokens = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                outputs = self._model(**tokens)
                embeddings = _mean_pool(outputs.last_hidden_state, tokens["attention_mask"], self._torch)
                embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
                batches.append(embeddings.detach().cpu().numpy())
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, 1), dtype=float)


class _HybridRetriever(_BaseRetriever):
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        self._tfidf = _TfidfRetriever()
        self._dense = _DenseRetriever(model_name_or_path, device=device, batch_size=batch_size)

    def rank(self, case: ExternalCase) -> list[ScoredMemory]:
        tfidf_ranked = self._tfidf.rank(case)
        dense_ranked = self._dense.rank(case)
        if not tfidf_ranked and not dense_ranked:
            return []
        fused_scores: dict[str, float] = {}
        memory_lookup: dict[str, dict[str, Any]] = {}
        for ranked in (tfidf_ranked, dense_ranked):
            for position, row in enumerate(ranked, start=1):
                memory_id = str(row.memory["memory_id"])
                fused_scores[memory_id] = fused_scores.get(memory_id, 0.0) + 1.0 / (RRF_K + position)
                memory_lookup[memory_id] = row.memory
        ranked_ids = sorted(
            fused_scores,
            key=lambda memory_id: (fused_scores[memory_id], -int(memory_lookup[memory_id].get("position", 0))),
            reverse=True,
        )
        return [
            ScoredMemory(memory=memory_lookup[memory_id], score=float(fused_scores[memory_id]), stage="retrieve")
            for memory_id in ranked_ids
        ]


class _CrossEncoderReranker:
    def __init__(self, model_name_or_path: str, device: str, batch_size: int):
        transformers, torch = _load_transformer_stack()
        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self._model.to(device)
        self._model.eval()

    def rerank(self, case: ExternalCase, candidates: list[ScoredMemory]) -> list[ScoredMemory]:
        if not candidates:
            return []
        retrieval_units = _collect_retrieval_units(
            [row.memory for row in candidates],
            tokenizer=self._tokenizer,
            use_chunking=True,
            max_tokens=LOCOMO_MAX_CHUNK_TOKENS,
        )
        pairs = [(case.prompt, unit.text) for unit in retrieval_units]
        scores: list[float] = []
        with self._torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[start : start + self.batch_size]
                tokenized = self._tokenizer(
                    [query for query, _ in batch_pairs],
                    [memory_text for _, memory_text in batch_pairs],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
                logits = self._model(**tokenized).logits
                batch_scores = logits.squeeze(-1).detach().cpu().numpy().reshape(-1)
                scores.extend(float(score) for score in batch_scores)
        return _aggregate_unit_scores(retrieval_units, scores, stage="rerank")


def _load_transformer_stack():
    try:
        import torch
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Dense or rerank external retrieval requires `torch` and `transformers` in the runtime environment."
        ) from exc
    return transformers, torch


def _mean_pool(last_hidden_state: Any, attention_mask: Any, torch_module: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch_module.sum(last_hidden_state * mask, dim=1)
    counts = torch_module.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _collect_retrieval_units(
    memories: list[dict[str, Any]],
    tokenizer: Any | None,
    use_chunking: bool,
    max_tokens: int,
) -> list[RetrievalUnit]:
    retrieval_units: list[RetrievalUnit] = []
    for memory in memories:
        retrieval_units.extend(
            _memory_retrieval_units(
                memory,
                tokenizer=tokenizer,
                use_chunking=use_chunking,
                max_tokens=max_tokens,
            )
        )
    return retrieval_units


def _memory_retrieval_units(
    memory: dict[str, Any],
    tokenizer: Any | None,
    use_chunking: bool,
    max_tokens: int,
) -> list[RetrievalUnit]:
    if not use_chunking or str(memory.get("benchmark", "")).lower() != "locomo":
        return [
            RetrievalUnit(
                memory=memory,
                unit_id=str(memory["memory_id"]),
                text=str(memory.get("text", "")),
            )
        ]

    metadata = memory.get("metadata") or {}
    turns = metadata.get("turns") or []
    if not isinstance(turns, list) or not turns:
        return [
            RetrievalUnit(
                memory=memory,
                unit_id=str(memory["memory_id"]),
                text=str(memory.get("text", "")),
            )
        ]

    chunking = metadata.get("retrieval_chunking") or {}
    turn_window = int(chunking.get("turn_window", LOCOMO_DEFAULT_TURN_WINDOW))
    turn_stride = int(chunking.get("turn_stride", LOCOMO_DEFAULT_TURN_STRIDE))
    chunk_texts = _build_locomo_chunk_texts(
        turns=turns,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        turn_window=turn_window,
        turn_stride=turn_stride,
    )
    if not chunk_texts:
        chunk_texts = [str(memory.get("text", ""))]
    return [
        RetrievalUnit(
            memory=memory,
            unit_id=f"{memory['memory_id']}::chunk_{chunk_index:03d}",
            text=chunk_text,
        )
        for chunk_index, chunk_text in enumerate(chunk_texts)
    ]


def _build_locomo_chunk_texts(
    turns: list[dict[str, Any]],
    tokenizer: Any | None,
    max_tokens: int,
    turn_window: int,
    turn_stride: int,
) -> list[str]:
    if not turns:
        return []
    window = max(int(turn_window), 1)
    stride = max(int(turn_stride), 1)
    starts = list(range(0, len(turns), stride))
    last_start = max(len(turns) - window, 0)
    if last_start not in starts:
        starts.append(last_start)

    chunk_texts: list[str] = []
    seen_texts: set[str] = set()
    for start in starts:
        window_turns = turns[start : start + window]
        if not window_turns:
            continue
        window_text = "\n".join(str(turn.get("text", "")) for turn in window_turns if str(turn.get("text", "")).strip())
        if not window_text.strip():
            continue
        for chunk_text in _split_text_for_model(window_text, tokenizer=tokenizer, max_tokens=max_tokens):
            normalized = chunk_text.strip()
            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            chunk_texts.append(normalized)
    return chunk_texts


def _split_text_for_model(text: str, tokenizer: Any | None, max_tokens: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if tokenizer is None:
        return [normalized]
    token_ids = tokenizer(normalized, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return [normalized]
    chunks: list[str] = []
    for start in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[start : start + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks or [normalized]


def _aggregate_unit_scores(
    retrieval_units: list[RetrievalUnit],
    scores: np.ndarray | list[float],
    stage: str,
) -> list[ScoredMemory]:
    best_by_memory_id: dict[str, tuple[dict[str, Any], float]] = {}
    for unit, raw_score in zip(retrieval_units, scores, strict=False):
        score = float(raw_score)
        memory_id = str(unit.memory["memory_id"])
        current = best_by_memory_id.get(memory_id)
        if current is None or score > current[1]:
            best_by_memory_id[memory_id] = (unit.memory, score)
    ranked = sorted(
        best_by_memory_id.values(),
        key=lambda item: (item[1], -int(item[0].get("position", 0))),
        reverse=True,
    )
    return [
        ScoredMemory(memory=memory, score=score, stage=stage)
        for memory, score in ranked
    ]
