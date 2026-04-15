from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from memory_collapse.anti_support import anti_support_model_exists, load_anti_support_bundle
from memory_collapse.baselines import (
    DIRECT_VALID_METHOD,
    DIRECT_VALID_RESOLVER_METHOD,
    _aggregate_prediction,
    _softmax_confidence,
)
from memory_collapse.estimators import build_query_memory_contexts
from memory_collapse.io_utils import ensure_dir, read_jsonl, write_csv, write_jsonl
from memory_collapse.query_validity import query_validity_model_exists, load_query_validity_bundle
from memory_collapse.relevance import load_relevance_bundle, normalized_similarity, relevance_model_exists
from memory_collapse.value_resolver import (
    build_value_candidate_feature_rows,
    load_value_resolver_bundle,
    value_resolver_model_exists,
)


RETRIEVAL_ONLY_BASELINE = "retrieval_only_baseline"
EXTERNAL_METHODS = (
    RETRIEVAL_ONLY_BASELINE,
    DIRECT_VALID_METHOD,
    DIRECT_VALID_RESOLVER_METHOD,
)
DEFAULT_EXTERNAL_METHODS = list(EXTERNAL_METHODS)
DEFAULT_SUMMARY_ROOT = Path("outputs") / "external_runs"
DEFAULT_STAGE_NAME = "final_ranked"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "user",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class ExternalRetrievedCase:
    benchmark: str
    retrieval_variant: str
    query_id: str
    prompt: str
    gold_answer: str
    answer_support_ids: list[str]
    retrieve_top_k: int | None
    final_top_k: int | None
    retrieved: list[dict[str, Any]]
    final_ranked: list[dict[str, Any]]
    raw_row: dict[str, Any]


@dataclass(frozen=True)
class ExternalMethodBundles:
    relevance_bundle: Any
    query_validity_bundles: dict[str, Any]
    anti_support_bundle: Any
    value_resolver_bundle: Any


class ExternalEndToEndRunner:
    def __init__(
        self,
        benchmark_name: str,
        output_root: str | Path,
        model_dir: str | Path | None = None,
        top_k: int | None = None,
        final_k: int | None = None,
        summary_root: str | Path | None = None,
    ):
        self.benchmark_name = benchmark_name.lower()
        self.output_root = ensure_dir(output_root)
        self.model_dir = Path(model_dir) if model_dir else None
        self.top_k = int(top_k) if top_k else None
        self.final_k = int(final_k) if final_k else None
        self.summary_root = ensure_dir(summary_root or DEFAULT_SUMMARY_ROOT)
        self._bundles: ExternalMethodBundles | None = None

    def run(
        self,
        retrieval_inputs: dict[str, str | Path],
        methods: list[str],
    ) -> dict[str, str]:
        methods = [method.strip() for method in methods if method.strip()]
        if not methods:
            methods = list(DEFAULT_EXTERNAL_METHODS)
        invalid_methods = sorted(set(methods) - set(EXTERNAL_METHODS))
        if invalid_methods:
            raise ValueError(f"Unsupported external methods: {', '.join(invalid_methods)}")
        if any(method != RETRIEVAL_ONLY_BASELINE for method in methods) and self.model_dir is None:
            raise ValueError("Proposed external methods require --model-dir with the trained synthetic artifacts.")

        all_metric_rows: list[dict[str, Any]] = []
        written_diagnostics: dict[str, str] = {}
        for retrieval_variant, input_dir in retrieval_inputs.items():
            cases = _load_retrieval_cases(input_dir, self.benchmark_name, retrieval_variant)
            for method in methods:
                diagnostics_rows = [self._run_case(case, method) for case in cases]
                metrics_row = _summarize_method_run(
                    benchmark=self.benchmark_name,
                    retrieval_variant=retrieval_variant,
                    method=method,
                    diagnostics_rows=diagnostics_rows,
                )
                all_metric_rows.append(metrics_row)

                method_dir = ensure_dir(self.output_root / self.benchmark_name / retrieval_variant / method)
                diagnostics_path = write_jsonl(method_dir / "prediction_diagnostics.jsonl", diagnostics_rows)
                write_csv(method_dir / "metrics.csv", [metrics_row])
                with (method_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump(metrics_row, handle, indent=2)
                written_diagnostics[f"{retrieval_variant}:{method}"] = str(diagnostics_path)

        metrics_csv_path = write_csv(self.summary_root / "external_end_to_end_metrics.csv", all_metric_rows)
        summary_rows = [_summary_projection(row) for row in all_metric_rows]
        summary_csv_path = write_csv(self.summary_root / "external_end_to_end_summary.csv", summary_rows)
        summary_md_path = self.summary_root / "external_end_to_end_summary.md"
        summary_md_path.write_text(_render_summary_markdown(summary_rows), encoding="utf-8")
        return {
            "metrics_csv": str(metrics_csv_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            "diagnostics": written_diagnostics,
        }

    def _run_case(self, case: ExternalRetrievedCase, method: str) -> dict[str, Any]:
        candidate_rows, candidate_source = _select_candidate_rows(case, top_k=self.top_k, final_k=self.final_k)
        pseudo_query, pseudo_memories = _build_external_method_inputs(case, candidate_rows, candidate_source)
        tfidf_lookup = _external_tfidf_lookup(pseudo_query, pseudo_memories)
        query_memory_contexts = build_query_memory_contexts(pseudo_memories, pseudo_query)

        if method == RETRIEVAL_ONLY_BASELINE:
            predicted_answer, value_scores, direct_valid_metadata, resolver_metadata = _run_retrieval_only(
                candidate_rows,
                pseudo_memories,
                candidate_source,
            )
        else:
            bundles = self._get_bundles()
            predicted_answer, value_scores, component_debug = _aggregate_prediction(
                method,
                pseudo_query,
                pseudo_memories,
                estimator_bundle=None,
                relevance_bundle=bundles.relevance_bundle,
                query_validity_bundles=bundles.query_validity_bundles,
                anti_support_bundle=bundles.anti_support_bundle,
                value_resolver_bundle=bundles.value_resolver_bundle,
                controller_calibration=None,
                tfidf_lookup=tfidf_lookup,
                query_memory_contexts=query_memory_contexts,
                component_cache={},
            )
            direct_valid_metadata = {
                "pool_source": candidate_source,
                "num_candidate_contexts": len(pseudo_memories),
                "component_debug": component_debug,
            }
            resolver_metadata = None
            if method == DIRECT_VALID_RESOLVER_METHOD:
                resolver_metadata = _build_resolver_metadata(
                    query=pseudo_query,
                    fallback_pool=pseudo_memories,
                    component_debug=component_debug,
                    value_resolver_bundle=bundles.value_resolver_bundle,
                )

        value_to_ids = _value_to_context_ids(pseudo_memories)
        chosen_ids = value_to_ids.get(predicted_answer, [])
        exact_match = _normalized_answer(predicted_answer) == _normalized_answer(case.gold_answer)
        is_correct = _answer_contains_gold(predicted_answer, case.gold_answer)
        retrieval_support_final = case.raw_row.get("support_recall_at_final_k")
        retrieval_support_retrieve = case.raw_row.get("support_recall_at_retrieve_k")
        retrieval_hit_at_1 = case.raw_row.get("support_hit_at_1")
        retrieval_mrr = case.raw_row.get("support_mrr")
        return {
            "benchmark": case.benchmark,
            "variant": case.retrieval_variant,
            "method": method,
            "query_id": case.query_id,
            "gold_answer": case.gold_answer,
            "predicted_answer": predicted_answer,
            "is_correct": bool(is_correct),
            "exact_match": bool(exact_match),
            "retrieved_ids": [str(row["memory_id"]) for row in candidate_rows],
            "candidate_value_scores": _serialize_value_scores(value_scores, value_to_ids),
            "chosen_value": predicted_answer,
            "confidence": float(_softmax_confidence(value_scores)),
            "supporting_context_ids": chosen_ids,
            "answer_support_ids": case.answer_support_ids,
            "candidate_pool_source": candidate_source,
            "num_candidate_contexts": len(candidate_rows),
            "retrieval_support_recall_at_retrieve_k": retrieval_support_retrieve,
            "retrieval_support_recall_at_final_k": retrieval_support_final,
            "hit_at_1": retrieval_hit_at_1,
            "mrr": retrieval_mrr,
            "resolver_metadata": resolver_metadata,
            "direct_valid_metadata": direct_valid_metadata,
        }

    def _get_bundles(self) -> ExternalMethodBundles:
        if self._bundles is None:
            assert self.model_dir is not None
            self._bundles = _load_external_method_bundles(self.model_dir)
        return self._bundles


def run_external_end_to_end(
    benchmark_name: str,
    retrieval_inputs: dict[str, str | Path],
    methods: list[str] | None,
    output_root: str | Path,
    model_dir: str | Path | None = None,
    top_k: int | None = None,
    final_k: int | None = None,
    summary_root: str | Path | None = None,
) -> dict[str, str]:
    runner = ExternalEndToEndRunner(
        benchmark_name=benchmark_name,
        output_root=output_root,
        model_dir=model_dir,
        top_k=top_k,
        final_k=final_k,
        summary_root=summary_root,
    )
    return runner.run(retrieval_inputs=retrieval_inputs, methods=methods or list(DEFAULT_EXTERNAL_METHODS))


def _load_external_method_bundles(model_dir: str | Path) -> ExternalMethodBundles:
    if not relevance_model_exists(model_dir):
        raise FileNotFoundError(f"Missing relevance model under {model_dir}.")
    if not query_validity_model_exists(model_dir):
        raise FileNotFoundError(f"Missing query-validity models under {model_dir}.")
    if not anti_support_model_exists(model_dir):
        raise FileNotFoundError(f"Missing anti-support model under {model_dir}.")
    if not value_resolver_model_exists(model_dir):
        raise FileNotFoundError(f"Missing value-resolver model under {model_dir}.")
    return ExternalMethodBundles(
        relevance_bundle=load_relevance_bundle(model_dir),
        query_validity_bundles={
            "useful_label": load_query_validity_bundle(model_dir, target_label="useful_label"),
            "valid_label": load_query_validity_bundle(model_dir, target_label="valid_label"),
        },
        anti_support_bundle=load_anti_support_bundle(model_dir),
        value_resolver_bundle=load_value_resolver_bundle(model_dir),
    )


def _load_retrieval_cases(
    input_dir: str | Path,
    benchmark_name: str,
    retrieval_variant: str,
) -> list[ExternalRetrievedCase]:
    diagnostics_path = Path(input_dir) / "retrieval_diagnostics.jsonl"
    rows = read_jsonl(diagnostics_path)
    if not rows:
        raise FileNotFoundError(f"No retrieval diagnostics found at {diagnostics_path}.")
    cases: list[ExternalRetrievedCase] = []
    for row in rows:
        row_benchmark = str(row.get("benchmark") or benchmark_name).lower()
        if row_benchmark != benchmark_name.lower():
            continue
        cases.append(
            ExternalRetrievedCase(
                benchmark=row_benchmark,
                retrieval_variant=retrieval_variant,
                query_id=str(row["query_id"]),
                prompt=str(row.get("prompt", "")),
                gold_answer=str(row.get("gold_answer", "")),
                answer_support_ids=[str(item) for item in row.get("answer_support_ids", [])],
                retrieve_top_k=_maybe_int(row.get("retrieve_top_k")),
                final_top_k=_maybe_int(row.get("final_top_k")),
                retrieved=list(row.get("retrieved", [])),
                final_ranked=list(row.get("final_ranked", [])),
                raw_row=row,
            )
        )
    if not cases:
        raise RuntimeError(f"No retrieval cases for benchmark={benchmark_name} found under {input_dir}.")
    return cases


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _select_candidate_rows(
    case: ExternalRetrievedCase,
    top_k: int | None,
    final_k: int | None,
) -> tuple[list[dict[str, Any]], str]:
    final_limit = final_k or case.final_top_k or top_k or case.retrieve_top_k or len(case.final_ranked) or len(case.retrieved)
    retrieve_limit = top_k or case.retrieve_top_k or len(case.retrieved) or len(case.final_ranked)
    if case.final_ranked:
        return list(case.final_ranked[: max(int(final_limit), 1)]), DEFAULT_STAGE_NAME
    return list(case.retrieved[: max(int(retrieve_limit), 1)]), "retrieved"


def _build_external_method_inputs(
    case: ExternalRetrievedCase,
    candidate_rows: list[dict[str, Any]],
    candidate_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subject_hint = _derive_subject_hint(case.prompt, case.query_id)
    pseudo_query = {
        "query_id": case.query_id,
        "episode_id": f"{case.benchmark}:{case.query_id}",
        "query_text": case.prompt,
        "entity": subject_hint,
        "slot": "answer",
        "query_time": 1,
        "stress_name": "external",
        "stress_value": 0.0,
        "query_lag": 0,
        "world_change_scale": 0.0,
        "write_error_rate": 0.0,
        "conflict_rate": 0.0,
        "distractor_overlap": 0.0,
        "gold_value": case.gold_answer,
        "benchmark": case.benchmark,
        "retrieval_variant": case.retrieval_variant,
        "candidate_pool_source": candidate_source,
    }

    pseudo_memories: list[dict[str, Any]] = []
    for row in candidate_rows:
        text = str(row.get("text", ""))
        candidate_value = _extract_candidate_value(case.prompt, text)
        pseudo_memories.append(
            {
                "memory_id": str(row["memory_id"]),
                "episode_id": f"{case.benchmark}:{case.query_id}",
                "memory_text": text,
                "entity": subject_hint,
                "entity_alias": subject_hint,
                "slot": "answer",
                "value_raw": candidate_value,
                "value_canonical": candidate_value,
                "write_time": 0,
                "source_id": str(row["memory_id"]),
                "source_quality": 1.0,
                "stress_name": "external",
                "stress_value": 0.0,
                "write_correct": True,
            }
        )
    return pseudo_query, pseudo_memories


def _external_tfidf_lookup(query: dict[str, Any], memories: list[dict[str, Any]]) -> dict[str, float]:
    if not memories:
        return {}
    corpus = [memory["memory_text"] for memory in memories] + [query["query_text"]]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return {memory["memory_id"]: 0.5 for memory in memories}
    memory_matrix = matrix[: len(memories)]
    query_vector = matrix[len(memories)]
    raw_scores = memory_matrix.dot(query_vector.T).toarray().ravel()
    return normalized_similarity(
        {
            memory["memory_id"]: float(score)
            for memory, score in zip(memories, raw_scores, strict=False)
        }
    )


def _run_retrieval_only(
    candidate_rows: list[dict[str, Any]],
    pseudo_memories: list[dict[str, Any]],
    candidate_source: str,
) -> tuple[str | None, dict[str, float], dict[str, Any], dict[str, Any] | None]:
    if not pseudo_memories:
        return None, {}, {"pool_source": "empty", "num_candidate_contexts": 0}, None
    memory_by_id = {memory["memory_id"]: memory for memory in pseudo_memories}
    raw_scores = {
        str(row["memory_id"]): float(row.get("score", 0.0))
        for row in candidate_rows
    }
    normalized_scores = normalized_similarity(raw_scores) if raw_scores else {}
    value_scores: dict[str, float] = {}
    for row in candidate_rows:
        memory_id = str(row["memory_id"])
        memory = memory_by_id[memory_id]
        value = memory["value_canonical"]
        value_scores[value] = max(value_scores.get(value, float("-inf")), normalized_scores.get(memory_id, 0.0))
    prediction = max(value_scores.items(), key=lambda item: (item[1], item[0]))[0]
    return (
        prediction,
        value_scores,
        {
            "pool_source": candidate_source,
            "num_candidate_contexts": len(candidate_rows),
            "retrieval_scores": {
                memory_id: round(score, 6)
                for memory_id, score in normalized_scores.items()
            },
        },
        None,
    )


def _build_resolver_metadata(
    query: dict[str, Any],
    fallback_pool: list[dict[str, Any]],
    component_debug: dict[str, dict[str, Any]],
    value_resolver_bundle: Any,
) -> dict[str, Any]:
    if not fallback_pool:
        return {
            "used": False,
            "condition": getattr(value_resolver_bundle, "condition", None),
            "objective": getattr(value_resolver_bundle, "objective", None),
        }
    candidate_rows = build_value_candidate_feature_rows(query, fallback_pool, component_debug)
    candidate_features = [
        {
            key: value
            for key, value in row.items()
            if key not in {"query_id", "episode_id", "candidate_value"}
        }
        for row in candidate_rows
    ]
    candidate_scores = value_resolver_bundle.predict_candidate_scores(candidate_features)
    return {
        "used": True,
        "condition": getattr(value_resolver_bundle, "condition", None),
        "objective": getattr(value_resolver_bundle, "objective", None),
        "candidate_rows": candidate_rows,
        "candidate_scores": [
            {
                "candidate_value": row["candidate_value"],
                "score": round(float(score), 6),
            }
            for row, score in zip(candidate_rows, candidate_scores, strict=False)
        ],
    }


def _serialize_value_scores(
    value_scores: dict[str, float],
    value_to_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_value": value,
            "score": round(float(score), 6),
            "supporting_context_ids": value_to_ids.get(value, []),
        }
        for value, score in sorted(value_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    ]


def _value_to_context_ids(memories: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for memory in memories:
        mapping.setdefault(str(memory["value_canonical"]), []).append(str(memory["memory_id"]))
    return mapping


def _normalized_answer(text: str | None) -> str:
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"[\W_]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _answer_contains_gold(predicted_answer: str | None, gold_answer: str | None) -> bool:
    predicted = _normalized_answer(predicted_answer)
    gold = _normalized_answer(gold_answer)
    if not predicted or not gold:
        return False
    if predicted == gold:
        return True
    return gold in predicted


def _derive_subject_hint(prompt: str, fallback: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9']+", prompt.lower()) if token not in STOPWORDS]
    if not tokens:
        return fallback
    return " ".join(tokens[:6])


def _extract_candidate_value(prompt: str, text: str) -> str:
    segments = _split_segments(text)
    if not segments:
        return text.strip()
    query_tokens = set(re.findall(r"[A-Za-z0-9']+", prompt.lower()))
    best_segment = max(
        segments,
        key=lambda segment: (_segment_score(segment, query_tokens), -len(segment)),
    )
    snippet = _segment_to_answer_snippet(best_segment, query_tokens)
    if not snippet:
        return best_segment.strip()
    snippet_tokens = re.findall(r"[A-Za-z0-9']+", snippet)
    if len(snippet_tokens) <= 2:
        return best_segment.strip()
    return snippet


def _split_segments(text: str) -> list[str]:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    segments: list[str] = []
    for line in lines:
        line = re.sub(r"^[A-Za-z0-9_ ]{1,20}:\s*", "", line).strip()
        for part in re.split(r"(?<=[\.\!\?])\s+|;\s+", line):
            cleaned = part.strip(" -\t")
            if cleaned:
                segments.append(cleaned)
    return segments


def _segment_score(segment: str, query_tokens: set[str]) -> float:
    tokens = re.findall(r"[A-Za-z0-9']+", segment)
    lowered = [token.lower() for token in tokens]
    overlap = sum(1 for token in lowered if token in query_tokens)
    novelty = sum(1 for token in lowered if token not in STOPWORDS)
    proper_bonus = sum(1 for token in tokens if any(char.isdigit() for char in token) or token[:1].isupper())
    return 1.8 * overlap + 0.8 * novelty + 0.5 * proper_bonus - 0.03 * len(tokens)


def _segment_to_answer_snippet(segment: str, query_tokens: set[str]) -> str:
    tokens = re.findall(r"[A-Za-z0-9']+", segment)
    if not tokens:
        return segment.strip()
    chunks: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        lowered = token.lower()
        keep = (
            lowered not in STOPWORDS
            or lowered in query_tokens
            or any(char.isdigit() for char in token)
            or token[:1].isupper()
        )
        if keep:
            current.append(token)
        elif current:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    if not chunks:
        return segment.strip()

    def _chunk_score(chunk: list[str]) -> float:
        lowered = [token.lower() for token in chunk]
        overlap = sum(1 for token in lowered if token in query_tokens)
        novelty = sum(1 for token in lowered if token not in STOPWORDS)
        proper_bonus = sum(1 for token in chunk if any(char.isdigit() for char in token) or token[:1].isupper())
        return 1.5 * overlap + 1.0 * novelty + 0.4 * proper_bonus - 0.15 * max(len(chunk) - 4, 0)

    best_chunk = max(chunks, key=_chunk_score)
    return " ".join(best_chunk[:8]).strip()


def _summarize_method_run(
    benchmark: str,
    retrieval_variant: str,
    method: str,
    diagnostics_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    num_queries = len(diagnostics_rows)
    return {
        "benchmark": benchmark,
        "retrieval_variant": retrieval_variant,
        "method": method,
        "num_queries": num_queries,
        "accuracy": _mean_bool(diagnostics_rows, "is_correct"),
        "exact_match": _mean_bool(diagnostics_rows, "exact_match"),
        "hit_at_1": _mean_value(diagnostics_rows, "hit_at_1"),
        "mrr": _mean_value(diagnostics_rows, "mrr"),
        "retrieval_support_recall_at_retrieve_k": _mean_value(diagnostics_rows, "retrieval_support_recall_at_retrieve_k"),
        "retrieval_support_recall_at_final_k": _mean_value(diagnostics_rows, "retrieval_support_recall_at_final_k"),
        "notes": _summary_note(method),
    }


def _summary_note(method: str) -> str:
    if method == RETRIEVAL_ONLY_BASELINE:
        return "Top retrieved context with local extractive answer snippet."
    if method == DIRECT_VALID_METHOD:
        return "Reuse synthetic direct_valid scorer on external retrieved candidates."
    if method == DIRECT_VALID_RESOLVER_METHOD:
        return "Reuse synthetic direct_valid plus resolver on external retrieved candidates."
    return ""


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    values = [1.0 if bool(row.get(key)) else 0.0 for row in rows]
    return float(np.mean(values))


def _mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _summary_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": row["benchmark"],
        "retrieval_variant": row["retrieval_variant"],
        "method": row["method"],
        "num_queries": row["num_queries"],
        "accuracy": row["accuracy"],
        "exact_match": row["exact_match"],
        "hit_at_1": row["hit_at_1"],
        "mrr": row["mrr"],
        "retrieval_support_recall_at_retrieve_k": row["retrieval_support_recall_at_retrieve_k"],
        "retrieval_support_recall_at_final_k": row["retrieval_support_recall_at_final_k"],
        "notes": row["notes"],
    }


def _render_summary_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "benchmark",
        "retrieval_variant",
        "method",
        "num_queries",
        "accuracy",
        "exact_match",
        "hit_at_1",
        "mrr",
        "retrieval_support_recall_at_retrieve_k",
        "retrieval_support_recall_at_final_k",
        "notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            _format_md_value(row.get(header))
            for header in headers
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _format_md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
