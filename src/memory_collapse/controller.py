from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from memory_collapse.domain import list_slot_specs
from memory_collapse.estimators import (
    LearnedEstimatorBundle,
    build_query_memory_contexts,
    load_estimator_bundle,
    load_episode_splits,
    resolve_models_dir,
)
from memory_collapse.io_utils import ensure_dir, read_jsonl
from memory_collapse.relevance import (
    LearnedRelevanceBundle,
    load_relevance_bundle,
    normalized_similarity,
    rule_relevance_score,
)


CONTROLLER_CALIBRATION_FILENAME = 'controller_calibration.json'


@dataclass
class EpisodeIndex:
    episode_id: str
    memories: list[dict[str, Any]]
    memory_by_id: dict[str, dict[str, Any]]
    vectorizer: TfidfVectorizer
    memory_matrix: Any
    memory_positions: dict[str, int]


@dataclass
class QueryRecord:
    query: dict[str, Any]
    candidates: list[dict[str, Any]]


def controller_artifact_paths(path: str | Path) -> dict[str, str]:
    models_dir = resolve_models_dir(path)
    return {
        'controller_calibration': str(models_dir / CONTROLLER_CALIBRATION_FILENAME),
    }


def controller_calibration_exists(path: str | Path) -> bool:
    return Path(controller_artifact_paths(path)['controller_calibration']).exists()


def load_controller_calibration(path: str | Path) -> dict[str, Any]:
    models_dir = resolve_models_dir(path)
    calibration_path = models_dir / CONTROLLER_CALIBRATION_FILENAME
    if not calibration_path.exists():
        return default_controller_calibration()
    with calibration_path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def default_controller_calibration() -> dict[str, Any]:
    return {
        'split': 'val',
        'selection_metric': 'accuracy',
        'rho_mode': 'learned',
        'c_blend': 1.0,
        's_blend': 1.0,
        'val_accuracy': None,
        'val_collapse_rate': None,
        'grid': [],
    }


def train_controller_calibration(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    run_path = Path(run_dir)
    models_dir = ensure_dir(resolve_models_dir(output_dir or run_path))
    estimator_bundle = load_estimator_bundle(models_dir)
    relevance_bundle = load_relevance_bundle(models_dir)
    episode_splits = load_episode_splits(models_dir)

    memories = read_jsonl(run_path / 'data' / 'memories.jsonl')
    queries = read_jsonl(run_path / 'data' / 'queries.jsonl')
    if not memories or not queries:
        raise RuntimeError('Controller calibration requires generated synthetic artifacts.')

    config_snapshot = yaml.safe_load((run_path / 'config_snapshot.yaml').read_text(encoding='utf-8')) or {}
    top_k = int(config_snapshot.get('dataset', {}).get('top_k', 12))
    val_episodes = set(episode_splits.get('val', []))
    calibration = default_controller_calibration()
    if not val_episodes:
        calibration_path = models_dir / CONTROLLER_CALIBRATION_FILENAME
        with calibration_path.open('w', encoding='utf-8') as handle:
            json.dump(calibration, handle, indent=2)
        return controller_artifact_paths(models_dir)

    val_queries = [query for query in queries if query['episode_id'] in val_episodes]
    episode_indices = _build_episode_indices(memories, queries)
    query_records = [
        _prepare_query_record(
            index=episode_indices[query['episode_id']],
            query=query,
            estimator_bundle=estimator_bundle,
            relevance_bundle=relevance_bundle,
        )
        for query in val_queries
    ]

    candidate_settings = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid_results: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    for c_blend in candidate_settings:
        for s_blend in candidate_settings:
            correct = 0
            total = 0
            for record in query_records:
                predicted = _predict_from_record(record, c_blend=float(c_blend), s_blend=float(s_blend), top_k=top_k)
                correct += int(predicted == record.query['gold_value'])
                total += 1
            accuracy = correct / max(total, 1)
            collapse_rate = 1.0 - accuracy
            current = {
                'c_blend': float(c_blend),
                's_blend': float(s_blend),
                'num_queries': total,
                'accuracy': float(accuracy),
                'collapse_rate': float(collapse_rate),
            }
            grid_results.append(current)
            current_key = (accuracy, -collapse_rate, -abs(c_blend - 0.5) - abs(s_blend - 0.5))
            best_key = (
                best_record['accuracy'],
                -best_record['collapse_rate'],
                -abs(best_record['c_blend'] - 0.5) - abs(best_record['s_blend'] - 0.5),
            ) if best_record else None
            if best_record is None or current_key > best_key:
                best_record = current

    calibration = {
        'split': 'val',
        'selection_metric': 'accuracy',
        'rho_mode': 'learned',
        'c_blend': float(best_record['c_blend']),
        's_blend': float(best_record['s_blend']),
        'val_accuracy': float(best_record['accuracy']),
        'val_collapse_rate': float(best_record['collapse_rate']),
        'grid': grid_results,
    }
    calibration_path = models_dir / CONTROLLER_CALIBRATION_FILENAME
    with calibration_path.open('w', encoding='utf-8') as handle:
        json.dump(calibration, handle, indent=2)
    return controller_artifact_paths(models_dir)


def _build_episode_indices(memories: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, EpisodeIndex]:
    grouped_memories: dict[str, list[dict[str, Any]]] = {}
    grouped_queries: dict[str, list[dict[str, Any]]] = {}
    for memory in memories:
        grouped_memories.setdefault(memory['episode_id'], []).append(memory)
    for query in queries:
        grouped_queries.setdefault(query['episode_id'], []).append(query)

    episode_indices: dict[str, EpisodeIndex] = {}
    for episode_id, memory_rows in grouped_memories.items():
        memory_rows = sorted(memory_rows, key=lambda row: row['memory_id'])
        query_rows = grouped_queries.get(episode_id, [])
        corpus = [memory['memory_text'] for memory in memory_rows] + [query['query_text'] for query in query_rows]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        all_matrix = vectorizer.fit_transform(corpus)
        memory_matrix = all_matrix[: len(memory_rows)]
        episode_indices[episode_id] = EpisodeIndex(
            episode_id=episode_id,
            memories=memory_rows,
            memory_by_id={memory['memory_id']: memory for memory in memory_rows},
            vectorizer=vectorizer,
            memory_matrix=memory_matrix,
            memory_positions={memory['memory_id']: idx for idx, memory in enumerate(memory_rows)},
        )
    return episode_indices


def _tfidf_similarity(index: EpisodeIndex, query: dict[str, Any], available_memories: list[dict[str, Any]]) -> dict[str, float]:
    if not available_memories:
        return {}
    query_vector = index.vectorizer.transform([query['query_text']])
    positions = [index.memory_positions[memory['memory_id']] for memory in available_memories]
    sims = index.memory_matrix[positions].dot(query_vector.T).toarray().ravel()
    return {memory['memory_id']: float(score) for memory, score in zip(available_memories, sims, strict=False)}


def _slot_hazard_map() -> dict[str, float]:
    return {slot.name: slot.volatility for slot in list_slot_specs()}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _heuristic_components(query: dict[str, Any], memory: dict[str, Any], tfidf: float) -> dict[str, float]:
    hazards = _slot_hazard_map()
    age = max(0, int(query['query_time']) - int(memory['write_time']))
    rho_hat = rule_relevance_score(query, memory, tfidf, method='proposed_heuristic')
    c_hat = _clip(float(memory['source_quality']) * 0.95 + 0.03, 0.02, 0.99)
    hazard = hazards.get(memory['slot'], 0.05) * (0.85 + float(query['stress_value']))
    s_hat = _clip(math.exp(-hazard * age), 0.02, 0.99)
    return {'rho_hat': rho_hat, 'c_hat': c_hat, 's_hat': s_hat}


def _learned_components(
    query: dict[str, Any],
    memory: dict[str, Any],
    tfidf: float,
    estimator_bundle: LearnedEstimatorBundle,
    relevance_bundle: LearnedRelevanceBundle,
    observable_context: dict[str, Any],
) -> dict[str, float]:
    return {
        'rho_hat': _clip(relevance_bundle.predict_relevance(query, memory, tfidf, observable_context), 0.02, 0.99),
        'c_hat': _clip(estimator_bundle.predict_write_correctness(memory, observable_context), 0.02, 0.99),
        's_hat': _clip(estimator_bundle.predict_survival(memory, query, observable_context), 0.02, 0.99),
    }


def _blended_u(learned: dict[str, float], heuristic: dict[str, float], c_blend: float, s_blend: float) -> float:
    rho_hat = learned['rho_hat']
    c_hat = _clip(c_blend * learned['c_hat'] + (1.0 - c_blend) * heuristic['c_hat'], 0.02, 0.99)
    s_hat = _clip(s_blend * learned['s_hat'] + (1.0 - s_blend) * heuristic['s_hat'], 0.02, 0.99)
    pi_hat = _clip(c_hat * s_hat, 0.02, 0.98)
    return _clip(rho_hat * pi_hat, 0.0001, 0.98)


def _prepare_query_record(
    index: EpisodeIndex,
    query: dict[str, Any],
    estimator_bundle: LearnedEstimatorBundle,
    relevance_bundle: LearnedRelevanceBundle,
) -> QueryRecord:
    available_memories = [memory for memory in index.memories if int(memory['write_time']) <= int(query['query_time'])]
    contexts = build_query_memory_contexts(available_memories, query)
    raw_tfidf = _tfidf_similarity(index, query, available_memories)
    normalized_tfidf = normalized_similarity(raw_tfidf)
    candidates: list[dict[str, Any]] = []
    for memory in available_memories:
        tfidf = normalized_tfidf.get(memory['memory_id'], 0.0)
        observable_context = contexts.get(memory['memory_id'], {})
        candidates.append(
            {
                'memory': memory,
                'learned': _learned_components(query, memory, tfidf, estimator_bundle, relevance_bundle, observable_context),
                'heuristic': _heuristic_components(query, memory, tfidf),
            }
        )
    return QueryRecord(query=query, candidates=candidates)


def _predict_from_record(record: QueryRecord, c_blend: float, s_blend: float, top_k: int) -> str | None:
    if not record.candidates:
        return None
    ranked = sorted(
        record.candidates,
        key=lambda row: (_blended_u(row['learned'], row['heuristic'], c_blend, s_blend), int(row['memory']['write_time'])),
        reverse=True,
    )[:top_k]
    retrieved = [row['memory'] for row in ranked]
    fallback_pool = [memory for memory in retrieved if memory['entity'] == record.query['entity'] and memory['slot'] == record.query['slot']]
    fallback_pool = fallback_pool or [memory for memory in retrieved if memory['slot'] == record.query['slot']] or retrieved[:1]
    if not fallback_pool:
        return None
    candidate_values = sorted({memory['value_canonical'] for memory in fallback_pool})
    background = 1.0 / max(len(candidate_values), 2)
    value_scores: dict[str, float] = {}
    for row in ranked:
        memory = row['memory']
        if memory not in fallback_pool:
            continue
        u_hat = _blended_u(row['learned'], row['heuristic'], c_blend, s_blend)
        weight = math.log1p(u_hat / max((1.0 - u_hat) * background, 1e-6))
        value_scores[memory['value_canonical']] = value_scores.get(memory['value_canonical'], 0.0) + weight
    return max(value_scores.items(), key=lambda item: (item[1], item[0]))[0]