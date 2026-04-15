from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from memory_collapse.anti_support import (
    LearnedAntiSupportBundle,
    anti_support_artifact_paths,
    anti_support_model_exists,
    load_anti_support_bundle,
)
from memory_collapse.controller import (
    controller_artifact_paths,
    controller_calibration_exists,
    default_controller_calibration,
    load_controller_calibration,
)
from memory_collapse.domain import list_slot_specs
from memory_collapse.estimators import (
    LearnedEstimatorBundle,
    _memory_feature_dict,
    _survival_feature_dict,
    build_query_memory_contexts,
    estimator_artifact_paths,
    estimator_models_exist,
    load_estimator_bundle,
    train_estimators,
)
from memory_collapse.evaluation import classify_error_attribution, summarize_metrics
from memory_collapse.io_utils import ensure_dir, read_jsonl, write_csv, write_jsonl
from memory_collapse.query_validity import (
    LearnedQueryValidityBundle,
    augment_query_validity_feature_dict,
    build_query_validity_feature_dict,
    load_query_validity_bundle,
    query_validity_artifact_paths,
    query_validity_model_exists,
)
from memory_collapse.relevance import (
    LearnedRelevanceBundle,
    _pair_feature_dict,
    entity_match_strength,
    evaluate_relevance,
    load_relevance_bundle,
    normalized_similarity,
    relevance_artifact_paths,
    relevance_model_exists,
    rule_relevance_score,
)
from memory_collapse.value_resolver import (
    LearnedValueResolverBundle,
    build_value_candidate_feature_rows,
    load_value_resolver_bundle,
    value_resolver_artifact_paths,
    value_resolver_model_exists,
)


METHODS = [
    'latest_write',
    'majority_vote',
    'recency_weighted_vote',
    'recency_only',
    'tfidf_only',
    'tfidf_plus_recency',
    'proposed_heuristic',
    'proposed_learned',
    'proposed_learned_direct',
    'proposed_learned_direct_valid',
    'proposed_learned_direct_valid_staleaware',
    'proposed_learned_direct_valid_conflictaware',
    'proposed_learned_direct_valid_resolver',
    'oracle_latest',
    'oracle_valid',
]
PROPOSED_METHODS = {
    'proposed_heuristic',
    'proposed_learned',
    'proposed_learned_direct',
    'proposed_learned_direct_valid',
    'proposed_learned_direct_valid_staleaware',
    'proposed_learned_direct_valid_conflictaware',
    'proposed_learned_direct_valid_resolver',
}
ORACLE_METHODS = {'oracle_latest', 'oracle_valid'}
FACTORIZED_METHOD = 'proposed_learned'
DIRECT_USEFUL_METHOD = 'proposed_learned_direct'
DIRECT_VALID_METHOD = 'proposed_learned_direct_valid'
DIRECT_VALID_STALEAWARE_METHOD = 'proposed_learned_direct_valid_staleaware'
DIRECT_VALID_CONFLICTAWARE_METHOD = 'proposed_learned_direct_valid_conflictaware'
DIRECT_VALID_RESOLVER_METHOD = 'proposed_learned_direct_valid_resolver'
DIRECT_METHODS = {
    DIRECT_USEFUL_METHOD: 'useful_label',
    DIRECT_VALID_METHOD: 'valid_label',
    DIRECT_VALID_STALEAWARE_METHOD: 'valid_label',
    DIRECT_VALID_CONFLICTAWARE_METHOD: 'valid_label',
    DIRECT_VALID_RESOLVER_METHOD: 'valid_label',
}


@dataclass
class EpisodeIndex:
    episode_id: str
    memories: list[dict[str, Any]]
    memory_by_id: dict[str, dict[str, Any]]
    vectorizer: TfidfVectorizer
    memory_matrix: Any
    memory_positions: dict[str, int]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _softmax_confidence(value_scores: dict[str, float]) -> float:
    if not value_scores:
        return 0.0
    values = np.array(list(value_scores.values()), dtype=float)
    values -= values.max()
    weights = np.exp(values)
    probs = weights / weights.sum()
    return float(probs.max())


def _stable_noise(query_id: str, memory_id: str, salt: str) -> float:
    import hashlib

    digest = hashlib.md5(f'{query_id}|{memory_id}|{salt}'.encode('utf-8')).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return 2.0 * raw - 1.0


def _slot_hazard_map() -> dict[str, float]:
    return {slot.name: slot.volatility for slot in list_slot_specs()}


def _load_run_artifacts(run_dir: str | Path) -> tuple[list[dict], list[dict], list[dict]]:
    root = Path(run_dir)
    memories = read_jsonl(root / 'data' / 'memories.jsonl')
    queries = read_jsonl(root / 'data' / 'queries.jsonl')
    labels = read_jsonl(root / 'data' / 'exact_labels.jsonl')
    if not memories or not queries or not labels:
        raise FileNotFoundError('Run directory is missing generated artifacts. Run generate first.')
    return memories, queries, labels


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


def _get_available_memories(index: EpisodeIndex, query_time: int) -> list[dict[str, Any]]:
    return [memory for memory in index.memories if int(memory['write_time']) <= int(query_time)]


def _tfidf_similarity(index: EpisodeIndex, query: dict[str, Any], available_memories: list[dict[str, Any]]) -> dict[str, float]:
    if not available_memories:
        return {}
    query_vector = index.vectorizer.transform([query['query_text']])
    positions = [index.memory_positions[memory['memory_id']] for memory in available_memories]
    sims = index.memory_matrix[positions].dot(query_vector.T).toarray().ravel()
    return {memory['memory_id']: float(score) for memory, score in zip(available_memories, sims, strict=False)}


def _heuristic_components(query: dict[str, Any], memory: dict[str, Any], tfidf: float) -> dict[str, float]:
    hazards = _slot_hazard_map()
    age = max(0, int(query['query_time']) - int(memory['write_time']))
    rho_hat = rule_relevance_score(query, memory, tfidf, 'proposed_heuristic')
    c_hat = _clip(float(memory['source_quality']) * 0.95 + 0.03, 0.02, 0.99)
    hazard = hazards.get(memory['slot'], 0.05) * (0.85 + float(query['stress_value']))
    s_hat = _clip(math.exp(-hazard * age), 0.02, 0.99)
    pi_hat = _clip(c_hat * s_hat, 0.02, 0.98)
    u_hat = _clip(rho_hat * pi_hat, 0.0001, 0.98)
    return {
        'rho_hat': rho_hat,
        'c_hat': c_hat,
        's_hat': s_hat,
        'pi_hat': pi_hat,
        'u_hat': u_hat,
    }


def _factorized_learned_components(
    query: dict[str, Any],
    memory: dict[str, Any],
    tfidf: float,
    estimator_bundle: LearnedEstimatorBundle,
    relevance_bundle: LearnedRelevanceBundle,
    controller_calibration: dict[str, Any] | None,
    observable_context: dict[str, Any],
) -> dict[str, float]:
    calibration = controller_calibration or default_controller_calibration()
    heuristic = _heuristic_components(query, memory, tfidf)
    rho_hat = _clip(relevance_bundle.predict_relevance(query, memory, tfidf, observable_context), 0.02, 0.99)
    learned_c_hat = _clip(estimator_bundle.predict_write_correctness(memory, observable_context), 0.02, 0.99)
    learned_s_hat = _clip(estimator_bundle.predict_survival(memory, query, observable_context), 0.02, 0.99)
    c_blend = float(calibration.get('c_blend', 1.0))
    s_blend = float(calibration.get('s_blend', 1.0))
    c_hat = _clip(c_blend * learned_c_hat + (1.0 - c_blend) * heuristic['c_hat'], 0.02, 0.99)
    s_hat = _clip(s_blend * learned_s_hat + (1.0 - s_blend) * heuristic['s_hat'], 0.02, 0.99)
    pi_hat = _clip(c_hat * s_hat, 0.02, 0.98)
    u_hat = _clip(rho_hat * pi_hat, 0.0001, 0.98)
    return {
        'rho_hat': rho_hat,
        'c_hat': c_hat,
        's_hat': s_hat,
        'pi_hat': pi_hat,
        'u_hat': u_hat,
        'c_hat_learned': learned_c_hat,
        's_hat_learned': learned_s_hat,
        'c_hat_heuristic': heuristic['c_hat'],
        's_hat_heuristic': heuristic['s_hat'],
        'c_blend': c_blend,
        's_blend': s_blend,
        'pi_mode': 'factorized_c_times_s',
    }


def _direct_query_validity_components(
    query: dict[str, Any],
    memory: dict[str, Any],
    tfidf: float,
    relevance_bundle: LearnedRelevanceBundle,
    query_validity_bundle: LearnedQueryValidityBundle,
    observable_context: dict[str, Any],
    apply_to_all_pairs: bool,
) -> dict[str, float]:
    rho_hat = _clip(relevance_bundle.predict_relevance(query, memory, tfidf, observable_context), 0.02, 0.99)
    exact_match = int(memory['entity'] == query['entity'] and memory['slot'] == query['slot'])
    should_apply = bool(apply_to_all_pairs or exact_match)
    if should_apply:
        pi_hat_direct = _clip(query_validity_bundle.predict_query_validity(query, memory, tfidf, observable_context), 0.02, 0.99)
    else:
        pi_hat_direct = 1.0
    u_hat = _clip(rho_hat * pi_hat_direct, 0.0001, 0.98)
    return {
        'rho_hat': rho_hat,
        'pi_hat_direct': pi_hat_direct,
        'u_hat': u_hat,
        'pi_mode': 'direct_query_time_useful' if query_validity_bundle.target_label == 'useful_label' else 'direct_query_time_valid',
        'query_validity_target': query_validity_bundle.target_label,
        'query_validity_condition': query_validity_bundle.condition,
        'query_validity_applied': int(should_apply),
        'query_entity_slot_exact_match': exact_match,
    }


def _score_components(
    method: str,
    query: dict[str, Any],
    memory: dict[str, Any],
    tfidf: float,
    estimator_bundle: LearnedEstimatorBundle | None,
    relevance_bundle: LearnedRelevanceBundle | None,
    query_validity_bundles: dict[str, LearnedQueryValidityBundle] | None,
    anti_support_bundle: LearnedAntiSupportBundle | None,
    value_resolver_bundle: LearnedValueResolverBundle | None,
    controller_calibration: dict[str, Any] | None,
    observable_context: dict[str, Any],
) -> dict[str, float]:
    if method == 'proposed_heuristic':
        components = _heuristic_components(query, memory, tfidf)
        components['pi_mode'] = 'heuristic_factorized'
        return components
    if method == FACTORIZED_METHOD:
        if estimator_bundle is None or relevance_bundle is None:
            raise RuntimeError('Estimator and relevance bundles required for factorized learned controller.')
        return _factorized_learned_components(
            query,
            memory,
            tfidf,
            estimator_bundle,
            relevance_bundle,
            controller_calibration,
            observable_context,
        )
    if method in DIRECT_METHODS:
        if relevance_bundle is None or query_validity_bundles is None:
            raise RuntimeError('Relevance and query-validity bundles required for direct learned controller.')
        target_label = DIRECT_METHODS[method]
        query_validity_bundle = query_validity_bundles[target_label]
        direct_components = _direct_query_validity_components(
            query,
            memory,
            tfidf,
            relevance_bundle,
            query_validity_bundle,
            observable_context,
            apply_to_all_pairs=(query_validity_bundle.condition == 'all_pairs'),
        )
        if method == DIRECT_VALID_STALEAWARE_METHOD:
            direct_features = build_query_validity_feature_dict(query, memory, tfidf, observable_context)
            return _staleaware_direct_valid_components(
                direct_features,
                direct_components['rho_hat'],
                direct_components['pi_hat_direct'],
                direct_components['query_entity_slot_exact_match'],
            )
        if method == DIRECT_VALID_CONFLICTAWARE_METHOD:
            direct_features = build_query_validity_feature_dict(query, memory, tfidf, observable_context)
            return _conflictaware_direct_valid_components(
                direct_features,
                direct_components['rho_hat'],
                direct_components['pi_hat_direct'],
                direct_components['query_entity_slot_exact_match'],
                anti_support_hat=(
                    _clip(anti_support_bundle.predict_anti_support(direct_features), 0.02, 0.99)
                    if anti_support_bundle is not None else 0.5
                ),
            )
        return direct_components
    raise KeyError(f'Unsupported proposed method: {method}')


def _conflictaware_direct_valid_components(
    query_validity_features: dict[str, Any],
    rho_hat: float,
    valid_pi: float,
    exact_match: int,
    anti_support_hat: float,
) -> dict[str, float]:
    observed_is_latest = float(query_validity_features.get('observed_is_latest', 0.0))
    conflict_ratio = float(query_validity_features.get('observed_conflict_ratio', 0.0))
    newer_different_fraction = float(query_validity_features.get('newer_different_value_fraction', 0.0))
    same_value_fraction = float(query_validity_features.get('same_value_fraction', 0.0))
    support_margin = float(query_validity_features.get('support_margin', 0.0))
    same_source_consistency = float(query_validity_features.get('same_source_consistency', 0.0))
    base_weight = _clip(rho_hat * valid_pi, 0.0001, 0.98)
    support_boost = _clip(
        0.30 * observed_is_latest
        + 0.25 * max(0.0, support_margin)
        + 0.20 * same_value_fraction
        + 0.15 * same_source_consistency,
        0.0,
        0.8,
    )
    heuristic_contradiction = _clip(
        0.50 * newer_different_fraction
        + 0.30 * conflict_ratio
        + 0.20 * (1.0 - observed_is_latest),
        0.0,
        0.95,
    )
    contradiction_mass = _clip(0.55 * anti_support_hat + 0.45 * heuristic_contradiction, 0.0, 0.99)
    adjusted_u_hat = _clip(base_weight * (1.0 + 0.25 * support_boost), 0.0001, 0.99)
    return {
        'rho_hat': rho_hat,
        'pi_hat_direct': valid_pi,
        'u_hat': adjusted_u_hat,
        'support_boost': support_boost,
        'anti_support_hat': anti_support_hat,
        'contradiction_mass': contradiction_mass,
        'base_u_hat': base_weight,
        'pi_mode': 'direct_query_time_valid_conflictaware',
        'query_validity_target': 'valid_label',
        'query_validity_condition': 'relevant_only',
        'query_validity_applied': int(exact_match),
        'query_entity_slot_exact_match': exact_match,
    }


def _staleaware_direct_valid_components(
    query_validity_features: dict[str, Any],
    rho_hat: float,
    valid_pi: float,
    exact_match: int,
) -> dict[str, float]:
    observed_is_latest = float(query_validity_features.get('observed_is_latest', 0.0))
    conflict_ratio = float(query_validity_features.get('observed_conflict_ratio', 0.0))
    newer_different_fraction = float(query_validity_features.get('newer_different_value_fraction', 0.0))
    staleness_pressure = float(query_validity_features.get('staleness_pressure', 0.0))
    same_value_fraction = float(query_validity_features.get('same_value_fraction', 0.0))
    support_margin = float(query_validity_features.get('support_margin', 0.0))
    heuristic_pi_hat = float(query_validity_features.get('heuristic_pi_hat', valid_pi))

    stale_penalty = _clip(
        0.45 * newer_different_fraction
        + 0.25 * staleness_pressure
        + 0.15 * conflict_ratio
        + 0.15 * (1.0 - observed_is_latest),
        0.0,
        0.95,
    )
    freshness_bonus = _clip(
        0.35 * observed_is_latest
        + 0.25 * max(0.0, support_margin)
        + 0.20 * same_value_fraction
        + 0.20 * heuristic_pi_hat
        - 0.20 * newer_different_fraction,
        0.0,
        0.85,
    )
    pi_hat_direct = _clip(valid_pi * (1.0 - 0.75 * stale_penalty) + 0.12 * freshness_bonus, 0.02, 0.99)
    u_hat = _clip(rho_hat * pi_hat_direct, 0.0001, 0.98)
    return {
        'rho_hat': rho_hat,
        'pi_hat_direct_base': valid_pi,
        'pi_hat_direct': pi_hat_direct,
        'stale_penalty': stale_penalty,
        'freshness_bonus': freshness_bonus,
        'u_hat': u_hat,
        'pi_mode': 'direct_query_time_valid_staleaware',
        'query_validity_target': 'valid_label',
        'query_validity_condition': 'relevant_only',
        'query_validity_applied': int(exact_match),
        'query_entity_slot_exact_match': exact_match,
    }


def _precompute_proposed_components(
    query: dict[str, Any],
    available_memories: list[dict[str, Any]],
    normalized_tfidf_scores: dict[str, float],
    estimator_bundle: LearnedEstimatorBundle,
    relevance_bundle: LearnedRelevanceBundle,
    query_validity_bundles: dict[str, LearnedQueryValidityBundle],
    anti_support_bundle: LearnedAntiSupportBundle,
    value_resolver_bundle: LearnedValueResolverBundle | None,
    controller_calibration: dict[str, Any] | None,
    query_memory_contexts: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    component_cache: dict[tuple[str, str], dict[str, float]] = {}
    if not available_memories:
        return component_cache

    pair_features: list[dict[str, Any]] = []
    write_features: list[dict[str, Any]] = []
    survival_features: list[dict[str, Any]] = []
    heuristic_by_memory: dict[str, dict[str, float]] = {}

    for memory in available_memories:
        tfidf = normalized_tfidf_scores.get(memory['memory_id'], 0.0)
        observable_context = query_memory_contexts.get(memory['memory_id'], {})
        pair_features.append(_pair_feature_dict(query, memory, tfidf, observable_context))
        write_features.append(_memory_feature_dict(memory, estimator_bundle.slot_volatility, observable_context))
        survival_features.append(_survival_feature_dict(memory, query, estimator_bundle.slot_volatility, observable_context))
        heuristic = _heuristic_components(query, memory, tfidf)
        heuristic['pi_mode'] = 'heuristic_factorized'
        heuristic_by_memory[memory['memory_id']] = heuristic
        component_cache[('proposed_heuristic', memory['memory_id'])] = heuristic

    relevance_probs = relevance_bundle.pipeline.predict_proba(pair_features)[:, 1]
    write_probs = estimator_bundle.write_pipeline.predict_proba(write_features)[:, 1]
    survival_probs = estimator_bundle.survival_pipeline.predict_proba(survival_features)[:, 1]
    query_validity_features = [augment_query_validity_feature_dict(features) for features in pair_features]
    useful_probs = query_validity_bundles['useful_label'].pipeline.predict_proba(query_validity_features)[:, 1]
    valid_probs = query_validity_bundles['valid_label'].pipeline.predict_proba(query_validity_features)[:, 1]
    anti_support_probs = anti_support_bundle.pipeline.predict_proba(query_validity_features)[:, 1]

    calibration = controller_calibration or default_controller_calibration()
    c_blend = float(calibration.get('c_blend', 1.0))
    s_blend = float(calibration.get('s_blend', 1.0))

    for idx, memory in enumerate(available_memories):
        memory_id = memory['memory_id']
        heuristic = heuristic_by_memory[memory_id]
        rho_hat = _clip(float(relevance_probs[idx]), 0.02, 0.99)
        learned_c_hat = _clip(float(write_probs[idx]), 0.02, 0.99)
        learned_s_hat = _clip(float(survival_probs[idx]), 0.02, 0.99)
        c_hat = _clip(c_blend * learned_c_hat + (1.0 - c_blend) * heuristic['c_hat'], 0.02, 0.99)
        s_hat = _clip(s_blend * learned_s_hat + (1.0 - s_blend) * heuristic['s_hat'], 0.02, 0.99)
        pi_hat = _clip(c_hat * s_hat, 0.02, 0.98)
        component_cache[(FACTORIZED_METHOD, memory_id)] = {
            'rho_hat': rho_hat,
            'c_hat': c_hat,
            's_hat': s_hat,
            'pi_hat': pi_hat,
            'u_hat': _clip(rho_hat * pi_hat, 0.0001, 0.98),
            'c_hat_learned': learned_c_hat,
            's_hat_learned': learned_s_hat,
            'c_hat_heuristic': heuristic['c_hat'],
            's_hat_heuristic': heuristic['s_hat'],
            'c_blend': c_blend,
            's_blend': s_blend,
            'pi_mode': 'factorized_c_times_s',
        }

        exact_match = int(memory['entity'] == query['entity'] and memory['slot'] == query['slot'])
        useful_pi = _clip(float(useful_probs[idx]), 0.02, 0.99)
        component_cache[(DIRECT_USEFUL_METHOD, memory_id)] = {
            'rho_hat': rho_hat,
            'pi_hat_direct': useful_pi,
            'u_hat': _clip(rho_hat * useful_pi, 0.0001, 0.98),
            'pi_mode': 'direct_query_time_useful',
            'query_validity_target': 'useful_label',
            'query_validity_condition': query_validity_bundles['useful_label'].condition,
            'query_validity_applied': 1,
            'query_entity_slot_exact_match': exact_match,
        }

        apply_valid = bool(query_validity_bundles['valid_label'].condition == 'all_pairs' or exact_match)
        valid_pi = _clip(float(valid_probs[idx]), 0.02, 0.99) if apply_valid else 1.0
        component_cache[(DIRECT_VALID_METHOD, memory_id)] = {
            'rho_hat': rho_hat,
            'pi_hat_direct': valid_pi,
            'u_hat': _clip(rho_hat * valid_pi, 0.0001, 0.98),
            'pi_mode': 'direct_query_time_valid',
            'query_validity_target': 'valid_label',
            'query_validity_condition': query_validity_bundles['valid_label'].condition,
            'query_validity_applied': int(apply_valid),
            'query_entity_slot_exact_match': exact_match,
        }
        component_cache[(DIRECT_VALID_RESOLVER_METHOD, memory_id)] = {
            'rho_hat': rho_hat,
            'pi_hat_direct': valid_pi,
            'u_hat': _clip(rho_hat * valid_pi, 0.0001, 0.98),
            'anti_support_hat': _clip(float(anti_support_probs[idx]), 0.02, 0.99),
            'pi_mode': 'direct_query_time_valid_resolver',
            'query_validity_target': 'valid_label',
            'query_validity_condition': query_validity_bundles['valid_label'].condition,
            'query_validity_applied': int(apply_valid),
            'query_entity_slot_exact_match': exact_match,
        }
        component_cache[(DIRECT_VALID_STALEAWARE_METHOD, memory_id)] = _staleaware_direct_valid_components(
            query_validity_features[idx],
            rho_hat,
            valid_pi,
            exact_match,
        )
        component_cache[(DIRECT_VALID_CONFLICTAWARE_METHOD, memory_id)] = _conflictaware_direct_valid_components(
            query_validity_features[idx],
            rho_hat,
            valid_pi,
            exact_match,
            anti_support_hat=_clip(float(anti_support_probs[idx]), 0.02, 0.99),
        )

    return component_cache


def _rank_available(
    method: str,
    query: dict[str, Any],
    label: dict[str, Any],
    available_memories: list[dict[str, Any]],
    tfidf_scores: dict[str, float],
    top_k: int,
    estimator_bundle: LearnedEstimatorBundle | None,
    relevance_bundle: LearnedRelevanceBundle | None,
    query_validity_bundles: dict[str, LearnedQueryValidityBundle] | None,
    anti_support_bundle: LearnedAntiSupportBundle | None,
    value_resolver_bundle: LearnedValueResolverBundle | None,
    controller_calibration: dict[str, Any] | None,
    query_memory_contexts: dict[str, dict[str, Any]],
    component_cache: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    if not available_memories:
        return []
    normalized_tfidf = normalized_similarity(tfidf_scores)
    relevant_ids = set(label['relevant_memory_ids'])
    valid_ids = set(label['valid_memory_ids'])
    useful_ids = set(label['useful_memory_ids'])
    scored: list[tuple[float, dict[str, Any]]] = []

    for memory in available_memories:
        age = max(0, int(query['query_time']) - int(memory['write_time']))
        recency_score = 1.0 / (age + 1.0)
        tfidf = normalized_tfidf.get(memory['memory_id'], 0.0)
        entity_match = entity_match_strength(query['entity'], memory['entity'])
        slot_match = 1.0 if query['slot'] == memory['slot'] else 0.0
        noise = _stable_noise(query['query_id'], memory['memory_id'], method)
        distractor_penalty = float(query['distractor_overlap']) * max(0.0, noise)
        observable_context = query_memory_contexts.get(memory['memory_id'], {})
        if method == 'latest_write':
            score = float(entity_match == 1.0 and slot_match == 1.0) * (1000.0 - age)
        elif method == 'majority_vote':
            score = float(entity_match == 1.0 and slot_match == 1.0) * (100.0 - age)
        elif method == 'recency_weighted_vote':
            score = float(entity_match == 1.0 and slot_match == 1.0) * (100.0 - age)
        elif method == 'recency_only':
            score = recency_score - distractor_penalty
        elif method == 'tfidf_only':
            score = tfidf + 0.08 * entity_match + 0.06 * slot_match - distractor_penalty
        elif method == 'tfidf_plus_recency':
            score = 0.72 * tfidf + 0.22 * recency_score + 0.04 * entity_match + 0.02 * slot_match - distractor_penalty
        elif method in PROPOSED_METHODS:
            cache_key = (method, memory['memory_id'])
            components = component_cache.get(cache_key)
            if components is None:
                components = _score_components(
                    method,
                    query,
                    memory,
                    tfidf,
                    estimator_bundle,
                    relevance_bundle,
                    query_validity_bundles,
                    anti_support_bundle,
                    value_resolver_bundle,
                    controller_calibration,
                    observable_context,
                )
                component_cache[cache_key] = components
            score = components['u_hat']
        elif method == 'oracle_latest':
            score = 10000.0 if memory['memory_id'] in valid_ids else -1.0
            score += recency_score
        elif method == 'oracle_valid':
            score = 10000.0 if memory['memory_id'] in useful_ids else -1.0
            score += 0.1 * recency_score
        else:
            raise KeyError(f'Unknown method: {method}')
        if method in ORACLE_METHODS and memory['memory_id'] not in relevant_ids:
            score -= 5000.0
        scored.append((float(score), memory))

    scored.sort(key=lambda item: (item[0], int(item[1]['write_time'])), reverse=True)
    return [memory for _, memory in scored[:top_k]]


def _aggregate_prediction(
    method: str,
    query: dict[str, Any],
    retrieved: list[dict[str, Any]],
    estimator_bundle: LearnedEstimatorBundle | None,
    relevance_bundle: LearnedRelevanceBundle | None,
    query_validity_bundles: dict[str, LearnedQueryValidityBundle] | None,
    anti_support_bundle: LearnedAntiSupportBundle | None,
    value_resolver_bundle: LearnedValueResolverBundle | None,
    controller_calibration: dict[str, Any] | None,
    tfidf_lookup: dict[str, float],
    query_memory_contexts: dict[str, dict[str, Any]],
    component_cache: dict[tuple[str, str], dict[str, float]],
) -> tuple[str | None, dict[str, float], dict[str, dict[str, float]]]:
    if not retrieved:
        return None, {}, {}
    relevant_retrieved = [
        memory for memory in retrieved if memory['entity'] == query['entity'] and memory['slot'] == query['slot']
    ]
    fallback_pool = relevant_retrieved or [memory for memory in retrieved if memory['slot'] == query['slot']] or retrieved[:1]
    if not fallback_pool:
        return None, {}, {}

    component_debug: dict[str, dict[str, float]] = {}

    if method == 'latest_write':
        chosen = max(fallback_pool, key=lambda memory: int(memory['write_time']))
        return chosen['value_canonical'], {chosen['value_canonical']: 1.0}, component_debug

    if method in {'majority_vote', 'tfidf_only', 'oracle_latest', 'oracle_valid'}:
        value_scores: dict[str, float] = {}
        for memory in fallback_pool:
            value_scores[memory['value_canonical']] = value_scores.get(memory['value_canonical'], 0.0) + 1.0
        prediction = max(value_scores.items(), key=lambda item: (item[1], item[0]))[0]
        return prediction, value_scores, component_debug

    if method in {'recency_weighted_vote', 'recency_only', 'tfidf_plus_recency'}:
        value_scores: dict[str, float] = {}
        for memory in fallback_pool:
            age = max(0, int(query['query_time']) - int(memory['write_time']))
            value_scores[memory['value_canonical']] = value_scores.get(memory['value_canonical'], 0.0) + 1.0 / (age + 1.0)
        prediction = max(value_scores.items(), key=lambda item: (item[1], item[0]))[0]
        return prediction, value_scores, component_debug

    if method in PROPOSED_METHODS:
        candidate_values = sorted({memory['value_canonical'] for memory in fallback_pool})
        background = 1.0 / max(len(candidate_values), 2)
        value_scores: dict[str, float] = {}
        contradiction_scores: dict[str, float] = {value: 0.0 for value in candidate_values}
        for memory in fallback_pool:
            tfidf = tfidf_lookup.get(memory['memory_id'], 0.0)
            observable_context = query_memory_contexts.get(memory['memory_id'], {})
            cache_key = (method, memory['memory_id'])
            components = component_cache.get(cache_key)
            if components is None:
                components = _score_components(
                    method,
                    query,
                    memory,
                    tfidf,
                    estimator_bundle,
                    relevance_bundle,
                    query_validity_bundles,
                    anti_support_bundle,
                    value_resolver_bundle,
                    controller_calibration,
                    observable_context,
                )
                component_cache[cache_key] = components
            component_debug[memory['memory_id']] = {
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in components.items()
            }
            weight = math.log1p(components['u_hat'] / max((1.0 - components['u_hat']) * background, 1e-6))
            memory_value = memory['value_canonical']
            if method == DIRECT_VALID_CONFLICTAWARE_METHOD:
                support_weight = weight * (1.0 + 0.35 * float(components.get('support_boost', 0.0)))
                contradiction_weight = weight * float(components.get('contradiction_mass', 0.0))
                value_scores[memory_value] = value_scores.get(memory_value, 0.0) + support_weight
                for candidate_value in candidate_values:
                    if candidate_value != memory_value:
                        contradiction_scores[candidate_value] = contradiction_scores.get(candidate_value, 0.0) + contradiction_weight
            else:
                value_scores[memory_value] = value_scores.get(memory_value, 0.0) + weight
        if method == DIRECT_VALID_CONFLICTAWARE_METHOD:
            for candidate_value in candidate_values:
                value_scores[candidate_value] = value_scores.get(candidate_value, 0.0) - contradiction_scores.get(candidate_value, 0.0)
        if method == DIRECT_VALID_RESOLVER_METHOD:
            if value_resolver_bundle is None:
                raise RuntimeError('Value-resolver bundle required for direct-valid resolver method.')
            resolver_lookup = {
                memory['memory_id']: component_cache[(DIRECT_VALID_RESOLVER_METHOD, memory['memory_id'])]
                for memory in fallback_pool
            }
            candidate_rows = build_value_candidate_feature_rows(query, fallback_pool, resolver_lookup)
            resolver_features = [
                {key: value for key, value in row.items() if key not in {'query_id', 'episode_id', 'candidate_value'}}
                for row in candidate_rows
            ]
            resolver_probs = value_resolver_bundle.predict_candidate_scores(resolver_features)
            value_scores = {
                row['candidate_value']: prob
                for row, prob in zip(candidate_rows, resolver_probs, strict=False)
            }
        prediction = max(value_scores.items(), key=lambda item: (item[1], item[0]))[0]
        return prediction, value_scores, component_debug

    raise KeyError(f'Unsupported aggregation method: {method}')


def _resolve_model_bundles(
    run_dir: str | Path,
    estimator_dir: str | Path | None = None,
    force_train: bool = False,
    skip_train: bool = False,
) -> tuple[
    LearnedEstimatorBundle,
    LearnedRelevanceBundle,
    dict[str, LearnedQueryValidityBundle],
    LearnedAntiSupportBundle,
    LearnedValueResolverBundle,
    dict[str, Any],
    dict[str, str],
    str,
]:
    target_dir = Path(estimator_dir) if estimator_dir else Path(run_dir)
    if force_train and skip_train:
        raise ValueError('force_train and skip_train cannot both be true.')

    def _all_models_exist() -> bool:
        return (
            estimator_models_exist(target_dir)
            and relevance_model_exists(target_dir)
            and query_validity_model_exists(target_dir)
            and anti_support_model_exists(target_dir)
            and value_resolver_model_exists(target_dir)
            and controller_calibration_exists(target_dir)
        )

    def _load_query_validity_bundles() -> dict[str, LearnedQueryValidityBundle]:
        return {
            'useful_label': load_query_validity_bundle(target_dir, target_label='useful_label'),
            'valid_label': load_query_validity_bundle(target_dir, target_label='valid_label'),
        }

    if force_train:
        artifacts = train_estimators(run_dir, output_dir=target_dir)
        estimator_bundle = load_estimator_bundle(target_dir)
        relevance_bundle = load_relevance_bundle(target_dir)
        query_validity_bundles = _load_query_validity_bundles()
        anti_support_bundle = load_anti_support_bundle(target_dir)
        value_resolver_bundle = load_value_resolver_bundle(target_dir)
        controller_calibration = load_controller_calibration(target_dir)
        return estimator_bundle, relevance_bundle, query_validity_bundles, anti_support_bundle, value_resolver_bundle, controller_calibration, artifacts, 'trained'

    if _all_models_exist():
        artifacts = estimator_artifact_paths(target_dir)
        artifacts.update(relevance_artifact_paths(target_dir))
        artifacts.update(query_validity_artifact_paths(target_dir))
        artifacts.update(anti_support_artifact_paths(target_dir))
        artifacts.update(value_resolver_artifact_paths(target_dir))
        artifacts.update(controller_artifact_paths(target_dir))
        estimator_bundle = load_estimator_bundle(target_dir)
        relevance_bundle = load_relevance_bundle(target_dir)
        query_validity_bundles = _load_query_validity_bundles()
        anti_support_bundle = load_anti_support_bundle(target_dir)
        value_resolver_bundle = load_value_resolver_bundle(target_dir)
        controller_calibration = load_controller_calibration(target_dir)
        return estimator_bundle, relevance_bundle, query_validity_bundles, anti_support_bundle, value_resolver_bundle, controller_calibration, artifacts, 'reused'

    if skip_train:
        raise FileNotFoundError(f'No learned models found under {target_dir} and skip_train was requested.')

    artifacts = train_estimators(run_dir, output_dir=target_dir)
    estimator_bundle = load_estimator_bundle(target_dir)
    relevance_bundle = load_relevance_bundle(target_dir)
    query_validity_bundles = _load_query_validity_bundles()
    anti_support_bundle = load_anti_support_bundle(target_dir)
    value_resolver_bundle = load_value_resolver_bundle(target_dir)
    controller_calibration = load_controller_calibration(target_dir)
    return estimator_bundle, relevance_bundle, query_validity_bundles, anti_support_bundle, value_resolver_bundle, controller_calibration, artifacts, 'trained'


def run_baselines(
    run_dir: str | Path,
    estimator_dir: str | Path | None = None,
    force_train: bool = False,
    skip_train: bool = False,
) -> dict[str, str]:
    memories, queries, labels = _load_run_artifacts(run_dir)
    estimator_bundle, relevance_bundle, query_validity_bundles, anti_support_bundle, value_resolver_bundle, controller_calibration, model_artifacts, estimator_mode = _resolve_model_bundles(
        run_dir,
        estimator_dir=estimator_dir,
        force_train=force_train,
        skip_train=skip_train,
    )
    model_dir = Path(estimator_dir) if estimator_dir else Path(run_dir)
    episode_indices = _build_episode_indices(memories, queries)
    label_by_query = {label['query_id']: label for label in labels}
    results_dir = ensure_dir(Path(run_dir) / 'results')
    config_snapshot = yaml.safe_load((Path(run_dir) / 'config_snapshot.yaml').read_text(encoding='utf-8'))
    default_top_k = int(config_snapshot['dataset'].get('top_k', 12))

    diagnostics_rows: list[dict[str, Any]] = []
    for query in sorted(queries, key=lambda row: row['query_id']):
        index = episode_indices[query['episode_id']]
        label = label_by_query[query['query_id']]
        available_memories = _get_available_memories(index, int(query['query_time']))
        query_memory_contexts = build_query_memory_contexts(available_memories, query)
        raw_tfidf_scores = _tfidf_similarity(index, query, available_memories)
        normalized_tfidf_scores = normalized_similarity(raw_tfidf_scores)
        component_cache = _precompute_proposed_components(
            query,
            available_memories,
            normalized_tfidf_scores,
            estimator_bundle,
            relevance_bundle,
            query_validity_bundles,
            anti_support_bundle,
            value_resolver_bundle,
            controller_calibration,
            query_memory_contexts,
        )
        for method in METHODS:
            if method == 'oracle_valid' and not label['useful_memory_ids']:
                ranked = []
                predicted_value = query['gold_value']
                value_scores = {query['gold_value']: 1.0}
                component_debug = {}
            else:
                top_k = 1 if method in {'latest_write', 'oracle_latest'} else default_top_k
                ranked = _rank_available(
                    method,
                    query,
                    label,
                    available_memories,
                    raw_tfidf_scores,
                    top_k=top_k,
                    estimator_bundle=estimator_bundle,
                    relevance_bundle=relevance_bundle,
                    query_validity_bundles=query_validity_bundles,
                    anti_support_bundle=anti_support_bundle,
                    value_resolver_bundle=value_resolver_bundle,
                    controller_calibration=controller_calibration,
                    query_memory_contexts=query_memory_contexts,
                    component_cache=component_cache,
                )
                predicted_value, value_scores, component_debug = _aggregate_prediction(
                    method,
                    query,
                    ranked,
                    estimator_bundle=estimator_bundle,
                    relevance_bundle=relevance_bundle,
                    query_validity_bundles=query_validity_bundles,
                    anti_support_bundle=anti_support_bundle,
                    value_resolver_bundle=value_resolver_bundle,
                    controller_calibration=controller_calibration,
                    tfidf_lookup=normalized_tfidf_scores,
                    query_memory_contexts=query_memory_contexts,
                    component_cache=component_cache,
                )
            prediction = {
                'predicted_value': predicted_value,
                'retrieved_ids': [memory['memory_id'] for memory in ranked],
                'value_scores': value_scores,
            }
            attribution = classify_error_attribution(prediction, query, label, index.memory_by_id)
            diagnostics_rows.append(
                {
                    'query_id': query['query_id'],
                    'method': method,
                    'stress_name': query['stress_name'],
                    'stress_value': float(query['stress_value']),
                    'episode_id': query['episode_id'],
                    'entity': query['entity'],
                    'slot': query['slot'],
                    'query_time': int(query['query_time']),
                    'gold_value': query['gold_value'],
                    'predicted_value': predicted_value,
                    'confidence': _softmax_confidence(value_scores),
                    'retrieved_ids': prediction['retrieved_ids'],
                    'candidate_scores': {key: round(value, 6) for key, value in sorted(value_scores.items())},
                    'memory_components': component_debug,
                    'rho_mode': (
                        'learned' if method in (PROPOSED_METHODS - {'proposed_heuristic'})
                        else ('heuristic' if method == 'proposed_heuristic' else None)
                    ),
                    'pi_mode': (
                        'factorized_c_times_s' if method == FACTORIZED_METHOD else (
                            'direct_query_time_useful' if method == DIRECT_USEFUL_METHOD else (
                                'direct_query_time_valid' if method == DIRECT_VALID_METHOD else (
                                    'direct_query_time_valid_staleaware' if method == DIRECT_VALID_STALEAWARE_METHOD else (
                                        'direct_query_time_valid_conflictaware' if method == DIRECT_VALID_CONFLICTAWARE_METHOD else (
                                            'direct_query_time_valid_resolver' if method == DIRECT_VALID_RESOLVER_METHOD else (
                                                'heuristic_factorized' if method == 'proposed_heuristic' else None
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    'controller_blend': controller_calibration if method == FACTORIZED_METHOD else None,
                    **attribution,
                }
            )

    metrics_rows = summarize_metrics(diagnostics_rows)
    diagnostics_path = write_jsonl(results_dir / 'query_diagnostics.jsonl', diagnostics_rows)
    metrics_path = write_csv(results_dir / 'metrics_by_method.csv', metrics_rows)
    relevance_eval_top_k = min(default_top_k, 3)
    relevance_summary = evaluate_relevance(run_dir, relevance_bundle, model_dir, top_k=relevance_eval_top_k)
    learned_variant_summary = _compute_learned_variant_summary(metrics_rows)
    relevance_done = _compute_relevance_done_criteria(relevance_summary, learned_variant_summary['comparisons'])
    relevance_eval_path = results_dir / 'relevance_eval.json'
    with relevance_eval_path.open('w', encoding='utf-8') as handle:
        json.dump(
            {
                **relevance_summary,
                'comparisons': learned_variant_summary['comparisons'],
                'done': relevance_done,
            },
            handle,
            indent=2,
        )
    ordering_summary = _compute_oracle_ordering(metrics_rows)
    monotonic_summary = _compute_monotonicity(metrics_rows)
    summary_path = results_dir / 'acceptance_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(
            {
                'oracle_ordering': ordering_summary,
                'collapse_monotonicity': monotonic_summary,
                'estimators': {
                    'mode': estimator_mode,
                    **model_artifacts,
                },
                'controller_calibration': controller_calibration,
                'learned_variants': learned_variant_summary,
                'relevance': {
                    **relevance_summary,
                    'comparisons': learned_variant_summary['comparisons'],
                    'done': relevance_done,
                    'artifact': str(relevance_eval_path),
                },
            },
            handle,
            indent=2,
        )
    outputs = {
        'diagnostics': str(diagnostics_path),
        'metrics': str(metrics_path),
        'acceptance_summary': str(summary_path),
        'relevance_eval': str(relevance_eval_path),
        'write_model': model_artifacts['write_model'],
        'survival_model': model_artifacts['survival_model'],
        'rho_model': model_artifacts['rho_model'],
        'estimator_summary': model_artifacts['summary'],
        'rho_summary': model_artifacts['rho_summary'],
        'rho_examples': model_artifacts['rho_examples'],
        'anti_support_model': model_artifacts['anti_support_model'],
        'anti_support_summary': model_artifacts['anti_support_summary'],
        'anti_support_examples': model_artifacts['anti_support_examples'],
        'value_resolver_model': model_artifacts['value_resolver_model'],
        'value_resolver_summary': model_artifacts['value_resolver_summary'],
        'value_resolver_examples': model_artifacts['value_resolver_examples'],
        'controller_calibration': model_artifacts['controller_calibration'],
        'estimator_splits': model_artifacts['splits'],
    }
    for key in [
        'query_validity_model',
        'query_validity_summary',
        'query_validity_examples',
        'query_validity_useful_model',
        'query_validity_useful_summary',
        'query_validity_useful_examples',
        'query_validity_valid_model',
        'query_validity_valid_summary',
        'query_validity_valid_examples',
    ]:
        if key in model_artifacts:
            outputs[key] = model_artifacts[key]
    return outputs


def _compute_oracle_ordering(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordering_values = {row['method']: row['accuracy'] for row in metrics_rows if row['stress_name'] == 'overall'}
    baseline_candidates = [method for method in ordering_values if method not in PROPOSED_METHODS and method not in ORACLE_METHODS]
    proposed_candidates = [method for method in ordering_values if method in PROPOSED_METHODS]
    strongest_baseline = max((ordering_values[method] for method in baseline_candidates), default=-1.0)
    strongest_proposed = max((ordering_values[method] for method in proposed_candidates), default=-1.0)
    holds = (
        ordering_values.get('oracle_valid', -1.0) >= ordering_values.get('oracle_latest', -1.0)
        >= strongest_proposed
        >= strongest_baseline
    )
    return {
        'holds': bool(holds),
        'accuracies': ordering_values,
        'strongest_baseline_methods': baseline_candidates,
        'strongest_proposed_methods': proposed_candidates,
    }


def _compute_monotonicity(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {
        row['method']: row
        for row in metrics_rows
        if row['stress_name'] == 'overall' and row['method'] in PROPOSED_METHODS
    }
    candidate_method = max(
        overall,
        key=lambda method: (
            float(overall[method]['accuracy']),
            -float(overall[method]['collapse_rate']),
            method,
        ),
        default='proposed_heuristic',
    )
    proposed_rows = [row for row in metrics_rows if row['method'] == candidate_method and row['stress_name'] != 'overall']
    proposed_rows.sort(key=lambda row: row['stress_value'])
    collapse = [float(row['collapse_rate']) for row in proposed_rows]
    decreases = [max(0.0, collapse[i] - collapse[i + 1]) for i in range(len(collapse) - 1)]
    max_drop = max(decreases or [0.0])
    return {
        'method': candidate_method,
        'near_monotone': bool(max_drop <= 0.08),
        'max_drop': float(max_drop),
        'collapse_rates': collapse,
    }


def _method_summary(overall: dict[str, dict[str, Any]], method: str) -> dict[str, float]:
    row = overall.get(method, {})
    return {
        'accuracy': float(row.get('accuracy', 0.0)),
        'collapse_rate': float(row.get('collapse_rate', 1.0)),
        'stale_dominance_rate': float(row.get('stale_dominance_rate', 0.0)),
    }


def _compare_against_heuristic(heuristic: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    accuracy_gain = candidate['accuracy'] - heuristic['accuracy']
    collapse_improvement = heuristic['collapse_rate'] - candidate['collapse_rate']
    stale_improvement = heuristic['stale_dominance_rate'] - candidate['stale_dominance_rate']
    return {
        'heuristic': heuristic,
        'candidate': candidate,
        'gains': {
            'accuracy': accuracy_gain,
            'collapse_improvement': collapse_improvement,
            'stale_dominance_improvement': stale_improvement,
        },
        'thresholds': {
            'accuracy_gain': 0.01,
            'collapse_improvement': 0.01,
            'stale_dominance_improvement': 0.01,
        },
        'done_conditions': {
            'accuracy_improved': bool(accuracy_gain >= 0.01),
            'collapse_improved': bool(collapse_improvement >= 0.01),
            'stale_dominance_improved': bool(stale_improvement >= 0.01),
        },
    }


def _compute_learned_variant_summary(metrics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {row['method']: row for row in metrics_rows if row['stress_name'] == 'overall'}
    heuristic = _method_summary(overall, 'proposed_heuristic')
    factorized = _method_summary(overall, FACTORIZED_METHOD)
    direct_useful = _method_summary(overall, DIRECT_USEFUL_METHOD)
    direct_valid = _method_summary(overall, DIRECT_VALID_METHOD)
    direct_valid_staleaware = _method_summary(overall, DIRECT_VALID_STALEAWARE_METHOD)
    direct_valid_conflictaware = _method_summary(overall, DIRECT_VALID_CONFLICTAWARE_METHOD)
    direct_valid_resolver = _method_summary(overall, DIRECT_VALID_RESOLVER_METHOD)
    comparisons = {
        'factorized_vs_heuristic': _compare_against_heuristic(heuristic, factorized),
        'direct_useful_vs_heuristic': _compare_against_heuristic(heuristic, direct_useful),
        'direct_valid_vs_heuristic': _compare_against_heuristic(heuristic, direct_valid),
        'direct_valid_staleaware_vs_heuristic': _compare_against_heuristic(heuristic, direct_valid_staleaware),
        'direct_valid_conflictaware_vs_heuristic': _compare_against_heuristic(heuristic, direct_valid_conflictaware),
        'direct_valid_resolver_vs_heuristic': _compare_against_heuristic(heuristic, direct_valid_resolver),
    }
    best_direct_key = max(
        ('direct_useful_vs_heuristic', 'direct_valid_vs_heuristic', 'direct_valid_staleaware_vs_heuristic', 'direct_valid_conflictaware_vs_heuristic', 'direct_valid_resolver_vs_heuristic'),
        key=lambda key: (
            comparisons[key]['candidate']['accuracy'],
            -comparisons[key]['candidate']['collapse_rate'],
            comparisons[key]['candidate']['stale_dominance_rate'],
            key,
        ),
    )
    return {
        'heuristic': heuristic,
        'factorized_learned': factorized,
        'direct_useful_learned': direct_useful,
        'direct_valid_learned': direct_valid,
        'direct_valid_staleaware_learned': direct_valid_staleaware,
        'direct_valid_conflictaware_learned': direct_valid_conflictaware,
        'direct_valid_resolver_learned': direct_valid_resolver,
        'direct_validity_learned': direct_valid,
        'best_direct_method': best_direct_key,
        'comparisons': {
            **comparisons,
            'best_direct_vs_heuristic': comparisons[best_direct_key],
        },
    }


def _compute_relevance_done_criteria(
    relevance_summary: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    relevant_ok = bool(relevance_summary['done_conditions']['relevant_recall_improved'])
    useful_ok = bool(relevance_summary['done_conditions']['useful_recall_improved'])
    accuracy_ok = any(bool(comp['done_conditions']['accuracy_improved']) for comp in comparisons.values())
    collapse_ok = any(bool(comp['done_conditions']['collapse_improved']) for comp in comparisons.values())
    stale_ok = any(bool(comp['done_conditions']['stale_dominance_improved']) for comp in comparisons.values())
    return {
        'holds': bool(relevant_ok or useful_ok or accuracy_ok or collapse_ok or stale_ok),
        'conditions': {
            'relevant_recall_improved': relevant_ok,
            'useful_recall_improved': useful_ok,
            'accuracy_improved': accuracy_ok,
            'collapse_improved': collapse_ok,
            'stale_dominance_improved': stale_ok,
        },
    }
