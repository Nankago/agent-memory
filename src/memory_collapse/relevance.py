from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from memory_collapse.estimators import (
    SPLIT_FILENAME,
    build_query_memory_contexts,
    load_episode_splits,
    resolve_models_dir,
    split_episode_ids,
)
from memory_collapse.io_utils import ensure_dir, read_jsonl, write_csv


RHO_MODEL_FILENAME = 'rho_model.pkl'
RHO_SUMMARY_FILENAME = 'rho_summary.json'
RHO_EXAMPLES_FILENAME = 'rho_training_examples.csv'


@dataclass
class EpisodeIndex:
    episode_id: str
    memories: list[dict[str, Any]]
    labels_by_query: dict[str, dict[str, Any]]
    vectorizer: TfidfVectorizer
    memory_matrix: Any
    memory_positions: dict[str, int]


@dataclass
class LearnedRelevanceBundle:
    pipeline: Pipeline

    def predict_relevance(
        self,
        query: dict[str, Any],
        memory: dict[str, Any],
        tfidf_score: float,
        observable_context: dict[str, Any] | None = None,
    ) -> float:
        features = [_pair_feature_dict(query, memory, tfidf_score, observable_context or {})]
        return float(self.pipeline.predict_proba(features)[0, 1])


def relevance_artifact_paths(path: str | Path) -> dict[str, str]:
    models_dir = resolve_models_dir(path)
    return {
        'rho_model': str(models_dir / RHO_MODEL_FILENAME),
        'rho_summary': str(models_dir / RHO_SUMMARY_FILENAME),
        'rho_examples': str(models_dir / RHO_EXAMPLES_FILENAME),
        'splits': str(models_dir / SPLIT_FILENAME),
    }


def relevance_model_exists(path: str | Path) -> bool:
    return Path(relevance_artifact_paths(path)['rho_model']).exists()


def load_relevance_bundle(path: str | Path) -> LearnedRelevanceBundle:
    models_dir = resolve_models_dir(path)
    with (models_dir / RHO_MODEL_FILENAME).open('rb') as handle:
        pipeline = pickle.load(handle)
    return LearnedRelevanceBundle(pipeline=pipeline)


def normalized_similarity(raw_scores: dict[str, float]) -> dict[str, float]:
    if not raw_scores:
        return {}
    values = list(raw_scores.values())
    minimum = min(values)
    maximum = max(values)
    if math.isclose(maximum, minimum):
        return {key: 0.5 for key in raw_scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in raw_scores.items()}


def entity_match_strength(query_entity: str, memory_entity: str) -> float:
    if query_entity == memory_entity:
        return 1.0
    query_tokens = set(query_entity.lower().split())
    memory_tokens = set(memory_entity.lower().split())
    overlap = len(query_tokens & memory_tokens)
    return 0.45 if overlap else 0.0


def rule_relevance_score(query: dict[str, Any], memory: dict[str, Any], tfidf: float, method: str = 'proposed_heuristic') -> float:
    age = max(0, int(query['query_time']) - int(memory['write_time']))
    recency_score = 1.0 / (age + 1.0)
    entity_match = entity_match_strength(query['entity'], memory['entity'])
    slot_match = 1.0 if query['slot'] == memory['slot'] else 0.0
    noise = _stable_noise(query['query_id'], memory['memory_id'], method)
    distractor_penalty = float(query['distractor_overlap']) * max(0.0, noise)
    score = 0.45 * tfidf + 0.25 * entity_match + 0.25 * slot_match + 0.05 * recency_score - distractor_penalty
    return _clip(score, 0.0, 1.0)


def train_relevance_model(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    run_path = Path(run_dir)
    models_dir = ensure_dir(resolve_models_dir(output_dir or run_path))

    memories = read_jsonl(run_path / 'data' / 'memories.jsonl')
    queries = read_jsonl(run_path / 'data' / 'queries.jsonl')
    labels = read_jsonl(run_path / 'data' / 'exact_labels.jsonl')
    if not memories or not queries or not labels:
        raise RuntimeError('Relevance training requires generated memories, queries, and exact labels.')

    config_snapshot = {}
    config_path = run_path / 'config_snapshot.yaml'
    if config_path.exists():
        config_snapshot = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    seed = int(config_snapshot.get('dataset', {}).get('seed', 0))

    examples = _build_relevance_examples(memories, queries, labels)
    splits_path = models_dir / SPLIT_FILENAME
    if splits_path.exists():
        episode_splits = load_episode_splits(models_dir)
    else:
        episode_splits = split_episode_ids(sorted({row['episode_id'] for row in examples}), seed)
        with splits_path.open('w', encoding='utf-8') as handle:
            json.dump(episode_splits, handle, indent=2)

    pipeline, summary = _fit_relevance_with_splits(examples, episode_splits)
    with (models_dir / RHO_MODEL_FILENAME).open('wb') as handle:
        pickle.dump(pipeline, handle)
    with (models_dir / RHO_SUMMARY_FILENAME).open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    write_csv(models_dir / RHO_EXAMPLES_FILENAME, examples)
    return relevance_artifact_paths(models_dir)


def evaluate_relevance(
    run_dir: str | Path,
    bundle: LearnedRelevanceBundle,
    model_dir: str | Path,
    top_k: int,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    memories = read_jsonl(run_path / 'data' / 'memories.jsonl')
    queries = read_jsonl(run_path / 'data' / 'queries.jsonl')
    labels = read_jsonl(run_path / 'data' / 'exact_labels.jsonl')
    episode_splits = load_episode_splits(model_dir)
    test_episodes = set(episode_splits.get('test', []))

    indices = _build_episode_indices(memories, queries, labels)
    query_metrics: list[dict[str, Any]] = []
    for query in queries:
        if query['episode_id'] not in test_episodes:
            continue
        index = indices[query['episode_id']]
        label = index.labels_by_query[query['query_id']]
        available_memories = [memory for memory in index.memories if int(memory['write_time']) <= int(query['query_time'])]
        contexts = build_query_memory_contexts(available_memories, query)
        raw_tfidf = _tfidf_similarity(index, query, available_memories)
        normalized_tfidf = normalized_similarity(raw_tfidf)

        rule_ranked = sorted(
            available_memories,
            key=lambda memory: (
                rule_relevance_score(query, memory, normalized_tfidf.get(memory['memory_id'], 0.0), method='proposed_heuristic'),
                int(memory['write_time']),
            ),
            reverse=True,
        )[:top_k]
        learned_ranked = sorted(
            available_memories,
            key=lambda memory: (
                bundle.predict_relevance(
                    query,
                    memory,
                    normalized_tfidf.get(memory['memory_id'], 0.0),
                    contexts.get(memory['memory_id'], {}),
                ),
                int(memory['write_time']),
            ),
            reverse=True,
        )[:top_k]

        relevant_ids = set(label['relevant_memory_ids'])
        valid_ids = set(label['valid_memory_ids'])
        useful_ids = set(label['useful_memory_ids'])
        query_metrics.append(
            {
                'query_id': query['query_id'],
                'rule_relevant_recall': _set_recall(rule_ranked, relevant_ids),
                'learned_relevant_recall': _set_recall(learned_ranked, relevant_ids),
                'rule_useful_recall': _set_recall(rule_ranked, useful_ids),
                'learned_useful_recall': _set_recall(learned_ranked, useful_ids),
                'has_useful': bool(useful_ids),
            }
        )

    relevant_rule = _mean_metric(query_metrics, 'rule_relevant_recall')
    relevant_learned = _mean_metric(query_metrics, 'learned_relevant_recall')
    useful_query_metrics = [row for row in query_metrics if row['has_useful']]
    useful_rule = _mean_metric(useful_query_metrics, 'rule_useful_recall')
    useful_learned = _mean_metric(useful_query_metrics, 'learned_useful_recall')
    relevant_gain = relevant_learned - relevant_rule
    useful_gain = useful_learned - useful_rule
    return {
        'split': 'test',
        'top_k': int(top_k),
        'num_test_queries': len(query_metrics),
        'num_test_queries_with_useful': len(useful_query_metrics),
        'rule': {
            'relevant_recall_at_k': relevant_rule,
            'useful_recall_at_k': useful_rule,
        },
        'learned': {
            'relevant_recall_at_k': relevant_learned,
            'useful_recall_at_k': useful_learned,
        },
        'gains': {
            'relevant_recall_at_k': relevant_gain,
            'useful_recall_at_k': useful_gain,
        },
        'thresholds': {
            'recall_gain': 0.02,
        },
        'done_conditions': {
            'relevant_recall_improved': bool(relevant_gain >= 0.02),
            'useful_recall_improved': bool(useful_gain >= 0.02),
        },
    }


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _stable_noise(query_id: str, memory_id: str, salt: str) -> float:
    import hashlib

    digest = hashlib.md5(f'{query_id}|{memory_id}|{salt}'.encode('utf-8')).hexdigest()
    raw = int(digest[:8], 16) / 0xFFFFFFFF
    return 2.0 * raw - 1.0


def _tokenize(text: str) -> list[str]:
    cleaned = text.lower().replace('?', ' ').replace(':', ' ').replace('(', ' ').replace(')', ' ')
    return [token for token in cleaned.replace("'s", ' ').split() if token]


def _pair_feature_dict(
    query: dict[str, Any],
    memory: dict[str, Any],
    tfidf_score: float,
    observable_context: dict[str, Any],
) -> dict[str, Any]:
    age = max(0, int(query['query_time']) - int(memory['write_time']))
    recency_score = 1.0 / (age + 1.0)
    query_tokens = set(_tokenize(query['query_text']))
    memory_tokens = set(_tokenize(memory['memory_text']))
    entity_tokens = set(_tokenize(query['entity']))
    alias_tokens = set(_tokenize(memory['entity_alias']))
    slot_tokens = set(_tokenize(query['slot'].replace('_', ' ')))
    canonical_tokens = set(_tokenize(str(memory['value_canonical'])))
    overlap = len(query_tokens & memory_tokens)
    union = max(len(query_tokens | memory_tokens), 1)
    entity_overlap = len(entity_tokens & memory_tokens)
    alias_overlap = len(entity_tokens & alias_tokens)
    slot_overlap = len(slot_tokens & memory_tokens)
    value_overlap = len(canonical_tokens & query_tokens)

    group_size = max(int(observable_context.get('observed_group_size', 0)), 1)
    same_value_count = int(observable_context.get('observed_same_value_count', 0))
    different_value_count = int(observable_context.get('observed_different_value_count', 0))
    newer_count = int(observable_context.get('observed_newer_count', 0))
    newer_same_value_count = int(observable_context.get('observed_newer_same_value_count', 0))
    newer_different_value_count = int(observable_context.get('observed_newer_different_value_count', 0))
    older_count = int(observable_context.get('observed_older_count', 0))
    same_source_count = int(observable_context.get('observed_same_source_count', 0))
    same_source_same_value_count = int(observable_context.get('observed_same_source_same_value_count', 0))
    same_source_different_value_count = int(observable_context.get('observed_same_source_different_value_count', 0))
    distinct_value_count = int(observable_context.get('observed_distinct_value_count', 0))

    same_value_fraction = same_value_count / group_size
    different_value_fraction = different_value_count / group_size
    newer_fraction = newer_count / group_size
    newer_same_value_fraction = newer_same_value_count / max(newer_count, 1)
    newer_different_value_fraction = newer_different_value_count / max(newer_count, 1)
    older_fraction = older_count / group_size
    same_source_consistency = same_source_same_value_count / max(same_source_count, 1)
    same_source_conflict_rate = same_source_different_value_count / max(same_source_count, 1)
    support_margin = (same_value_count - different_value_count) / group_size
    staleness_pressure = newer_different_value_count / group_size
    distinct_value_density = distinct_value_count / group_size
    age_conflict_interaction = age * float(observable_context.get('observed_conflict_ratio', 0.0))
    age_staleness_interaction = age * staleness_pressure

    return {
        'query_slot': query['slot'],
        'memory_slot': memory['slot'],
        'query_stress_name': query['stress_name'],
        'memory_source_id': memory['source_id'],
        'tfidf_score': float(tfidf_score),
        'recency_score': recency_score,
        'age': age,
        'age_squared': age * age,
        'same_entity': int(memory['entity'] == query['entity']),
        'same_slot': int(memory['slot'] == query['slot']),
        'entity_match_strength': entity_match_strength(query['entity'], memory['entity']),
        'query_memory_token_overlap': overlap,
        'query_memory_token_jaccard': overlap / union,
        'query_entity_token_overlap': entity_overlap,
        'query_alias_token_overlap': alias_overlap,
        'query_slot_token_overlap': slot_overlap,
        'value_query_token_overlap': value_overlap,
        'source_quality': float(memory['source_quality']),
        'stress_value': float(query['stress_value']),
        'query_lag': int(query['query_lag']),
        'world_change_scale': float(query['world_change_scale']),
        'write_error_rate': float(query['write_error_rate']),
        'conflict_rate': float(query['conflict_rate']),
        'distractor_overlap': float(query['distractor_overlap']),
        'memory_raw_length': len(str(memory['value_raw'])),
        'memory_canonical_length': len(str(memory['value_canonical'])),
        'memory_canonical_token_count': len(canonical_tokens),
        'same_value_fraction': same_value_fraction,
        'different_value_fraction': different_value_fraction,
        'newer_fraction': newer_fraction,
        'newer_same_value_fraction': newer_same_value_fraction,
        'newer_different_value_fraction': newer_different_value_fraction,
        'older_fraction': older_fraction,
        'same_source_consistency': same_source_consistency,
        'same_source_conflict_rate': same_source_conflict_rate,
        'support_margin': support_margin,
        'staleness_pressure': staleness_pressure,
        'distinct_value_density': distinct_value_density,
        'age_conflict_interaction': age_conflict_interaction,
        'age_staleness_interaction': age_staleness_interaction,
        **observable_context,
    }


def _build_relevance_examples(
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indices = _build_episode_indices(memories, queries, labels)
    examples: list[dict[str, Any]] = []
    for query in queries:
        index = indices[query['episode_id']]
        label = index.labels_by_query[query['query_id']]
        available_memories = [memory for memory in index.memories if int(memory['write_time']) <= int(query['query_time'])]
        contexts = build_query_memory_contexts(available_memories, query)
        raw_tfidf = _tfidf_similarity(index, query, available_memories)
        normalized_tfidf = normalized_similarity(raw_tfidf)
        relevant_ids = set(label['relevant_memory_ids'])
        valid_ids = set(label['valid_memory_ids'])
        useful_ids = set(label['useful_memory_ids'])
        for memory in available_memories:
            features = _pair_feature_dict(
                query,
                memory,
                normalized_tfidf.get(memory['memory_id'], 0.0),
                contexts.get(memory['memory_id'], {}),
            )
            examples.append(
                {
                    'episode_id': query['episode_id'],
                    'query_id': query['query_id'],
                    'memory_id': memory['memory_id'],
                    **features,
                    'label': int(memory['memory_id'] in relevant_ids),
                    'valid_label': int(memory['memory_id'] in valid_ids),
                    'useful_label': int(memory['memory_id'] in useful_ids),
                }
            )
    if not examples:
        raise RuntimeError('Relevance examples could not be created.')
    return examples


def _fit_relevance_with_splits(
    examples: list[dict[str, Any]],
    episode_splits: dict[str, list[str]],
) -> tuple[Pipeline, dict[str, Any]]:
    split_frames = {
        split_name: [row for row in examples if row['episode_id'] in episode_ids]
        for split_name, episode_ids in episode_splits.items()
    }
    train_rows = split_frames['train']
    train_features = [_strip_metadata(row) for row in train_rows]
    train_labels = [int(row['label']) for row in train_rows]
    pipeline = _train_binary_pipeline(train_features, train_labels)

    summary: dict[str, Any] = {
        'split_strategy': 'episode_level',
        'task': 'relevance',
        'num_examples': len(examples),
        'num_episodes': len({row['episode_id'] for row in examples}),
        'splits': {},
    }
    for split_name, rows in split_frames.items():
        features = [_strip_metadata(row) for row in rows]
        labels = [int(row['label']) for row in rows]
        useful_labels = [int(row['useful_label']) for row in rows]
        metrics = _evaluate_pipeline(pipeline, features, labels)
        summary['splits'][split_name] = {
            'num_examples': len(rows),
            'num_episodes': len({row['episode_id'] for row in rows}),
            'positive_rate': float(np.mean(labels)) if labels else None,
            'useful_positive_rate': float(np.mean(useful_labels)) if useful_labels else None,
            **metrics,
        }
    summary['selected_checkpoint'] = 'train_only_logreg'
    return pipeline, summary


def _train_binary_pipeline(feature_dicts: list[dict[str, Any]], labels: list[int]) -> Pipeline:
    if len(set(labels)) <= 1:
        classifier = DummyClassifier(strategy='constant', constant=labels[0])
    else:
        classifier = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=0)
    pipeline = Pipeline(
        steps=[
            ('vectorizer', DictVectorizer(sparse=True)),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', classifier),
        ]
    )
    pipeline.fit(feature_dicts, labels)
    return pipeline


def _evaluate_pipeline(pipeline: Pipeline, feature_dicts: list[dict[str, Any]], labels: list[int]) -> dict[str, Any]:
    if not feature_dicts:
        return {
            'accuracy': None,
            'mean_probability': None,
            'roc_auc': None,
            'average_precision': None,
        }
    probabilities = pipeline.predict_proba(feature_dicts)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'mean_probability': float(np.mean(probabilities)),
        'roc_auc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else None,
        'average_precision': float(average_precision_score(labels, probabilities)) if len(set(labels)) > 1 else None,
    }


def _strip_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {'episode_id', 'query_id', 'memory_id', 'label', 'valid_label', 'useful_label'}
    }


def _build_episode_indices(
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> dict[str, EpisodeIndex]:
    grouped_memories: dict[str, list[dict[str, Any]]] = {}
    grouped_queries: dict[str, list[dict[str, Any]]] = {}
    grouped_labels: dict[str, dict[str, dict[str, Any]]] = {}
    query_episode_lookup = {query['query_id']: query['episode_id'] for query in queries}
    for memory in memories:
        grouped_memories.setdefault(memory['episode_id'], []).append(memory)
    for query in queries:
        grouped_queries.setdefault(query['episode_id'], []).append(query)
    for label in labels:
        episode_id = query_episode_lookup[label['query_id']]
        grouped_labels.setdefault(episode_id, {})[label['query_id']] = label

    indices: dict[str, EpisodeIndex] = {}
    for episode_id, memory_rows in grouped_memories.items():
        memory_rows = sorted(memory_rows, key=lambda row: row['memory_id'])
        query_rows = grouped_queries.get(episode_id, [])
        corpus = [memory['memory_text'] for memory in memory_rows] + [query['query_text'] for query in query_rows]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        all_matrix = vectorizer.fit_transform(corpus)
        memory_matrix = all_matrix[: len(memory_rows)]
        indices[episode_id] = EpisodeIndex(
            episode_id=episode_id,
            memories=memory_rows,
            labels_by_query=grouped_labels.get(episode_id, {}),
            vectorizer=vectorizer,
            memory_matrix=memory_matrix,
            memory_positions={memory['memory_id']: idx for idx, memory in enumerate(memory_rows)},
        )
    return indices


def _tfidf_similarity(index: EpisodeIndex, query: dict[str, Any], available_memories: list[dict[str, Any]]) -> dict[str, float]:
    if not available_memories:
        return {}
    query_vector = index.vectorizer.transform([query['query_text']])
    positions = [index.memory_positions[memory['memory_id']] for memory in available_memories]
    sims = index.memory_matrix[positions].dot(query_vector.T).toarray().ravel()
    return {memory['memory_id']: float(score) for memory, score in zip(available_memories, sims, strict=False)}


def _set_recall(ranked_memories: list[dict[str, Any]], target_ids: set[str]) -> float:
    if not target_ids:
        return 1.0
    retrieved_ids = {memory['memory_id'] for memory in ranked_memories}
    return float(len(retrieved_ids & target_ids) / len(target_ids))


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row[key]) for row in rows]))

