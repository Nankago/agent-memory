from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.pipeline import Pipeline

from memory_collapse.domain import list_slot_specs
from memory_collapse.estimators import SPLIT_FILENAME, load_episode_splits, resolve_models_dir, split_episode_ids
from memory_collapse.io_utils import ensure_dir, read_jsonl, write_csv
from memory_collapse.relevance import _build_relevance_examples, _evaluate_pipeline, _pair_feature_dict, _strip_metadata, _train_binary_pipeline


TARGET_CONFIGS = {
    'useful_label': {
        'name': 'useful',
        'condition': 'all_pairs',
    },
    'valid_label': {
        'name': 'valid',
        'condition': 'relevant_only',
    },
}
SLOT_VOLATILITY = {slot.name: slot.volatility for slot in list_slot_specs()}


@dataclass
class LearnedQueryValidityBundle:
    pipeline: Pipeline
    target_label: str
    condition: str

    def predict_query_validity(
        self,
        query: dict[str, Any],
        memory: dict[str, Any],
        tfidf_score: float,
        observable_context: dict[str, Any] | None = None,
    ) -> float:
        features = [build_query_validity_feature_dict(query, memory, tfidf_score, observable_context or {})]
        return float(self.pipeline.predict_proba(features)[0, 1])


def _artifact_names(target_label: str) -> tuple[str, str, str]:
    suffix = TARGET_CONFIGS[target_label]['name']
    return (
        f'query_validity_{suffix}_model.pkl',
        f'query_validity_{suffix}_summary.json',
        f'query_validity_{suffix}_training_examples.csv',
    )


def query_validity_artifact_paths(path: str | Path, target_label: str | None = None) -> dict[str, str]:
    models_dir = resolve_models_dir(path)
    labels = [target_label] if target_label else list(TARGET_CONFIGS)
    artifacts: dict[str, str] = {'splits': str(models_dir / SPLIT_FILENAME)}
    for label in labels:
        model_name, summary_name, examples_name = _artifact_names(label)
        prefix = TARGET_CONFIGS[label]['name']
        artifacts[f'query_validity_{prefix}_model'] = str(models_dir / model_name)
        artifacts[f'query_validity_{prefix}_summary'] = str(models_dir / summary_name)
        artifacts[f'query_validity_{prefix}_examples'] = str(models_dir / examples_name)
        if label == 'useful_label':
            artifacts['query_validity_model'] = str(models_dir / model_name)
            artifacts['query_validity_summary'] = str(models_dir / summary_name)
            artifacts['query_validity_examples'] = str(models_dir / examples_name)
    return artifacts


def query_validity_model_exists(path: str | Path, target_label: str | None = None) -> bool:
    artifacts = query_validity_artifact_paths(path, target_label=target_label)
    model_keys = [key for key in artifacts if key.endswith('_model')]
    return all(Path(artifacts[key]).exists() for key in model_keys)


def load_query_validity_bundle(path: str | Path, target_label: str = 'useful_label') -> LearnedQueryValidityBundle:
    if target_label not in TARGET_CONFIGS:
        raise KeyError(f'Unsupported query-validity target label: {target_label}')
    models_dir = resolve_models_dir(path)
    model_name, _, _ = _artifact_names(target_label)
    with (models_dir / model_name).open('rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return LearnedQueryValidityBundle(
            pipeline=payload['pipeline'],
            target_label=payload.get('target_label', target_label),
            condition=payload.get('condition', TARGET_CONFIGS[target_label]['condition']),
        )
    return LearnedQueryValidityBundle(
        pipeline=payload,
        target_label=target_label,
        condition=TARGET_CONFIGS[target_label]['condition'],
    )


def train_query_validity_models(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    run_path = Path(run_dir)
    models_dir = ensure_dir(resolve_models_dir(output_dir or run_path))

    memories = read_jsonl(run_path / 'data' / 'memories.jsonl')
    queries = read_jsonl(run_path / 'data' / 'queries.jsonl')
    labels = read_jsonl(run_path / 'data' / 'exact_labels.jsonl')
    if not memories or not queries or not labels:
        raise RuntimeError('Query-validity training requires generated memories, queries, and exact labels.')

    config_snapshot = {}
    config_path = run_path / 'config_snapshot.yaml'
    if config_path.exists():
        config_snapshot = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    seed = int(config_snapshot.get('dataset', {}).get('seed', 0))

    pair_examples = _build_relevance_examples(memories, queries, labels)
    if not pair_examples:
        raise RuntimeError('Query-validity examples could not be created.')

    splits_path = models_dir / SPLIT_FILENAME
    if splits_path.exists():
        episode_splits = load_episode_splits(models_dir)
    else:
        episode_splits = split_episode_ids(sorted({row['episode_id'] for row in pair_examples}), seed)
        with splits_path.open('w', encoding='utf-8') as handle:
            json.dump(episode_splits, handle, indent=2)

    combined_artifacts = query_validity_artifact_paths(models_dir)
    for target_label, config in TARGET_CONFIGS.items():
        examples = _select_examples(pair_examples, target_label)
        pipeline, summary = _fit_query_validity_with_splits(examples, episode_splits, target_label=target_label)
        model_name, summary_name, examples_name = _artifact_names(target_label)
        with (models_dir / model_name).open('wb') as handle:
            pickle.dump(
                {
                    'pipeline': pipeline,
                    'target_label': target_label,
                    'condition': config['condition'],
                },
                handle,
            )
        with (models_dir / summary_name).open('w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)
        write_csv(models_dir / examples_name, examples)
    return combined_artifacts


def build_query_validity_feature_dict(
    query: dict[str, Any],
    memory: dict[str, Any],
    tfidf_score: float,
    observable_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = _pair_feature_dict(query, memory, tfidf_score, observable_context or {})
    return augment_query_validity_feature_dict(base)


def augment_query_validity_feature_dict(base_features: dict[str, Any]) -> dict[str, Any]:
    features = dict(base_features)
    age = float(features.get('age', 0.0))
    source_quality = float(features.get('source_quality', 0.0))
    stress_value = float(features.get('stress_value', 0.0))
    same_entity = float(features.get('same_entity', 0.0))
    same_slot = float(features.get('same_slot', 0.0))
    same_value_fraction = float(features.get('same_value_fraction', 0.0))
    different_value_fraction = float(features.get('different_value_fraction', 0.0))
    newer_fraction = float(features.get('newer_fraction', 0.0))
    newer_same_value_fraction = float(features.get('newer_same_value_fraction', 0.0))
    newer_different_value_fraction = float(features.get('newer_different_value_fraction', 0.0))
    same_source_consistency = float(features.get('same_source_consistency', 0.0))
    same_source_conflict_rate = float(features.get('same_source_conflict_rate', 0.0))
    support_margin = float(features.get('support_margin', 0.0))
    staleness_pressure = float(features.get('staleness_pressure', 0.0))
    conflict_ratio = float(features.get('observed_conflict_ratio', 0.0))
    observed_is_latest = float(features.get('observed_is_latest', 0.0))
    observed_group_size = float(features.get('observed_group_size', 0.0))
    observed_same_value_count = float(features.get('observed_same_value_count', 0.0))
    observed_different_value_count = float(features.get('observed_different_value_count', 0.0))
    observed_newer_different_count = float(features.get('observed_newer_different_value_count', 0.0))
    observed_newer_same_count = float(features.get('observed_newer_same_value_count', 0.0))
    observed_same_source_same_count = float(features.get('observed_same_source_same_value_count', 0.0))
    observed_same_source_different_count = float(features.get('observed_same_source_different_value_count', 0.0))
    slot_name = str(features.get('memory_slot', features.get('query_slot', '')))
    slot_volatility = float(SLOT_VOLATILITY.get(slot_name, 0.05))
    hazard = slot_volatility * (0.85 + stress_value)
    heuristic_c_hat = _clip(source_quality * 0.95 + 0.03, 0.02, 0.99)
    heuristic_s_hat = _clip(math.exp(-hazard * age), 0.02, 0.99)
    heuristic_pi_hat = _clip(heuristic_c_hat * heuristic_s_hat, 0.02, 0.98)

    features.update(
        {
            'slot_volatility': slot_volatility,
            'slot_stress_hazard': hazard,
            'heuristic_c_hat': heuristic_c_hat,
            'heuristic_s_hat': heuristic_s_hat,
            'heuristic_pi_hat': heuristic_pi_hat,
            'observed_support_balance': observed_same_value_count - observed_different_value_count,
            'observed_newer_support_balance': observed_newer_same_count - observed_newer_different_count,
            'observed_same_source_balance': observed_same_source_same_count - observed_same_source_different_count,
            'stale_conflict_pressure': newer_different_value_fraction * (1.0 + conflict_ratio),
            'support_conflict_tension': support_margin - conflict_ratio,
            'source_conflict_tension': same_source_consistency - same_source_conflict_rate,
            'age_times_source_quality': age * source_quality,
            'age_times_newer_fraction': age * newer_fraction,
            'age_times_newer_different_fraction': age * newer_different_value_fraction,
            'latest_conflict_agreement': observed_is_latest * (1.0 - conflict_ratio),
            'not_latest_conflict_pressure': (1.0 - observed_is_latest) * conflict_ratio,
            'slot_match_strength': same_entity * same_slot,
            'observed_density': observed_group_size / max(age + 1.0, 1.0),
            'same_value_vs_stale_margin': same_value_fraction - staleness_pressure,
            'same_source_vs_conflict_margin': same_source_consistency - conflict_ratio,
            'heuristic_pi_minus_staleness': heuristic_pi_hat - staleness_pressure,
        }
    )
    return features


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _select_examples(pair_examples: list[dict[str, Any]], target_label: str) -> list[dict[str, Any]]:
    if target_label == 'useful_label':
        return pair_examples
    if target_label == 'valid_label':
        return [row for row in pair_examples if int(row['label']) == 1]
    raise KeyError(f'Unsupported query-validity target label: {target_label}')


def _prepare_query_validity_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [augment_query_validity_feature_dict(_strip_metadata(row)) for row in rows]


def _fit_query_validity_with_splits(
    examples: list[dict[str, Any]],
    episode_splits: dict[str, list[str]],
    target_label: str,
) -> tuple[Pipeline, dict[str, Any]]:
    split_frames = {
        split_name: [row for row in examples if row['episode_id'] in episode_ids]
        for split_name, episode_ids in episode_splits.items()
    }
    train_rows = split_frames['train']
    train_features = _prepare_query_validity_features(train_rows)
    train_labels = [int(row[target_label]) for row in train_rows]
    pipeline = _train_binary_pipeline(train_features, train_labels)

    summary: dict[str, Any] = {
        'split_strategy': 'episode_level',
        'task': 'query_time_validity',
        'condition': TARGET_CONFIGS[target_label]['condition'],
        'target_label': target_label,
        'feature_builder': 'pair_plus_heuristic_validity_features',
        'num_examples': len(examples),
        'num_episodes': len({row['episode_id'] for row in examples}),
        'splits': {},
    }
    for split_name, rows in split_frames.items():
        features = _prepare_query_validity_features(rows)
        labels = [int(row[target_label]) for row in rows]
        metrics = _evaluate_pipeline(pipeline, features, labels)
        summary['splits'][split_name] = {
            'num_examples': len(rows),
            'num_episodes': len({row['episode_id'] for row in rows}),
            'positive_rate': float(np.mean(labels)) if labels else None,
            **metrics,
        }
    summary['selected_checkpoint'] = 'train_only_logreg'
    return pipeline, summary
