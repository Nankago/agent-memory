from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.pipeline import Pipeline

from memory_collapse.estimators import SPLIT_FILENAME, load_episode_splits, resolve_models_dir, split_episode_ids
from memory_collapse.io_utils import ensure_dir, read_jsonl, write_csv
from memory_collapse.query_validity import augment_query_validity_feature_dict
from memory_collapse.relevance import _build_relevance_examples, _evaluate_pipeline, _strip_metadata, _train_binary_pipeline


ANTI_SUPPORT_MODEL_FILENAME = 'anti_support_model.pkl'
ANTI_SUPPORT_SUMMARY_FILENAME = 'anti_support_summary.json'
ANTI_SUPPORT_EXAMPLES_FILENAME = 'anti_support_training_examples.csv'


@dataclass
class LearnedAntiSupportBundle:
    pipeline: Pipeline
    condition: str

    def predict_anti_support(self, feature_dict: dict[str, Any]) -> float:
        features = [augment_query_validity_feature_dict(dict(feature_dict))]
        return float(self.pipeline.predict_proba(features)[0, 1])


def anti_support_artifact_paths(path: str | Path) -> dict[str, str]:
    models_dir = resolve_models_dir(path)
    return {
        'anti_support_model': str(models_dir / ANTI_SUPPORT_MODEL_FILENAME),
        'anti_support_summary': str(models_dir / ANTI_SUPPORT_SUMMARY_FILENAME),
        'anti_support_examples': str(models_dir / ANTI_SUPPORT_EXAMPLES_FILENAME),
        'splits': str(models_dir / SPLIT_FILENAME),
    }


def anti_support_model_exists(path: str | Path) -> bool:
    return Path(anti_support_artifact_paths(path)['anti_support_model']).exists()


def load_anti_support_bundle(path: str | Path) -> LearnedAntiSupportBundle:
    models_dir = resolve_models_dir(path)
    with (models_dir / ANTI_SUPPORT_MODEL_FILENAME).open('rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return LearnedAntiSupportBundle(
            pipeline=payload['pipeline'],
            condition=payload.get('condition', 'relevant_only'),
        )
    return LearnedAntiSupportBundle(pipeline=payload, condition='relevant_only')


def train_anti_support_model(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    run_path = Path(run_dir)
    models_dir = ensure_dir(resolve_models_dir(output_dir or run_path))

    memories = read_jsonl(run_path / 'data' / 'memories.jsonl')
    queries = read_jsonl(run_path / 'data' / 'queries.jsonl')
    labels = read_jsonl(run_path / 'data' / 'exact_labels.jsonl')
    if not memories or not queries or not labels:
        raise RuntimeError('Anti-support training requires generated memories, queries, and exact labels.')

    config_snapshot = {}
    config_path = run_path / 'config_snapshot.yaml'
    if config_path.exists():
        config_snapshot = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    seed = int(config_snapshot.get('dataset', {}).get('seed', 0))

    examples = _build_anti_support_examples(memories, queries, labels)
    if not examples:
        raise RuntimeError('Anti-support examples could not be created.')

    splits_path = models_dir / SPLIT_FILENAME
    if splits_path.exists():
        episode_splits = load_episode_splits(models_dir)
    else:
        episode_splits = split_episode_ids(sorted({row['episode_id'] for row in examples}), seed)
        with splits_path.open('w', encoding='utf-8') as handle:
            json.dump(episode_splits, handle, indent=2)

    pipeline, summary = _fit_anti_support_with_splits(examples, episode_splits)
    with (models_dir / ANTI_SUPPORT_MODEL_FILENAME).open('wb') as handle:
        pickle.dump({'pipeline': pipeline, 'condition': 'relevant_only'}, handle)
    with (models_dir / ANTI_SUPPORT_SUMMARY_FILENAME).open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    write_csv(models_dir / ANTI_SUPPORT_EXAMPLES_FILENAME, examples)
    return anti_support_artifact_paths(models_dir)


def _build_anti_support_examples(
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pair_examples = _build_relevance_examples(memories, queries, labels)
    memory_by_id = {memory['memory_id']: memory for memory in memories}
    gold_by_query = {query['query_id']: query['gold_value'] for query in queries}
    examples: list[dict[str, Any]] = []
    for row in pair_examples:
        if int(row['label']) != 1:
            continue
        memory = memory_by_id[row['memory_id']]
        gold_value = gold_by_query[row['query_id']]
        contradiction = int(
            row['useful_label'] == 0
            and memory['value_canonical'] != gold_value
        )
        feature_row = augment_query_validity_feature_dict(_strip_metadata(row))
        examples.append(
            {
                'episode_id': row['episode_id'],
                'query_id': row['query_id'],
                'memory_id': row['memory_id'],
                **feature_row,
                'label': contradiction,
            }
        )
    return examples


def _fit_anti_support_with_splits(
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
        'task': 'anti_support',
        'condition': 'relevant_only',
        'feature_builder': 'query_validity_augmented_features',
        'num_examples': len(examples),
        'num_episodes': len({row['episode_id'] for row in examples}),
        'splits': {},
    }
    for split_name, rows in split_frames.items():
        features = [_strip_metadata(row) for row in rows]
        labels = [int(row['label']) for row in rows]
        metrics = _evaluate_pipeline(pipeline, features, labels)
        summary['splits'][split_name] = {
            'num_examples': len(rows),
            'num_episodes': len({row['episode_id'] for row in rows}),
            'positive_rate': float(np.mean(labels)) if labels else None,
            **metrics,
        }
    summary['selected_checkpoint'] = 'train_only_logreg'
    return pipeline, summary
