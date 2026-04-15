from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from memory_collapse.anti_support import load_anti_support_bundle
from memory_collapse.estimators import SPLIT_FILENAME, build_query_memory_contexts, load_episode_splits, resolve_models_dir, split_episode_ids
from memory_collapse.io_utils import ensure_dir, read_jsonl, write_csv
from memory_collapse.query_validity import build_query_validity_feature_dict, load_query_validity_bundle
from memory_collapse.relevance import _build_episode_indices, _tfidf_similarity, load_relevance_bundle, normalized_similarity


VALUE_RESOLVER_MODEL_FILENAME = 'value_resolver_model.pkl'
VALUE_RESOLVER_SUMMARY_FILENAME = 'value_resolver_summary.json'
VALUE_RESOLVER_EXAMPLES_FILENAME = 'value_resolver_training_examples.csv'
VALUE_RESOLVER_METADATA_FIELDS = {'episode_id', 'query_id', 'candidate_value', 'label'}
VALUE_RESOLVER_SELECTION_TOP1_TOLERANCE = 0.015
VALUE_RESOLVER_BOOTSTRAP_RESAMPLES = 200
VALUE_RESOLVER_FEATURE_GROUPS = {
    'support': (
        'candidate_support_sum',
        'candidate_support_max',
        'candidate_support_mean',
        'candidate_support_count',
        'candidate_support_share',
        'candidate_rho_mean',
        'candidate_pi_mean',
        'support_margin_vs_rival',
        'adjusted_support_margin',
        'external_support_sum',
        'num_candidate_values',
    ),
    'freshness': (
        'candidate_freshest_age',
        'candidate_mean_age',
        'candidate_is_global_newest',
        'candidate_newest_gap',
        'query_lag',
    ),
    'anti_support': (
        'candidate_self_anti_sum',
        'candidate_self_anti_mean',
        'candidate_contradiction_sum',
        'external_anti_sum',
        'external_anti_share',
    ),
    'context': (
        'query_stress_value',
        'query_conflict_rate',
        'query_world_change_scale',
    ),
}
ALL_VALUE_RESOLVER_FEATURES = tuple(
    feature_name
    for group in VALUE_RESOLVER_FEATURE_GROUPS.values()
    for feature_name in group
)


@dataclass
class LinearValueResolverRanker:
    weights: np.ndarray
    bias: float = 0.0

    def decision_function(self, matrix: Any) -> np.ndarray:
        return np.asarray(matrix @ self.weights).ravel() + float(self.bias)


@dataclass
class LearnedValueResolverBundle:
    condition: str
    objective: str
    pipeline: Pipeline | None = None
    vectorizer: DictVectorizer | None = None
    scaler: StandardScaler | None = None
    classifier: Any | None = None
    active_feature_names: tuple[str, ...] | None = None

    def predict_candidate_scores(self, feature_rows: list[dict[str, Any]]) -> list[float]:
        if not feature_rows:
            return []
        projected_rows = [_candidate_feature_dict(row, self.active_feature_names) for row in feature_rows]
        if self.pipeline is not None:
            return [float(score) for score in self.pipeline.predict_proba(projected_rows)[:, 1]]
        if self.vectorizer is None or self.scaler is None or self.classifier is None:
            raise RuntimeError('Value resolver bundle is missing ranking components.')
        transformed = self.vectorizer.transform(projected_rows)
        transformed = self.scaler.transform(transformed)
        scores = np.asarray(self.classifier.decision_function(transformed)).ravel()
        return [float(score) for score in scores]


def value_resolver_artifact_paths(path: str | Path) -> dict[str, str]:
    models_dir = resolve_models_dir(path)
    return {
        'value_resolver_model': str(models_dir / VALUE_RESOLVER_MODEL_FILENAME),
        'value_resolver_summary': str(models_dir / VALUE_RESOLVER_SUMMARY_FILENAME),
        'value_resolver_examples': str(models_dir / VALUE_RESOLVER_EXAMPLES_FILENAME),
        'splits': str(models_dir / SPLIT_FILENAME),
    }


def value_resolver_model_exists(path: str | Path) -> bool:
    return Path(value_resolver_artifact_paths(path)['value_resolver_model']).exists()


def load_value_resolver_bundle(path: str | Path) -> LearnedValueResolverBundle:
    models_dir = resolve_models_dir(path)
    with (models_dir / VALUE_RESOLVER_MODEL_FILENAME).open('rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        return LearnedValueResolverBundle(
            condition=payload.get('condition', 'top_k_direct_valid_candidates'),
            objective=payload.get('objective', 'candidate_binary_classification'),
            pipeline=payload.get('pipeline'),
            vectorizer=payload.get('vectorizer'),
            scaler=payload.get('scaler'),
            classifier=payload.get('classifier'),
            active_feature_names=tuple(payload.get('active_feature_names', ())) or None,
        )
    return LearnedValueResolverBundle(
        condition='top_k_direct_valid_candidates',
        objective='candidate_binary_classification',
        pipeline=payload,
        active_feature_names=None,
    )


def train_value_resolver(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    run_path = Path(run_dir)
    models_dir = ensure_dir(resolve_models_dir(output_dir or run_path))

    memories = read_jsonl(run_path / 'data' / 'memories.jsonl')
    queries = read_jsonl(run_path / 'data' / 'queries.jsonl')
    labels = read_jsonl(run_path / 'data' / 'exact_labels.jsonl')
    if not memories or not queries or not labels:
        raise RuntimeError('Value-resolver training requires generated memories, queries, and exact labels.')

    config_snapshot = {}
    config_path = run_path / 'config_snapshot.yaml'
    if config_path.exists():
        config_snapshot = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    seed = int(config_snapshot.get('dataset', {}).get('seed', 0))
    top_k = int(config_snapshot.get('dataset', {}).get('top_k', 12))

    examples, query_stats = _build_value_resolver_examples(run_path, models_dir, memories, queries, labels, top_k=top_k)
    if not examples:
        raise RuntimeError('Value-resolver examples could not be created.')

    splits_path = models_dir / SPLIT_FILENAME
    if splits_path.exists():
        episode_splits = load_episode_splits(models_dir)
    else:
        episode_splits = split_episode_ids(sorted({row['episode_id'] for row in examples}), seed)
        with splits_path.open('w', encoding='utf-8') as handle:
            json.dump(episode_splits, handle, indent=2)

    bundle, summary = _fit_value_resolver_with_splits(examples, episode_splits, query_stats)
    with (models_dir / VALUE_RESOLVER_MODEL_FILENAME).open('wb') as handle:
        pickle.dump(
            {
                'condition': bundle.condition,
                'objective': bundle.objective,
                'pipeline': bundle.pipeline,
                'vectorizer': bundle.vectorizer,
                'scaler': bundle.scaler,
                'classifier': bundle.classifier,
                'active_feature_names': list(bundle.active_feature_names or ()),
            },
            handle,
        )
    with (models_dir / VALUE_RESOLVER_SUMMARY_FILENAME).open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    write_csv(models_dir / VALUE_RESOLVER_EXAMPLES_FILENAME, examples)
    return value_resolver_artifact_paths(models_dir)


def build_value_candidate_feature_rows(
    query: dict[str, Any],
    fallback_pool: list[dict[str, Any]],
    component_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not fallback_pool:
        return []
    candidate_values = sorted({memory['value_canonical'] for memory in fallback_pool})
    by_value: dict[str, list[dict[str, Any]]] = {value: [] for value in candidate_values}
    for memory in fallback_pool:
        by_value[memory['value_canonical']].append(memory)

    total_support = sum(float(component_lookup[memory['memory_id']]['u_hat']) for memory in fallback_pool)
    total_contradiction = sum(
        float(component_lookup[memory['memory_id']].get('anti_support_hat', 0.0))
        * float(component_lookup[memory['memory_id']]['u_hat'])
        for memory in fallback_pool
    )
    global_newest_time = max(int(memory['write_time']) for memory in fallback_pool)

    rows: list[dict[str, Any]] = []
    value_support_sums = {
        value: sum(float(component_lookup[memory['memory_id']]['u_hat']) for memory in memories)
        for value, memories in by_value.items()
    }
    for value, memories_for_value in by_value.items():
        supports = [float(component_lookup[memory['memory_id']]['u_hat']) for memory in memories_for_value]
        pis = [float(component_lookup[memory['memory_id']].get('pi_hat_direct', 0.0)) for memory in memories_for_value]
        rhos = [float(component_lookup[memory['memory_id']].get('rho_hat', 0.0)) for memory in memories_for_value]
        anti_scores = [
            float(component_lookup[memory['memory_id']].get('anti_support_hat', 0.0))
            for memory in memories_for_value
        ]
        contradiction_scores = [
            float(component_lookup[memory['memory_id']].get('contradiction_mass', 0.0))
            for memory in memories_for_value
        ]
        ages = [max(0, int(query['query_time']) - int(memory['write_time'])) for memory in memories_for_value]
        freshest_age = min(ages)
        newest_write = max(int(memory['write_time']) for memory in memories_for_value)
        external_memories = [memory for memory in fallback_pool if memory['value_canonical'] != value]
        external_support_sum = sum(float(component_lookup[memory['memory_id']]['u_hat']) for memory in external_memories)
        external_anti_sum = sum(
            float(component_lookup[memory['memory_id']].get('anti_support_hat', 0.0))
            * float(component_lookup[memory['memory_id']]['u_hat'])
            for memory in external_memories
        )
        rival_support = max([score for other_value, score in value_support_sums.items() if other_value != value] or [0.0])
        support_sum = sum(supports)
        support_count = len(supports)
        self_anti_sum = sum(score * support for score, support in zip(anti_scores, supports, strict=False))
        contradiction_sum = sum(score * support for score, support in zip(contradiction_scores, supports, strict=False))
        rows.append(
            {
                'query_id': query['query_id'],
                'episode_id': query['episode_id'],
                'candidate_value': value,
                'num_candidate_values': len(candidate_values),
                'candidate_support_sum': support_sum,
                'candidate_support_max': max(supports),
                'candidate_support_mean': float(np.mean(supports)),
                'candidate_support_count': support_count,
                'candidate_support_share': support_sum / max(total_support, 1e-6),
                'candidate_rho_mean': float(np.mean(rhos)),
                'candidate_pi_mean': float(np.mean(pis)),
                'candidate_self_anti_sum': self_anti_sum,
                'candidate_self_anti_mean': float(np.mean(anti_scores)),
                'candidate_contradiction_sum': contradiction_sum,
                'candidate_freshest_age': freshest_age,
                'candidate_mean_age': float(np.mean(ages)),
                'candidate_is_global_newest': int(newest_write == global_newest_time),
                'candidate_newest_gap': global_newest_time - newest_write,
                'support_margin_vs_rival': support_sum - rival_support,
                'adjusted_support_margin': (support_sum - self_anti_sum - contradiction_sum) - rival_support,
                'external_support_sum': external_support_sum,
                'external_anti_sum': external_anti_sum,
                'external_anti_share': external_anti_sum / max(total_contradiction, 1e-6) if total_contradiction > 0 else 0.0,
                'query_stress_value': float(query['stress_value']),
                'query_lag': int(query['query_lag']),
                'query_conflict_rate': float(query['conflict_rate']),
                'query_world_change_scale': float(query['world_change_scale']),
            }
        )
    return rows


def _build_value_resolver_examples(
    run_path: Path,
    models_dir: Path,
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relevance_bundle = load_relevance_bundle(models_dir)
    query_validity_bundle = load_query_validity_bundle(models_dir, target_label='valid_label')
    anti_support_bundle = load_anti_support_bundle(models_dir)
    indices = _build_episode_indices(memories, queries, labels)

    examples: list[dict[str, Any]] = []
    query_stats: list[dict[str, Any]] = []
    for query in queries:
        index = indices[query['episode_id']]
        available_memories = [memory for memory in index.memories if int(memory['write_time']) <= int(query['query_time'])]
        if not available_memories:
            continue
        contexts = build_query_memory_contexts(available_memories, query)
        raw_tfidf = _tfidf_similarity(index, query, available_memories)
        normalized_tfidf = normalized_similarity(raw_tfidf)

        direct_valid_features: list[dict[str, Any]] = []
        for memory in available_memories:
            direct_valid_features.append(
                build_query_validity_feature_dict(
                    query,
                    memory,
                    normalized_tfidf.get(memory['memory_id'], 0.0),
                    contexts.get(memory['memory_id'], {}),
                )
            )

        rho_probs = relevance_bundle.pipeline.predict_proba(direct_valid_features)[:, 1]
        valid_probs = query_validity_bundle.pipeline.predict_proba(direct_valid_features)[:, 1]
        anti_probs = anti_support_bundle.pipeline.predict_proba(direct_valid_features)[:, 1]

        memory_components: dict[str, dict[str, Any]] = {}
        for memory, rho_hat, valid_pi, anti_hat in zip(available_memories, rho_probs, valid_probs, anti_probs, strict=False):
            memory_components[memory['memory_id']] = {
                'rho_hat': float(rho_hat),
                'pi_hat_direct': float(valid_pi),
                'u_hat': float(rho_hat) * float(valid_pi),
                'anti_support_hat': float(anti_hat),
            }

        ranked = sorted(
            available_memories,
            key=lambda memory: (
                memory_components[memory['memory_id']]['u_hat'],
                int(memory['write_time']),
            ),
            reverse=True,
        )[:top_k]
        relevant_retrieved = [
            memory for memory in ranked if memory['entity'] == query['entity'] and memory['slot'] == query['slot']
        ]
        fallback_pool = relevant_retrieved or [memory for memory in ranked if memory['slot'] == query['slot']] or ranked[:1]
        candidate_rows = build_value_candidate_feature_rows(query, fallback_pool, memory_components)
        candidate_values = {row['candidate_value'] for row in candidate_rows}
        query_stats.append(
            {
                'query_id': query['query_id'],
                'episode_id': query['episode_id'],
                'num_candidates': len(candidate_rows),
                'gold_present': bool(query['gold_value'] in candidate_values),
            }
        )
        if query['gold_value'] not in candidate_values:
            continue
        for row in candidate_rows:
            examples.append(
                {
                    **row,
                    'label': int(row['candidate_value'] == query['gold_value']),
                }
            )
    return examples, query_stats


def _fit_value_resolver_with_splits(
    examples: list[dict[str, Any]],
    episode_splits: dict[str, list[str]],
    query_stats: list[dict[str, Any]],
) -> tuple[LearnedValueResolverBundle, dict[str, Any]]:
    split_frames = {
        split_name: [row for row in examples if row['episode_id'] in episode_ids]
        for split_name, episode_ids in episode_splits.items()
    }
    query_stats_by_split = {
        split_name: [row for row in query_stats if row['episode_id'] in episode_ids]
        for split_name, episode_ids in episode_splits.items()
    }
    train_rows = split_frames['train']

    candidate_specs = {
        'linear_listwise_full': {
            'trainer': _train_listwise_value_resolver,
            'active_feature_names': ALL_VALUE_RESOLVER_FEATURES,
        },
        'linear_listwise_no_anti_support': {
            'trainer': _train_listwise_value_resolver,
            'active_feature_names': _feature_names_without_groups('anti_support'),
        },
        'linear_listwise_support_freshness': {
            'trainer': _train_listwise_value_resolver,
            'active_feature_names': _feature_names_for_groups('support', 'freshness'),
        },
        'nonlinear_tree_full': {
            'trainer': _train_tree_value_resolver,
            'active_feature_names': ALL_VALUE_RESOLVER_FEATURES,
        },
        'nonlinear_tree_no_anti_support': {
            'trainer': _train_tree_value_resolver,
            'active_feature_names': _feature_names_without_groups('anti_support'),
        },
    }

    candidates: dict[str, tuple[LearnedValueResolverBundle, dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    val_rows = split_frames.get('val', [])
    for offset, (candidate_name, spec) in enumerate(candidate_specs.items()):
        bundle, training = spec['trainer'](train_rows, active_feature_names=spec['active_feature_names'])
        split_summary = _summarize_bundle_by_split(bundle, split_frames, query_stats_by_split)
        bootstrap = _bootstrap_query_metrics(val_rows, bundle, seed=17 + offset)
        candidates[candidate_name] = (bundle, training, split_summary, bootstrap)

    selected_name, selection_diagnostics = _select_value_resolver_candidate(candidates)
    selected_bundle, selected_training, selected_summaries, _selected_bootstrap = candidates[selected_name]

    summary: dict[str, Any] = {
        'split_strategy': 'episode_level',
        'task': 'value_resolver',
        'condition': selected_bundle.condition,
        'objective': selected_bundle.objective,
        'selection_metric': 'query_top1_accuracy',
        'selection_policy': {
            'top1_tolerance': VALUE_RESOLVER_SELECTION_TOP1_TOLERANCE,
            'tie_breakers': ['pairwise_accuracy', 'roc_auc', 'bootstrap_query_top1_mean'],
            'bootstrap_resamples': VALUE_RESOLVER_BOOTSTRAP_RESAMPLES,
        },
        'primary_metrics': ['query_top1_accuracy', 'pairwise_accuracy'],
        'selected_model': selected_name,
        'selection_diagnostics': selection_diagnostics,
        'num_examples': len(examples),
        'num_queries': len(query_stats),
        'num_queries_with_gold_candidate': int(sum(1 for row in query_stats if row['gold_present'])),
        'optimization': selected_training.get('optimization'),
        'splits': selected_summaries,
        'model_candidates': {},
        'feature_ablation': {
            split_name: _feature_ablation(selected_bundle, split_frames.get(split_name, []))
            for split_name in ('val', 'test')
        },
    }
    for name, (bundle, training, split_summary, bootstrap) in candidates.items():
        summary['model_candidates'][name] = {
            'objective': bundle.objective,
            'selected': bool(name == selected_name),
            'active_feature_names': list(bundle.active_feature_names or ()),
            'training': training,
            'bootstrap': bootstrap,
            'splits': split_summary,
        }
    summary['selected_checkpoint'] = f'{selected_name}_selected_by_robust_val_policy'
    return selected_bundle, summary


def _candidate_feature_dict(row: dict[str, Any], active_feature_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    if active_feature_names is None:
        active_keys = set(ALL_VALUE_RESOLVER_FEATURES)
    else:
        active_keys = set(active_feature_names)
    return {
        key: value
        for key, value in row.items()
        if key not in VALUE_RESOLVER_METADATA_FIELDS and key in active_keys
    }


def _train_listwise_value_resolver(
    train_rows: list[dict[str, Any]],
    active_feature_names: tuple[str, ...],
) -> tuple[LearnedValueResolverBundle, dict[str, Any]]:
    if not train_rows:
        raise RuntimeError('Value-resolver training requires non-empty train rows.')

    feature_rows = [_candidate_feature_dict(row, active_feature_names) for row in train_rows]
    vectorizer = DictVectorizer(sparse=True)
    candidate_matrix = vectorizer.fit_transform(feature_rows)
    scaler = StandardScaler(with_mean=False)
    candidate_matrix = scaler.fit_transform(candidate_matrix).tocsr()

    labels = np.asarray([int(row['label']) for row in train_rows], dtype=float)
    grouped_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(train_rows):
        grouped_indices.setdefault(row['query_id'], []).append(idx)

    supervised_groups = []
    for indices in grouped_indices.values():
        group_indices = np.asarray(indices, dtype=int)
        group_labels = labels[group_indices]
        if np.sum(group_labels) <= 0 or np.sum(group_labels) >= len(group_labels):
            continue
        supervised_groups.append(group_indices)

    if not supervised_groups:
        classifier = LinearValueResolverRanker(weights=np.zeros(candidate_matrix.shape[1], dtype=float), bias=0.0)
        optimization = {
            'converged': True,
            'message': 'No supervised groups; falling back to zero-weight ranker.',
            'iterations': 0,
            'final_loss': 0.0,
            'l2_reg': 0.0,
        }
    else:
        weights, optimization = _fit_listwise_linear_ranker(candidate_matrix, labels, supervised_groups, l2_reg=0.05)
        classifier = LinearValueResolverRanker(weights=weights, bias=0.0)

    bundle = LearnedValueResolverBundle(
        condition='top_k_direct_valid_candidates',
        objective='listwise_query_ranking',
        vectorizer=vectorizer,
        scaler=scaler,
        classifier=classifier,
        active_feature_names=tuple(active_feature_names),
    )
    return bundle, {
        'num_queries_with_supervision': len(supervised_groups),
        'optimization': optimization,
    }


def _train_tree_value_resolver(
    train_rows: list[dict[str, Any]],
    active_feature_names: tuple[str, ...],
) -> tuple[LearnedValueResolverBundle, dict[str, Any]]:
    feature_rows = [_candidate_feature_dict(row, active_feature_names) for row in train_rows]
    labels = [int(row['label']) for row in train_rows]
    pipeline = Pipeline(
        steps=[
            ('vectorizer', DictVectorizer(sparse=True)),
            (
                'classifier',
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight='balanced_subsample',
                    random_state=0,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(feature_rows, labels)
    bundle = LearnedValueResolverBundle(
        condition='top_k_direct_valid_candidates',
        objective='query_normalized_tree_scoring',
        pipeline=pipeline,
        active_feature_names=tuple(active_feature_names),
    )
    return bundle, {
        'model_family': 'extra_trees',
        'n_estimators': 300,
        'min_samples_leaf': 2,
    }


def _fit_listwise_linear_ranker(
    candidate_matrix: Any,
    labels: np.ndarray,
    supervised_groups: list[np.ndarray],
    l2_reg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    num_features = candidate_matrix.shape[1]

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        grad = np.zeros(num_features, dtype=float)
        for indices in supervised_groups:
            group_matrix = candidate_matrix[indices]
            group_labels = labels[indices]
            label_mass = group_labels / max(float(np.sum(group_labels)), 1.0)
            scores = np.asarray(group_matrix @ weights).ravel()
            max_score = float(np.max(scores))
            shifted = scores - max_score
            exp_shifted = np.exp(shifted)
            probs = exp_shifted / max(float(np.sum(exp_shifted)), 1e-12)
            loss += -float(np.dot(label_mass, scores)) + max_score + float(np.log(np.sum(exp_shifted)))
            grad += np.asarray(group_matrix.T @ (probs - label_mass)).ravel()
        loss = loss / len(supervised_groups)
        grad = grad / len(supervised_groups)
        loss += 0.5 * l2_reg * float(np.dot(weights, weights))
        grad += l2_reg * weights
        return loss, grad

    initial = np.zeros(num_features, dtype=float)
    result = minimize(
        fun=lambda weights: objective(weights)[0],
        x0=initial,
        jac=lambda weights: objective(weights)[1],
        method='L-BFGS-B',
        options={'maxiter': 400, 'ftol': 1e-9},
    )
    return np.asarray(result.x, dtype=float), {
        'converged': bool(result.success),
        'message': str(result.message),
        'iterations': int(getattr(result, 'nit', 0)),
        'final_loss': float(result.fun),
        'l2_reg': float(l2_reg),
    }


def _summarize_bundle_by_split(
    bundle: LearnedValueResolverBundle,
    split_frames: dict[str, list[dict[str, Any]]],
    query_stats_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split_name, rows in split_frames.items():
        labels = [int(row['label']) for row in rows]
        candidate_scores = bundle.predict_candidate_scores(rows)
        candidate_metrics = _evaluate_candidate_scores(candidate_scores, labels)
        query_rows = query_stats_by_split[split_name]
        summaries[split_name] = {
            'num_examples': len(rows),
            'num_episodes': len({row['episode_id'] for row in rows}),
            'num_queries': len(query_rows),
            'num_queries_with_gold_candidate': int(sum(1 for row in query_rows if row['gold_present'])),
            'gold_candidate_rate': float(np.mean([row['gold_present'] for row in query_rows])) if query_rows else None,
            'positive_rate': float(np.mean(labels)) if labels else None,
            'query_top1_accuracy': _query_top1_accuracy(rows, bundle),
            'pairwise_accuracy': _query_pairwise_accuracy(rows, bundle),
            'candidate_level_metrics': candidate_metrics,
            **candidate_metrics,
        }
    return summaries


def _select_value_resolver_candidate(
    candidates: dict[str, tuple[LearnedValueResolverBundle, dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    val_top1 = {
        name: float(split_summary.get('val', {}).get('query_top1_accuracy') or -1.0)
        for name, (_, _, split_summary, _) in candidates.items()
    }
    best_top1 = max(val_top1.values()) if val_top1 else -1.0
    shortlist = [
        name for name, score in val_top1.items()
        if score >= best_top1 - VALUE_RESOLVER_SELECTION_TOP1_TOLERANCE
    ]
    if not shortlist:
        shortlist = list(candidates)

    selected_name = max(
        shortlist,
        key=lambda name: _selection_key(candidates[name][2].get('val', {}), candidates[name][3]),
    )
    diagnostics = {
        'best_raw_val_query_top1': best_top1,
        'top1_tolerance': VALUE_RESOLVER_SELECTION_TOP1_TOLERANCE,
        'shortlist': shortlist,
        'ranking': [],
    }
    ranked_names = sorted(
        candidates,
        key=lambda name: _selection_key(candidates[name][2].get('val', {}), candidates[name][3]),
        reverse=True,
    )
    for name in ranked_names:
        split_summary = candidates[name][2].get('val', {})
        bootstrap = candidates[name][3]
        diagnostics['ranking'].append(
            {
                'name': name,
                'val_query_top1_accuracy': split_summary.get('query_top1_accuracy'),
                'val_pairwise_accuracy': split_summary.get('pairwise_accuracy'),
                'val_roc_auc': split_summary.get('roc_auc'),
                'bootstrap': bootstrap,
                'selected': bool(name == selected_name),
            }
        )
    return selected_name, diagnostics


def _selection_key(split_summary: dict[str, Any], bootstrap: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(split_summary.get('pairwise_accuracy') or -1.0),
        float(split_summary.get('roc_auc') or -1.0),
        float(bootstrap.get('query_top1_mean') or -1.0),
        float(bootstrap.get('pairwise_mean') or -1.0),
        float(split_summary.get('query_top1_accuracy') or -1.0),
    )


def _bootstrap_query_metrics(
    rows: list[dict[str, Any]],
    bundle: LearnedValueResolverBundle,
    seed: int,
    num_resamples: int = VALUE_RESOLVER_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if not rows:
        return {
            'num_resamples': 0,
            'num_queries': 0,
            'query_top1_mean': None,
            'query_top1_std': None,
            'query_top1_ci_low': None,
            'query_top1_ci_high': None,
            'pairwise_mean': None,
            'pairwise_std': None,
            'pairwise_ci_low': None,
            'pairwise_ci_high': None,
        }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row['query_id'], []).append(row)
    query_groups = list(grouped.values())
    if not query_groups:
        return {
            'num_resamples': 0,
            'num_queries': 0,
            'query_top1_mean': None,
            'query_top1_std': None,
            'query_top1_ci_low': None,
            'query_top1_ci_high': None,
            'pairwise_mean': None,
            'pairwise_std': None,
            'pairwise_ci_low': None,
            'pairwise_ci_high': None,
        }
    rng = np.random.default_rng(seed)
    top1_values: list[float] = []
    pairwise_values: list[float] = []
    num_queries = len(query_groups)
    for _ in range(num_resamples):
        sampled_indices = rng.integers(0, num_queries, size=num_queries)
        sampled_rows = [row for idx in sampled_indices for row in query_groups[int(idx)]]
        top1 = _query_top1_accuracy(sampled_rows, bundle)
        pairwise = _query_pairwise_accuracy(sampled_rows, bundle)
        if top1 is not None:
            top1_values.append(float(top1))
        if pairwise is not None:
            pairwise_values.append(float(pairwise))
    return {
        'num_resamples': int(num_resamples),
        'num_queries': int(num_queries),
        'query_top1_mean': float(np.mean(top1_values)) if top1_values else None,
        'query_top1_std': float(np.std(top1_values)) if top1_values else None,
        'query_top1_ci_low': float(np.quantile(top1_values, 0.1)) if top1_values else None,
        'query_top1_ci_high': float(np.quantile(top1_values, 0.9)) if top1_values else None,
        'pairwise_mean': float(np.mean(pairwise_values)) if pairwise_values else None,
        'pairwise_std': float(np.std(pairwise_values)) if pairwise_values else None,
        'pairwise_ci_low': float(np.quantile(pairwise_values, 0.1)) if pairwise_values else None,
        'pairwise_ci_high': float(np.quantile(pairwise_values, 0.9)) if pairwise_values else None,
    }


def _feature_ablation(bundle: LearnedValueResolverBundle, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    baseline_top1 = _query_top1_accuracy(rows, bundle)
    baseline_pairwise = _query_pairwise_accuracy(rows, bundle)
    report = {
        'baseline_query_top1_accuracy': baseline_top1,
        'baseline_pairwise_accuracy': baseline_pairwise,
        'groups': {},
    }
    for group_name, feature_names in VALUE_RESOLVER_FEATURE_GROUPS.items():
        ablated_rows = [_ablate_feature_row(row, feature_names) for row in rows]
        ablated_top1 = _query_top1_accuracy(ablated_rows, bundle)
        ablated_pairwise = _query_pairwise_accuracy(ablated_rows, bundle)
        report['groups'][group_name] = {
            'query_top1_accuracy': ablated_top1,
            'pairwise_accuracy': ablated_pairwise,
            'query_top1_drop': (baseline_top1 - ablated_top1) if baseline_top1 is not None and ablated_top1 is not None else None,
            'pairwise_drop': (baseline_pairwise - ablated_pairwise) if baseline_pairwise is not None and ablated_pairwise is not None else None,
        }
    return report


def _ablate_feature_row(row: dict[str, Any], feature_names: tuple[str, ...]) -> dict[str, Any]:
    ablated = dict(row)
    for feature_name in feature_names:
        if feature_name in ablated:
            ablated[feature_name] = 0.0
    return ablated


def _feature_names_for_groups(*group_names: str) -> tuple[str, ...]:
    return tuple(
        feature_name
        for group_name in group_names
        for feature_name in VALUE_RESOLVER_FEATURE_GROUPS[group_name]
    )


def _feature_names_without_groups(*excluded_group_names: str) -> tuple[str, ...]:
    excluded = set(excluded_group_names)
    return tuple(
        feature_name
        for group_name, feature_names in VALUE_RESOLVER_FEATURE_GROUPS.items()
        if group_name not in excluded
        for feature_name in feature_names
    )


def _evaluate_candidate_scores(candidate_scores: list[float], labels: list[int]) -> dict[str, Any]:
    if not candidate_scores:
        return {
            'accuracy': None,
            'mean_probability': None,
            'roc_auc': None,
            'average_precision': None,
        }
    scores = np.asarray(candidate_scores, dtype=float)
    threshold = 0.5 if np.all((scores >= 0.0) & (scores <= 1.0)) else 0.0
    predictions = (scores >= threshold).astype(int)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'mean_probability': float(np.mean(scores)),
        'roc_auc': float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None,
        'average_precision': float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else None,
    }


def _query_top1_accuracy(rows: list[dict[str, Any]], bundle: LearnedValueResolverBundle) -> float | None:
    if not rows:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row['query_id'], []).append(row)
    correct = 0
    total = 0
    for group in grouped.values():
        scores = bundle.predict_candidate_scores(group)
        best_idx = int(np.argmax(scores))
        correct += int(group[best_idx]['label'])
        total += 1
    return float(correct / total) if total else None


def _query_pairwise_accuracy(rows: list[dict[str, Any]], bundle: LearnedValueResolverBundle) -> float | None:
    if not rows:
        return None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row['query_id'], []).append(row)

    pair_scores: list[float] = []
    for group in grouped.values():
        scores = bundle.predict_candidate_scores(group)
        positive_scores = [score for row, score in zip(group, scores, strict=False) if int(row['label']) == 1]
        negative_scores = [score for row, score in zip(group, scores, strict=False) if int(row['label']) == 0]
        if not positive_scores or not negative_scores:
            continue
        for positive_score in positive_scores:
            for negative_score in negative_scores:
                if positive_score > negative_score:
                    pair_scores.append(1.0)
                elif positive_score < negative_score:
                    pair_scores.append(0.0)
                else:
                    pair_scores.append(0.5)
    return float(np.mean(pair_scores)) if pair_scores else None
