from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from memory_collapse.domain import list_slot_specs
from memory_collapse.io_utils import ensure_dir, read_jsonl, write_csv


WRITE_MODEL_FILENAME = "write_correctness_model.pkl"
SURVIVAL_MODEL_FILENAME = "survival_model.pkl"
SUMMARY_FILENAME = "estimator_summary.json"
SPLIT_FILENAME = "episode_splits.json"


@dataclass
class LearnedEstimatorBundle:
    write_pipeline: Pipeline
    survival_pipeline: Pipeline
    slot_volatility: dict[str, float]

    def predict_write_correctness(self, memory: dict[str, Any], observable_context: dict[str, Any] | None = None) -> float:
        features = [_memory_feature_dict(memory, self.slot_volatility, observable_context or {})]
        return float(self.write_pipeline.predict_proba(features)[0, 1])

    def predict_survival(
        self,
        memory: dict[str, Any],
        query: dict[str, Any],
        observable_context: dict[str, Any] | None = None,
    ) -> float:
        features = [_survival_feature_dict(memory, query, self.slot_volatility, observable_context or {})]
        return float(self.survival_pipeline.predict_proba(features)[0, 1])


def resolve_models_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.name == "models":
        return candidate
    if (candidate / WRITE_MODEL_FILENAME).exists() and (candidate / SURVIVAL_MODEL_FILENAME).exists():
        return candidate
    return candidate / "models"


def estimator_artifact_paths(path: str | Path) -> dict[str, str]:
    models_dir = resolve_models_dir(path)
    return {
        "write_model": str(models_dir / WRITE_MODEL_FILENAME),
        "survival_model": str(models_dir / SURVIVAL_MODEL_FILENAME),
        "summary": str(models_dir / SUMMARY_FILENAME),
        "splits": str(models_dir / SPLIT_FILENAME),
    }


def estimator_models_exist(path: str | Path) -> bool:
    artifact_paths = estimator_artifact_paths(path)
    return Path(artifact_paths["write_model"]).exists() and Path(artifact_paths["survival_model"]).exists()


def load_episode_splits(path: str | Path) -> dict[str, list[str]]:
    models_dir = resolve_models_dir(path)
    with (models_dir / SPLIT_FILENAME).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_episode_ids(episode_ids: list[str], seed: int) -> dict[str, list[str]]:
    return _split_episodes(episode_ids, seed)


def load_estimator_bundle(path: str | Path) -> LearnedEstimatorBundle:
    models_dir = resolve_models_dir(path)
    with (models_dir / WRITE_MODEL_FILENAME).open("rb") as handle:
        write_pipeline = pickle.load(handle)
    with (models_dir / SURVIVAL_MODEL_FILENAME).open("rb") as handle:
        survival_pipeline = pickle.load(handle)
    return LearnedEstimatorBundle(
        write_pipeline=write_pipeline,
        survival_pipeline=survival_pipeline,
        slot_volatility=_slot_volatility_map(),
    )


def build_query_memory_contexts(available_memories: list[dict[str, Any]], query: dict[str, Any]) -> dict[str, dict[str, Any]]:
    query_time = int(query["query_time"])
    by_entity_slot: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for memory in available_memories:
        by_entity_slot.setdefault((memory["entity"], memory["slot"]), []).append(memory)

    contexts: dict[str, dict[str, Any]] = {}
    for key, group in by_entity_slot.items():
        group_sorted = sorted(group, key=lambda row: (int(row["write_time"]), row["memory_id"]))
        distinct_values = {row["value_canonical"] for row in group_sorted}
        latest_write_time = max(int(row["write_time"]) for row in group_sorted)
        for memory in group_sorted:
            memory_time = int(memory["write_time"])
            newer = [row for row in group_sorted if int(row["write_time"]) > memory_time]
            older = [row for row in group_sorted if int(row["write_time"]) < memory_time]
            same_value = [row for row in group_sorted if row["value_canonical"] == memory["value_canonical"]]
            diff_value = [row for row in group_sorted if row["value_canonical"] != memory["value_canonical"]]
            same_source = [row for row in group_sorted if row["source_id"] == memory["source_id"]]
            contexts[memory["memory_id"]] = {
                "observed_group_size": len(group_sorted),
                "observed_distinct_value_count": len(distinct_values),
                "observed_same_value_count": len(same_value),
                "observed_different_value_count": len(diff_value),
                "observed_newer_count": len(newer),
                "observed_newer_same_value_count": sum(1 for row in newer if row["value_canonical"] == memory["value_canonical"]),
                "observed_newer_different_value_count": sum(1 for row in newer if row["value_canonical"] != memory["value_canonical"]),
                "observed_older_count": len(older),
                "observed_older_same_value_count": sum(1 for row in older if row["value_canonical"] == memory["value_canonical"]),
                "observed_same_source_count": len(same_source),
                "observed_same_source_same_value_count": sum(1 for row in same_source if row["value_canonical"] == memory["value_canonical"]),
                "observed_same_source_different_value_count": sum(1 for row in same_source if row["value_canonical"] != memory["value_canonical"]),
                "observed_is_latest": int(memory_time == latest_write_time),
                "observed_latest_gap": query_time - latest_write_time,
                "observed_age_rank": sum(1 for row in group_sorted if int(row["write_time"]) > memory_time),
                "observed_conflict_ratio": (len(diff_value) / max(len(group_sorted), 1)),
                "observed_has_conflict": int(len(distinct_values) > 1),
                "observed_query_entity_slot_match": int(memory["entity"] == query["entity"] and memory["slot"] == query["slot"]),
            }
    return contexts


def train_estimators(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    run_path = Path(run_dir)
    models_dir = ensure_dir(resolve_models_dir(output_dir or run_path))

    memories = read_jsonl(run_path / "data" / "memories.jsonl")
    queries = read_jsonl(run_path / "data" / "queries.jsonl")
    world_rows = read_jsonl(run_path / "data" / "world.jsonl")
    config_snapshot = json.loads(json.dumps({}))
    config_path = run_path / "config_snapshot.yaml"
    if config_path.exists():
        import yaml

        config_snapshot = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    seed = int(config_snapshot.get("dataset", {}).get("seed", 0))
    slot_volatility = _slot_volatility_map()

    write_examples, survival_examples = _build_training_examples(memories, queries, world_rows, slot_volatility)
    episode_splits = _split_episodes(sorted({row["episode_id"] for row in write_examples}), seed)

    write_pipeline, write_summary = _fit_task_with_splits(write_examples, slot_volatility, episode_splits, task="write")
    survival_pipeline, survival_summary = _fit_task_with_splits(survival_examples, slot_volatility, episode_splits, task="survival")

    with (models_dir / WRITE_MODEL_FILENAME).open("wb") as handle:
        pickle.dump(write_pipeline, handle)
    with (models_dir / SURVIVAL_MODEL_FILENAME).open("wb") as handle:
        pickle.dump(survival_pipeline, handle)

    with (models_dir / SUMMARY_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split_strategy": "episode_level",
                "write_correctness": write_summary,
                "survival": survival_summary,
            },
            handle,
            indent=2,
        )
    with (models_dir / SPLIT_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(episode_splits, handle, indent=2)

    write_csv(models_dir / "write_training_examples.csv", write_examples[:10000])
    write_csv(models_dir / "survival_training_examples.csv", survival_examples[:10000])

    from memory_collapse.anti_support import train_anti_support_model
    from memory_collapse.controller import train_controller_calibration
    from memory_collapse.query_validity import train_query_validity_models
    from memory_collapse.relevance import train_relevance_model
    from memory_collapse.value_resolver import train_value_resolver

    relevance_artifacts = train_relevance_model(run_path, output_dir=models_dir)
    query_validity_artifacts = train_query_validity_models(run_path, output_dir=models_dir)
    anti_support_artifacts = train_anti_support_model(run_path, output_dir=models_dir)
    value_resolver_artifacts = train_value_resolver(run_path, output_dir=models_dir)
    controller_artifacts = train_controller_calibration(run_path, output_dir=models_dir)
    artifacts = estimator_artifact_paths(models_dir)
    artifacts.update(relevance_artifacts)
    artifacts.update(query_validity_artifacts)
    artifacts.update(anti_support_artifacts)
    artifacts.update(value_resolver_artifacts)
    artifacts.update(controller_artifacts)
    return artifacts


def _build_training_examples(
    memories: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    world_rows: list[dict[str, Any]],
    slot_volatility: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    last_change = _build_last_change_lookup(world_rows)
    memories_by_episode: dict[str, list[dict[str, Any]]] = {}
    queries_by_episode: dict[str, list[dict[str, Any]]] = {}
    for memory in memories:
        memories_by_episode.setdefault(memory["episode_id"], []).append(memory)
    for query in queries:
        queries_by_episode.setdefault(query["episode_id"], []).append(query)

    write_examples: list[dict[str, Any]] = []
    survival_examples: list[dict[str, Any]] = []
    for episode_id, episode_queries in queries_by_episode.items():
        episode_memories = sorted(memories_by_episode.get(episode_id, []), key=lambda row: (int(row["write_time"]), row["memory_id"]))
        for query in episode_queries:
            available_memories = [memory for memory in episode_memories if int(memory["write_time"]) <= int(query["query_time"])]
            contexts = build_query_memory_contexts(available_memories, query)
            for memory in available_memories:
                context = contexts[memory["memory_id"]]
                write_examples.append(
                    {
                        "episode_id": episode_id,
                        "query_id": query["query_id"],
                        "memory_id": memory["memory_id"],
                        **_memory_feature_dict(memory, slot_volatility, context),
                        "label": int(memory["write_correct"]),
                    }
                )
                alive_key = (episode_id, memory["entity"], memory["slot"], int(query["query_time"]))
                alive = int(last_change[alive_key] <= int(memory["write_time"]))
                survival_examples.append(
                    {
                        "episode_id": episode_id,
                        "query_id": query["query_id"],
                        "memory_id": memory["memory_id"],
                        **_survival_feature_dict(memory, query, slot_volatility, context),
                        "label": alive,
                    }
                )
    if not write_examples or not survival_examples:
        raise RuntimeError("Estimator training examples could not be created.")
    return write_examples, survival_examples


def _split_episodes(episode_ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = list(episode_ids)
    random.Random(seed + 17).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        raise RuntimeError("No episodes available for splitting.")
    if n == 1:
        return {"train": shuffled, "val": [], "test": []}
    if n == 2:
        return {"train": [shuffled[0]], "val": [], "test": [shuffled[1]]}

    n_train = max(1, int(round(0.6 * n)))
    n_val = max(1, int(round(0.2 * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def _fit_task_with_splits(
    examples: list[dict[str, Any]],
    slot_volatility: dict[str, float],
    episode_splits: dict[str, list[str]],
    task: str,
) -> tuple[Pipeline, dict[str, Any]]:
    split_frames = {
        split_name: [row for row in examples if row["episode_id"] in episode_ids]
        for split_name, episode_ids in episode_splits.items()
    }
    train_rows = split_frames["train"]
    train_features = [_strip_metadata(row) for row in train_rows]
    train_labels = [int(row["label"]) for row in train_rows]
    pipeline = _train_binary_pipeline(train_features, train_labels)

    summary: dict[str, Any] = {
        "num_examples": len(examples),
        "num_episodes": len({row["episode_id"] for row in examples}),
        "splits": {},
    }
    for split_name, rows in split_frames.items():
        features = [_strip_metadata(row) for row in rows]
        labels = [int(row["label"]) for row in rows]
        metrics = _evaluate_pipeline(pipeline, features, labels)
        summary["splits"][split_name] = {
            "num_examples": len(rows),
            "num_episodes": len({row["episode_id"] for row in rows}),
            **metrics,
        }
    summary["selected_checkpoint"] = "train_only_logreg"
    summary["task"] = task
    return pipeline, summary


def _train_binary_pipeline(feature_dicts: list[dict[str, Any]], labels: list[int]) -> Pipeline:
    if len(set(labels)) <= 1:
        classifier = DummyClassifier(strategy="constant", constant=labels[0])
    else:
        classifier = LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)
    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(feature_dicts, labels)
    return pipeline


def _evaluate_pipeline(pipeline: Pipeline, feature_dicts: list[dict[str, Any]], labels: list[int]) -> dict[str, Any]:
    if not feature_dicts:
        return {
            "accuracy": None,
            "positive_rate": None,
            "mean_probability": None,
            "roc_auc": None,
        }
    probabilities = pipeline.predict_proba(feature_dicts)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "positive_rate": float(np.mean(labels)),
        "mean_probability": float(np.mean(probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else None,
    }


def _strip_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"episode_id", "query_id", "memory_id", "label"}
    }


def _slot_volatility_map() -> dict[str, float]:
    return {slot.name: slot.volatility for slot in list_slot_specs()}


def _memory_feature_dict(memory: dict[str, Any], slot_volatility: dict[str, float], observable_context: dict[str, Any]) -> dict[str, Any]:
    raw_value = str(memory["value_raw"])
    canonical_value = str(memory["value_canonical"])
    uppercase_chars = sum(1 for char in raw_value if char.isupper())
    punctuation_chars = sum(1 for char in raw_value if not char.isalnum() and not char.isspace())
    token_count = len([token for token in canonical_value.split() if token])
    return {
        "source_id": memory["source_id"],
        "slot": memory["slot"],
        "stress_name": memory["stress_name"],
        "stress_value": float(memory["stress_value"]),
        "source_quality": float(memory["source_quality"]),
        "write_time": int(memory["write_time"]),
        "slot_volatility": float(slot_volatility.get(memory["slot"], 0.05)),
        "raw_length": len(raw_value),
        "canonical_length": len(canonical_value),
        "canonical_token_count": token_count,
        "length_delta": abs(len(raw_value) - len(canonical_value)),
        "uppercase_ratio": uppercase_chars / max(len(raw_value), 1),
        "punctuation_chars": punctuation_chars,
        "contains_digit": int(any(char.isdigit() for char in raw_value)),
        "raw_equals_canonical": int(raw_value.strip().lower() == canonical_value.strip().lower()),
        "entity_alias_overlap": int(memory["entity_alias"].split()[0].lower() in memory["memory_text"].lower()),
        **observable_context,
    }


def _survival_feature_dict(
    memory: dict[str, Any],
    query: dict[str, Any],
    slot_volatility: dict[str, float],
    observable_context: dict[str, Any],
) -> dict[str, Any]:
    age = max(0, int(query["query_time"]) - int(memory["write_time"]))
    return {
        "slot": memory["slot"],
        "stress_name": query["stress_name"],
        "stress_value": float(query["stress_value"]),
        "source_id": memory["source_id"],
        "source_quality": float(memory["source_quality"]),
        "slot_volatility": float(slot_volatility.get(memory["slot"], 0.05)),
        "age": age,
        "age_squared": age * age,
        "write_time": int(memory["write_time"]),
        "query_time": int(query["query_time"]),
        "query_lag": int(query["query_lag"]),
        "world_change_scale": float(query["world_change_scale"]),
        "write_error_rate": float(query["write_error_rate"]),
        "conflict_rate": float(query["conflict_rate"]),
        "same_slot_as_query": int(memory["slot"] == query["slot"]),
        "same_entity_as_query": int(memory["entity"] == query["entity"]),
        **observable_context,
    }


def _build_last_change_lookup(world_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], int]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in world_rows:
        grouped.setdefault((row["episode_id"], row["entity"], row["slot"]), []).append(row)

    lookup: dict[tuple[str, str, str, int], int] = {}
    for (episode_id, entity, slot), rows in grouped.items():
        rows.sort(key=lambda row: int(row["time_step"]))
        previous_value = None
        last_change = 0
        for row in rows:
            time_step = int(row["time_step"])
            current_value = row["value_canonical"]
            if previous_value is None:
                last_change = 0
            elif current_value != previous_value:
                last_change = time_step
            lookup[(episode_id, entity, slot, time_step)] = last_change
            previous_value = current_value
    return lookup
