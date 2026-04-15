from __future__ import annotations

from statistics import mean
from typing import Any

import pandas as pd


ERROR_LABELS = ["none", "forgetting", "stale_dominance", "residual_controller_error"]


def classify_error_attribution(
    prediction: dict[str, Any],
    query: dict[str, Any],
    label: dict[str, Any],
    memory_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Classify error causes for benchmark diagnostics.

    The original Sprint 1/2 proxy used wrong valid support, which rarely fires because
    stale interference typically comes from relevant memories that are no longer valid.
    We keep that legacy count for continuity, but the main stale-dominance decision now
    uses retrieved wrong stale support versus retrieved gold useful support.
    """
    gold_value = query["gold_value"]
    predicted_value = prediction["predicted_value"]
    retrieved_ids = prediction["retrieved_ids"]
    useful_ids = set(label["useful_memory_ids"])
    valid_ids = set(label["valid_memory_ids"])
    relevant_ids = set(label["relevant_memory_ids"])
    retrieved_useful_ids = [memory_id for memory_id in retrieved_ids if memory_id in useful_ids]
    retrieved_valid_ids = [memory_id for memory_id in retrieved_ids if memory_id in valid_ids]
    retrieved_relevant_ids = [memory_id for memory_id in retrieved_ids if memory_id in relevant_ids]

    useful_value_scores: dict[str, float] = {}
    legacy_valid_value_scores: dict[str, float] = {}
    stale_wrong_value_scores: dict[str, float] = {}
    corrupted_wrong_value_scores: dict[str, float] = {}
    stale_wrong_ids: list[str] = []
    corrupted_wrong_ids: list[str] = []
    wrong_relevant_ids: list[str] = []

    for memory_id in retrieved_relevant_ids:
        memory = memory_by_id[memory_id]
        value = memory["value_canonical"]
        if memory_id in useful_ids:
            useful_value_scores[value] = useful_value_scores.get(value, 0.0) + 1.0
        if memory_id in valid_ids:
            legacy_valid_value_scores[value] = legacy_valid_value_scores.get(value, 0.0) + 1.0
        if value == gold_value:
            continue
        wrong_relevant_ids.append(memory_id)
        if bool(memory.get("write_correct")):
            stale_wrong_ids.append(memory_id)
            stale_wrong_value_scores[value] = stale_wrong_value_scores.get(value, 0.0) + 1.0
        else:
            corrupted_wrong_ids.append(memory_id)
            corrupted_wrong_value_scores[value] = corrupted_wrong_value_scores.get(value, 0.0) + 1.0

    is_error = predicted_value != gold_value
    gold_useful_support = useful_value_scores.get(gold_value, 0.0)
    stale_wrong_support = max(stale_wrong_value_scores.values() or [0.0])
    corrupted_wrong_support = max(corrupted_wrong_value_scores.values() or [0.0])
    legacy_wrong_valid_support = max(
        [score for value, score in legacy_valid_value_scores.items() if value != gold_value] or [0.0]
    )

    if not is_error:
        attribution = "none"
    elif not retrieved_useful_ids:
        attribution = "forgetting"
    elif stale_wrong_support >= max(1.0, gold_useful_support) and stale_wrong_support >= corrupted_wrong_support:
        attribution = "stale_dominance"
    elif (stale_wrong_support + corrupted_wrong_support) >= max(1.0, gold_useful_support) and stale_wrong_support > 0:
        attribution = "stale_dominance"
    else:
        attribution = "residual_controller_error"

    gold_valid_ids = [
        memory_id
        for memory_id in label["valid_memory_ids"]
        if memory_by_id[memory_id]["value_canonical"] == gold_value
    ]
    wrong_support_ids = stale_wrong_ids or wrong_relevant_ids

    def _average_age(memory_ids: list[str]) -> float | None:
        if not memory_ids:
            return None
        return mean(query["query_time"] - int(memory_by_id[memory_id]["write_time"]) for memory_id in memory_ids)

    def _average_quality(memory_ids: list[str]) -> float | None:
        if not memory_ids:
            return None
        return mean(float(memory_by_id[memory_id]["source_quality"]) for memory_id in memory_ids)

    gold_age = _average_age(gold_valid_ids)
    wrong_age = _average_age(wrong_support_ids)
    gold_quality = _average_quality(gold_valid_ids)
    wrong_quality = _average_quality(wrong_support_ids)

    return {
        "error_attribution": attribution,
        "error_decomposition_mode": "label_space_stale_interference_proxy",
        "stale_dominance_definition": "retrieved_wrong_stale_support_vs_retrieved_gold_useful_support",
        "is_error": bool(is_error),
        "retrieved_relevant_count": len(retrieved_relevant_ids),
        "retrieved_valid_count": len(retrieved_valid_ids),
        "retrieved_useful_count": len(retrieved_useful_ids),
        "retrieved_stale_wrong_count": len(stale_wrong_ids),
        "retrieved_corrupted_wrong_count": len(corrupted_wrong_ids),
        "true_gold_support": gold_useful_support,
        "true_wrong_support": stale_wrong_support,
        "true_wrong_stale_support": stale_wrong_support,
        "true_wrong_corrupted_support": corrupted_wrong_support,
        "legacy_true_wrong_valid_support": legacy_wrong_valid_support,
        "conflict_present": bool(label.get("dominant_wrong_value") or stale_wrong_support > 0 or corrupted_wrong_support > 0),
        "age_gap": None if gold_age is None or wrong_age is None else float(wrong_age - gold_age),
        "reliability_gap": None if gold_quality is None or wrong_quality is None else float(gold_quality - wrong_quality),
    }


def summarize_metrics(diagnostics_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(diagnostics_rows)
    metrics: list[dict[str, Any]] = []
    grouping_keys = ["method", "stress_name", "stress_value"]
    for (method, stress_name, stress_value), group in frame.groupby(grouping_keys, dropna=False):
        error_mask = group["is_error"]
        num_errors = int(error_mask.sum())
        row = {
            "method": method,
            "stress_name": stress_name,
            "stress_value": float(stress_value),
            "num_queries": int(len(group)),
            "num_errors": num_errors,
            "accuracy": float((~error_mask).mean()),
            "collapse_rate": float(error_mask.mean()),
            "forgetting_rate": float((group["error_attribution"] == "forgetting").mean()),
            "stale_dominance_rate": float((group["error_attribution"] == "stale_dominance").mean()),
            "residual_rate": float((group["error_attribution"] == "residual_controller_error").mean()),
            "forgetting_share_of_errors": float(((group["error_attribution"] == "forgetting") & error_mask).sum() / num_errors) if num_errors else 0.0,
            "stale_dominance_share_of_errors": float(((group["error_attribution"] == "stale_dominance") & error_mask).sum() / num_errors) if num_errors else 0.0,
            "residual_share_of_errors": float(((group["error_attribution"] == "residual_controller_error") & error_mask).sum() / num_errors) if num_errors else 0.0,
            "mean_confidence": float(group["confidence"].mean()),
        }
        metrics.append(row)
    for method, group in frame.groupby("method", dropna=False):
        error_mask = group["is_error"]
        num_errors = int(error_mask.sum())
        metrics.append(
            {
                "method": method,
                "stress_name": "overall",
                "stress_value": -1.0,
                "num_queries": int(len(group)),
                "num_errors": num_errors,
                "accuracy": float((~error_mask).mean()),
                "collapse_rate": float(error_mask.mean()),
                "forgetting_rate": float((group["error_attribution"] == "forgetting").mean()),
                "stale_dominance_rate": float((group["error_attribution"] == "stale_dominance").mean()),
                "residual_rate": float((group["error_attribution"] == "residual_controller_error").mean()),
                "forgetting_share_of_errors": float(((group["error_attribution"] == "forgetting") & error_mask).sum() / num_errors) if num_errors else 0.0,
                "stale_dominance_share_of_errors": float(((group["error_attribution"] == "stale_dominance") & error_mask).sum() / num_errors) if num_errors else 0.0,
                "residual_share_of_errors": float(((group["error_attribution"] == "residual_controller_error") & error_mask).sum() / num_errors) if num_errors else 0.0,
                "mean_confidence": float(group["confidence"].mean()),
            }
        )
    return sorted(metrics, key=lambda row: (row["method"], row["stress_value"], row["stress_name"]))
