from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from memory_collapse.config import save_config_snapshot
from memory_collapse.domain import (
    build_entity_names,
    canonicalize_value,
    list_slot_specs,
    list_slot_values,
    render_raw_value,
)
from memory_collapse.io_utils import ensure_dir, write_jsonl


SOURCE_PROFILES = [
    {"source_id": "trusted", "source_quality": 0.96, "weight": 0.45},
    {"source_id": "standard", "source_quality": 0.85, "weight": 0.35},
    {"source_id": "weak", "source_quality": 0.68, "weight": 0.20},
]


def _weighted_choice(rng: random.Random, items: list[dict[str, Any]]) -> dict[str, Any]:
    threshold = rng.random() * sum(item["weight"] for item in items)
    cumulative = 0.0
    for item in items:
        cumulative += item["weight"]
        if cumulative >= threshold:
            return item
    return items[-1]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _slot_key(entity: str, slot: str) -> str:
    return f"{entity}::{slot}"


def _choose_value(slot: str, entities: list[str], rng: random.Random, current: str | None = None) -> str:
    values = list_slot_values(slot, entities)
    candidates = [value for value in values if value != current]
    if not candidates:
        return current or values[0]
    return candidates[rng.randrange(len(candidates))]


def _sample_wrong_value(
    slot: str,
    entities: list[str],
    world: list[dict[str, dict[str, str]]],
    entity: str,
    query_time: int,
    current_value: str,
    rng: random.Random,
) -> str:
    historical = []
    seen = set()
    for time_index in range(query_time - 1, -1, -1):
        candidate = world[time_index][entity][slot]
        if candidate != current_value and candidate not in seen:
            seen.add(candidate)
            historical.append(candidate)
    if historical and rng.random() < 0.65:
        return historical[rng.randrange(len(historical))]
    return _choose_value(slot, entities, rng, current=current_value)


def _build_memory_text(memory: dict[str, Any]) -> str:
    slot_text = memory["slot"].replace("_", " ")
    return (
        f"profile note for {memory['entity']}: {slot_text} -> {memory['value_raw']} "
        f"(source={memory['source_id']} time={memory['write_time']})"
    )


def _query_text(entity: str, slot: str) -> str:
    return f"What is {entity}'s current {slot.replace('_', ' ')}?"


def _noise_entity_alias(entity: str) -> str:
    first, last = entity.split(" ", 1)
    return f"{first} {last[:3]}"


def generate_artifacts(config: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    dataset = config["dataset"]
    output_root = ensure_dir(output_dir)
    data_dir = ensure_dir(output_root / "data")
    save_config_snapshot(config, output_root)

    seed = int(dataset["seed"])
    entities = build_entity_names(int(dataset["num_entities"]))
    slot_specs = list_slot_specs()

    world_rows: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []

    memory_counter = 0
    query_counter = 0

    for level_index, level in enumerate(dataset["composite_levels"]):
        stress_name = level["name"]
        for episode_index in range(int(dataset["episodes_per_level"])):
            episode_seed = seed * 10_000 + level_index * 1_000 + episode_index * 37
            rng = random.Random(episode_seed)
            episode_id = f"{stress_name}_ep{episode_index:02d}"
            initial_state: dict[str, dict[str, str]] = {}
            last_change_state: dict[str, int] = {}
            for entity in entities:
                initial_state[entity] = {}
                for slot_spec in slot_specs:
                    current = _choose_value(slot_spec.name, entities, rng)
                    initial_state[entity][slot_spec.name] = current
                    last_change_state[_slot_key(entity, slot_spec.name)] = 0

            world: list[dict[str, dict[str, str]]] = [copy.deepcopy(initial_state)]
            last_change_snapshots: list[dict[str, int]] = [copy.deepcopy(last_change_state)]

            for time_index in range(1, int(dataset["time_steps"])):
                current_state = copy.deepcopy(world[-1])
                for entity in entities:
                    for slot_spec in slot_specs:
                        drift_prob = _clip(slot_spec.volatility * float(level["world_change_scale"]), 0.0, 0.85)
                        if rng.random() < drift_prob:
                            previous = current_state[entity][slot_spec.name]
                            current_state[entity][slot_spec.name] = _choose_value(
                                slot_spec.name,
                                entities,
                                rng,
                                current=previous,
                            )
                            last_change_state[_slot_key(entity, slot_spec.name)] = time_index
                world.append(current_state)
                last_change_snapshots.append(copy.deepcopy(last_change_state))

            for time_index, state in enumerate(world):
                for entity in entities:
                    for slot_spec in slot_specs:
                        world_rows.append(
                            {
                                "stress_name": stress_name,
                                "stress_value": float(level["stress"]),
                                "episode_id": episode_id,
                                "time_step": time_index,
                                "entity": entity,
                                "slot": slot_spec.name,
                                "value_canonical": state[entity][slot_spec.name],
                            }
                        )

            for time_index, state in enumerate(world):
                for entity in entities:
                    for slot_spec in slot_specs:
                        if rng.random() >= float(dataset["memory_write_prob"]):
                            continue
                        source = _weighted_choice(rng, SOURCE_PROFILES)
                        current_value = state[entity][slot_spec.name]
                        error_rate = _clip(
                            float(level["write_error_rate"]) + 0.35 * (1.0 - float(source["source_quality"])),
                            0.01,
                            0.80,
                        )
                        write_correct = rng.random() >= error_rate
                        if write_correct:
                            stored_value = current_value
                        else:
                            stored_value = _sample_wrong_value(
                                slot_spec.name,
                                entities,
                                world,
                                entity,
                                time_index,
                                current_value,
                                rng,
                            )
                        raw_value = render_raw_value(slot_spec.name, stored_value, rng)
                        canonical_value = canonicalize_value(slot_spec.name, raw_value)
                        memory_counter += 1
                        memory = {
                            "memory_id": f"m{memory_counter:06d}",
                            "stress_name": stress_name,
                            "stress_value": float(level["stress"]),
                            "episode_id": episode_id,
                            "entity": entity,
                            "entity_alias": _noise_entity_alias(entity),
                            "slot": slot_spec.name,
                            "value_raw": raw_value,
                            "value_canonical": canonical_value,
                            "write_time": time_index,
                            "source_id": source["source_id"],
                            "source_quality": float(source["source_quality"]),
                            "write_correct": bool(canonical_value == current_value),
                            "truth_at_write": current_value,
                        }
                        memory["memory_text"] = _build_memory_text(memory)
                        memories.append(memory)

                        if rng.random() < float(level["conflict_rate"]):
                            alt_source = _weighted_choice(rng, SOURCE_PROFILES)
                            if stored_value == current_value:
                                alt_value = _sample_wrong_value(
                                    slot_spec.name,
                                    entities,
                                    world,
                                    entity,
                                    time_index,
                                    current_value,
                                    rng,
                                )
                            else:
                                alt_value = current_value if rng.random() < 0.55 else _choose_value(
                                    slot_spec.name,
                                    entities,
                                    rng,
                                    current=stored_value,
                                )
                            alt_raw = render_raw_value(slot_spec.name, alt_value, rng)
                            alt_canonical = canonicalize_value(slot_spec.name, alt_raw)
                            memory_counter += 1
                            alt_memory = {
                                "memory_id": f"m{memory_counter:06d}",
                                "stress_name": stress_name,
                                "stress_value": float(level["stress"]),
                                "episode_id": episode_id,
                                "entity": entity,
                                "entity_alias": _noise_entity_alias(entity),
                                "slot": slot_spec.name,
                                "value_raw": alt_raw,
                                "value_canonical": alt_canonical,
                                "write_time": time_index,
                                "source_id": alt_source["source_id"],
                                "source_quality": float(alt_source["source_quality"]),
                                "write_correct": bool(alt_canonical == current_value),
                                "truth_at_write": current_value,
                            }
                            alt_memory["memory_text"] = _build_memory_text(alt_memory)
                            memories.append(alt_memory)

            episode_memories = [m for m in memories if m["episode_id"] == episode_id and m["stress_name"] == stress_name]
            query_candidates: list[tuple[float, str, str, int]] = []
            for time_index in range(1, int(dataset["time_steps"])):
                for entity in entities:
                    for slot_spec in slot_specs:
                        relevant_count = sum(
                            1
                            for memory in episode_memories
                            if memory["entity"] == entity
                            and memory["slot"] == slot_spec.name
                            and memory["write_time"] <= time_index
                        )
                        if relevant_count == 0:
                            continue
                        lag = time_index - last_change_snapshots[time_index][_slot_key(entity, slot_spec.name)]
                        weight = 1.0 + float(level["query_lag_bias"]) * float(lag + 1)
                        query_candidates.append((weight, entity, slot_spec.name, time_index))

            if not query_candidates:
                raise RuntimeError(f"No valid query candidates for episode {episode_id}.")

            for _ in range(int(dataset["queries_per_episode"])):
                total_weight = sum(weight for weight, _, _, _ in query_candidates)
                threshold = rng.random() * total_weight
                cumulative = 0.0
                chosen = query_candidates[-1]
                for candidate in query_candidates:
                    cumulative += candidate[0]
                    if cumulative >= threshold:
                        chosen = candidate
                        break
                _, entity, slot, query_time = chosen
                gold_value = world[query_time][entity][slot]
                query_counter += 1
                query_id = f"q{query_counter:06d}"
                relevant_ids: list[str] = []
                alive_ids: list[str] = []
                valid_ids: list[str] = []
                useful_ids: list[str] = []
                relevant_value_counts: dict[str, int] = {}
                valid_value_counts: dict[str, int] = {}
                for memory in episode_memories:
                    if memory["write_time"] > query_time:
                        continue
                    if memory["entity"] != entity or memory["slot"] != slot:
                        continue
                    relevant_ids.append(memory["memory_id"])
                    relevant_value_counts[memory["value_canonical"]] = relevant_value_counts.get(memory["value_canonical"], 0) + 1
                    latest_change = last_change_snapshots[query_time][_slot_key(entity, slot)]
                    alive = latest_change <= memory["write_time"]
                    if alive:
                        alive_ids.append(memory["memory_id"])
                    valid = bool(memory["write_correct"] and alive)
                    if valid:
                        valid_ids.append(memory["memory_id"])
                        useful_ids.append(memory["memory_id"])
                        valid_value_counts[memory["value_canonical"]] = valid_value_counts.get(memory["value_canonical"], 0) + 1
                latest_valid_memory_id = None
                if valid_ids:
                    latest_valid_memory_id = max(
                        valid_ids,
                        key=lambda memory_id: next(m["write_time"] for m in episode_memories if m["memory_id"] == memory_id),
                    )
                dominant_wrong_value = None
                dominant_wrong_support = 0
                for value, count in valid_value_counts.items():
                    if value == gold_value:
                        continue
                    if count > dominant_wrong_support:
                        dominant_wrong_value = value
                        dominant_wrong_support = count
                query = {
                    "query_id": query_id,
                    "stress_name": stress_name,
                    "stress_value": float(level["stress"]),
                    "episode_id": episode_id,
                    "entity": entity,
                    "slot": slot,
                    "query_time": query_time,
                    "query_text": _query_text(entity, slot),
                    "gold_value": gold_value,
                    "query_lag": query_time - last_change_snapshots[query_time][_slot_key(entity, slot)],
                    "distractor_overlap": float(level["distractor_overlap"]),
                    "world_change_scale": float(level["world_change_scale"]),
                    "write_error_rate": float(level["write_error_rate"]),
                    "conflict_rate": float(level["conflict_rate"]),
                }
                queries.append(query)
                label_rows.append(
                    {
                        "query_id": query_id,
                        "stress_name": stress_name,
                        "episode_id": episode_id,
                        "relevant_memory_ids": relevant_ids,
                        "alive_memory_ids": alive_ids,
                        "valid_memory_ids": valid_ids,
                        "useful_memory_ids": useful_ids,
                        "latest_valid_memory_id": latest_valid_memory_id,
                        "gold_value": gold_value,
                        "valid_value_support": valid_value_counts,
                        "relevant_value_support": relevant_value_counts,
                        "dominant_wrong_value": dominant_wrong_value,
                        "dominant_wrong_support": dominant_wrong_support,
                    }
                )

    world_rows.sort(key=lambda row: (row["stress_name"], row["episode_id"], row["time_step"], row["entity"], row["slot"]))
    memories.sort(key=lambda row: row["memory_id"])
    queries.sort(key=lambda row: row["query_id"])
    label_rows.sort(key=lambda row: row["query_id"])

    world_path = write_jsonl(data_dir / "world.jsonl", world_rows)
    memory_path = write_jsonl(data_dir / "memories.jsonl", memories)
    query_path = write_jsonl(data_dir / "queries.jsonl", queries)
    label_path = write_jsonl(data_dir / "exact_labels.jsonl", label_rows)
    manifest = {
        "world": str(world_path),
        "memories": str(memory_path),
        "queries": str(query_path),
        "exact_labels": str(label_path),
    }
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
