from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Iterable


FIRST_NAMES = [
    "Ava",
    "Mia",
    "Liam",
    "Noah",
    "Emma",
    "Sophia",
    "Lucas",
    "Olivia",
    "Ethan",
    "Ella",
    "Ivy",
    "Mason",
]

LAST_NAMES = [
    "Stone",
    "Stark",
    "Reed",
    "Ward",
    "Blake",
    "Brooks",
    "Hayes",
    "Parker",
    "Carter",
    "Cole",
    "Shaw",
    "Lane",
]

CITY_VALUES = [
    "new york city",
    "san francisco",
    "berlin",
    "london",
    "toronto",
    "seattle",
    "singapore",
    "boston",
]

EMPLOYER_VALUES = [
    "openai",
    "google",
    "anthropic",
    "microsoft",
    "meta",
    "nvidia",
    "amazon",
    "deepmind",
]

TITLE_VALUES = [
    "research engineer",
    "staff scientist",
    "product manager",
    "software engineer",
    "design lead",
    "operations analyst",
]

SCHOOL_VALUES = [
    "mit",
    "stanford",
    "cmu",
    "berkeley",
    "eth zurich",
    "tsinghua",
]

PROJECT_VALUES = [
    "atlas",
    "beacon",
    "comet",
    "delta",
    "ember",
    "flux",
    "helios",
]

DATE_VALUES = [
    date(1988, 3, 17),
    date(1990, 6, 2),
    date(1992, 11, 14),
    date(1994, 1, 29),
    date(1995, 9, 7),
    date(1997, 12, 21),
]

ALIASES: dict[str, dict[str, list[str]]] = {
    "city": {
        "new york city": ["new york city", "nyc", "new york"],
        "san francisco": ["san francisco", "sf", "san fran"],
        "berlin": ["berlin", "berlin, germany"],
        "london": ["london", "london uk"],
        "toronto": ["toronto", "toronto, canada"],
        "seattle": ["seattle", "seattle wa"],
        "singapore": ["singapore", "singapore city"],
        "boston": ["boston", "boston ma"],
    },
    "birth_city": {
        "new york city": ["new york city", "nyc", "new york"],
        "san francisco": ["san francisco", "sf"],
        "berlin": ["berlin", "berlin, germany"],
        "london": ["london", "london uk"],
        "toronto": ["toronto", "toronto, canada"],
        "seattle": ["seattle", "seattle wa"],
        "singapore": ["singapore", "singapore city"],
        "boston": ["boston", "boston ma"],
    },
    "employer": {
        "openai": ["openai", "open ai", "openai inc"],
        "google": ["google", "google llc"],
        "anthropic": ["anthropic", "anthropic pbc"],
        "microsoft": ["microsoft", "microsoft corp"],
        "meta": ["meta", "meta platforms"],
        "nvidia": ["nvidia", "nvidia corp"],
        "amazon": ["amazon", "amazon.com"],
        "deepmind": ["deepmind", "google deepmind"],
    },
    "school": {
        "mit": ["mit", "massachusetts institute of technology"],
        "stanford": ["stanford", "stanford university"],
        "cmu": ["cmu", "carnegie mellon"],
        "berkeley": ["berkeley", "uc berkeley"],
        "eth zurich": ["eth zurich", "eth"],
        "tsinghua": ["tsinghua", "tsinghua university"],
    },
    "title": {},
    "project": {},
    "birthday": {},
    "manager": {},
}


@dataclass(frozen=True)
class SlotSpec:
    name: str
    volatility: float
    category: str


DEFAULT_SLOTS = [
    SlotSpec("birthday", 0.001, "stable"),
    SlotSpec("birth_city", 0.003, "stable"),
    SlotSpec("school", 0.015, "medium"),
    SlotSpec("employer", 0.035, "medium"),
    SlotSpec("title", 0.045, "medium"),
    SlotSpec("city", 0.070, "volatile"),
    SlotSpec("project", 0.085, "volatile"),
    SlotSpec("manager", 0.090, "volatile"),
]


def list_slot_specs() -> list[SlotSpec]:
    return list(DEFAULT_SLOTS)


def build_entity_names(num_entities: int) -> list[str]:
    candidates = [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]
    if num_entities > len(candidates):
        raise ValueError(f"Requested {num_entities} entities but only {len(candidates)} available.")
    return candidates[:num_entities]


def list_slot_values(slot: str, entities: Iterable[str]) -> list[str]:
    if slot in {"city", "birth_city"}:
        return list(CITY_VALUES)
    if slot == "employer":
        return list(EMPLOYER_VALUES)
    if slot == "title":
        return list(TITLE_VALUES)
    if slot == "school":
        return list(SCHOOL_VALUES)
    if slot == "project":
        return list(PROJECT_VALUES)
    if slot == "birthday":
        return [value.isoformat() for value in DATE_VALUES]
    if slot == "manager":
        return list(entities)
    raise KeyError(f"Unsupported slot: {slot}")


def render_raw_value(slot: str, canonical_value: str, rng) -> str:
    if slot == "birthday":
        year, month, day = canonical_value.split("-")
        formats = [
            canonical_value,
            f"{month}/{day}/{year}",
            f"{year}.{month}.{day}",
            f"{day}-{month}-{year}",
        ]
        return formats[rng.randrange(len(formats))]
    aliases = ALIASES.get(slot, {}).get(canonical_value, [])
    if aliases:
        return aliases[rng.randrange(len(aliases))]
    variants = [
        canonical_value,
        canonical_value.title(),
        canonical_value.upper(),
        canonical_value.replace(" ", "-"),
    ]
    return variants[rng.randrange(len(variants))]


def canonicalize_value(slot: str, raw_value: str) -> str:
    original = raw_value.strip().lower()
    if slot == "birthday":
        compact = re.sub(r"\s+", "", original)
        match = re.match(r"^(\d{4})[.\-/](\d{2})[.\-/](\d{2})$", compact)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        match = re.match(r"^(\d{2})[.\-/](\d{2})[.\-/](\d{4})$", compact)
        if match:
            return f"{match.group(3)}-{match.group(1)}-{match.group(2)}"
    cleaned = re.sub(r"[\(\),]", " ", original)
    cleaned = re.sub(r"[-_/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for canonical, aliases in ALIASES.get(slot, {}).items():
        normalized_aliases = {re.sub(r"\s+", " ", alias.strip().lower()).strip() for alias in aliases}
        if cleaned in normalized_aliases:
            return canonical
    return cleaned

