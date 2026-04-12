from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict] = []
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_csv(path: str | Path, rows: Sequence[dict]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

