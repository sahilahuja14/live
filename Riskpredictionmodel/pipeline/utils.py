from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bson import ObjectId


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, pd.Timestamp):
        return pd.isna(value)
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def get_nested(data: dict, path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(data: dict, path: str, value: Any, overwrite: bool = True) -> bool:
    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    leaf = parts[-1]
    if not overwrite and leaf in current and not is_missing(current[leaf]):
        return False
    current[leaf] = value
    return True


def flatten_dict(data: dict, prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_dict(value, full_key))
        else:
            output[full_key] = value
    return output


def stable_hash_int(*parts: Any, modulo: int | None = None) -> int:
    payload = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    value = int(digest[:16], 16)
    if modulo and modulo > 0:
        return value % modulo
    return value


def deterministic_choice(options: list[Any], *seed_parts: Any) -> Any:
    if not options:
        return None
    idx = stable_hash_int(*seed_parts, modulo=len(options))
    return options[idx]


def deterministic_weighted_choice(weighted: list[tuple[Any, float]], *seed_parts: Any) -> Any:
    if not weighted:
        return None
    total = sum(max(float(weight), 0.0) for _, weight in weighted)
    if total <= 0:
        return deterministic_choice([value for value, _ in weighted], *seed_parts)
    draw = stable_hash_int(*seed_parts, modulo=10_000_000) / 10_000_000 * total
    running = 0.0
    for value, weight in weighted:
        running += max(float(weight), 0.0)
        if draw <= running:
            return value
    return weighted[-1][0]


def safe_ratio(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except Exception:
        return float(default)
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float(default)
    return float(num / den)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    return target


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row)))
            handle.write("\n")
    return target
