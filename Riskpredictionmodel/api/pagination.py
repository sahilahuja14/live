from __future__ import annotations

import base64
import json
from typing import TypeVar

import pandas as pd


T = TypeVar("T")


def encode_cursor(payload: dict) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("utf-8")


def decode_cursor(cursor: str) -> dict:
    try:
        decoded = base64.urlsafe_b64decode(str(cursor).encode("utf-8")).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid cursor payload.")
    return payload


def slice_page(items: pd.DataFrame | list[T], *, page_size: int, offset: int):
    total_available = int(len(items))
    end_offset = int(offset) + int(page_size)
    if isinstance(items, pd.DataFrame):
        page_items = items.iloc[offset:end_offset].copy().reset_index(drop=True)
    else:
        page_items = list(items[offset:end_offset])
    returned = int(len(page_items))
    next_offset = int(offset) + returned
    return page_items, total_available, returned, next_offset
