from __future__ import annotations

import pandas as pd

from ..features.registry import SEGMENT_COL


def filter_segment(
    df: pd.DataFrame,
    segment: str,
    *,
    allow_all: bool = True,
    missing: str = "empty",
) -> pd.DataFrame:
    frame = df.copy()
    segment_value = (segment or "").strip().lower()
    if allow_all and segment_value == "all":
        return frame
    if SEGMENT_COL not in frame.columns:
        if missing == "input":
            return frame
        return pd.DataFrame(columns=frame.columns)
    mask = frame[SEGMENT_COL].fillna("").astype(str).str.lower() == segment_value
    return frame.loc[mask].copy()
