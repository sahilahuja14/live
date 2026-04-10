from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.production_registry import ValidationResult, validate_feature_frame
from ..logging_config import get_logger
from ..scoring.model import get_active_production_model_family, load_production_artifacts, score_production_frame
from .risk_main import build_risk_main_scoring_frame


logger = get_logger(__name__)


@dataclass(frozen=True)
class ScoredFrameDetails:
    scored_frame: pd.DataFrame
    scoring_frame: pd.DataFrame
    validation: ValidationResult


def _validation_message(validation) -> str:
    parts = []
    if validation.missing_columns:
        parts.append(f"missing canonical columns={validation.missing_columns}")
    if validation.missing_features:
        preview = validation.missing_features[:10]
        parts.append(f"missing model features(sample)={preview} total={len(validation.missing_features)}")
    if validation.invalid_datetime_features:
        parts.append(f"datetime features={validation.invalid_datetime_features}")
    if validation.invalid_object_features:
        parts.append(f"object features={validation.invalid_object_features}")
    return "; ".join(parts) if parts else "feature frame is valid"


def score_mongo_frame(
    current_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    *,
    top_n: int = 5,
    approval_threshold_override: float | None = None,
    scoring_context: str | None = None,
) -> pd.DataFrame:
    details = score_mongo_frame_with_details(
        current_df,
        history_df=history_df,
        top_n=top_n,
        approval_threshold_override=approval_threshold_override,
        scoring_context=scoring_context,
    )
    return details.scored_frame


def score_mongo_frame_with_details(
    current_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    *,
    top_n: int = 5,
    approval_threshold_override: float | None = None,
    scoring_context: str | None = None,
) -> ScoredFrameDetails:
    model_family = get_active_production_model_family()
    logger.debug("Scoring frame requested model_family=%s rows=%d", model_family, len(current_df))

    # Temporary diagnostics to surface numeric overflows during feature build.
    previous_err = np.seterr(over="warn", invalid="warn")
    try:
        scoring_frame = build_risk_main_scoring_frame(current_df, history_df=history_df)
    finally:
        np.seterr(**previous_err)

    _log_nonfinite_columns(scoring_frame, scoring_context=scoring_context)
    artifacts = load_production_artifacts()
    validation = validate_feature_frame(scoring_frame, artifacts["features"])
    if not validation.is_valid:
        message = _validation_message(validation)
        raise ValueError(f"Feature frame validation failed: {message}")

    scored_frame = score_production_frame(
        scoring_frame,
        top_n=top_n,
        approval_threshold_override=approval_threshold_override,
        scoring_context=scoring_context,
    )
    return ScoredFrameDetails(
        scored_frame=scored_frame,
        scoring_frame=scoring_frame,
        validation=validation,
    )


def _log_nonfinite_columns(frame: pd.DataFrame, *, scoring_context: str | None = None) -> None:
    if frame.empty:
        return
    numeric_frame = frame.select_dtypes(include=["number"])
    if numeric_frame.empty:
        return
    # Drop numeric columns that are known to be null-encoded placeholders.
    numeric_frame = numeric_frame.dropna(axis=1, how="all")
    if numeric_frame.empty:
        return

    # Some identifier-like columns can be inferred as numeric when mostly empty.
    # Filter out columns that still contain any string-like values in the source.
    safe_columns = []
    for column in numeric_frame.columns:
        source_col = frame[column]
        if source_col.dropna().map(lambda value: isinstance(value, str)).any():
            continue
        safe_columns.append(column)
    if not safe_columns:
        return
    numeric_frame = numeric_frame[safe_columns]
    if numeric_frame.empty:
        return

    nonfinite = ~np.isfinite(numeric_frame.to_numpy())
    if not nonfinite.any():
        return

    bad_cols = list(numeric_frame.columns[nonfinite.any(axis=0)])
    context = f" context={scoring_context}" if scoring_context else ""
    logger.warning("Non-finite values detected in scoring frame%s columns=%s", context, bad_cols)
