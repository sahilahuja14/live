from __future__ import annotations

import pandas as pd

from ..config import get_source_mode
from ..features.production_registry import validate_feature_frame
from ..logging_config import get_logger
from ..scoring.model import get_active_production_model_family, load_production_artifacts, score_production_frame
from .risk_main import build_risk_main_scoring_frame


logger = get_logger(__name__)


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
    model_family = get_active_production_model_family()
    logger.debug("Scoring frame requested model_family=%s rows=%d", model_family, len(current_df))
    scoring_frame = build_risk_main_scoring_frame(current_df, history_df=history_df)
    artifacts = load_production_artifacts()
    validation = validate_feature_frame(scoring_frame, artifacts["features"])
    source_mode = get_source_mode()
    if not validation.is_valid:
        message = _validation_message(validation)
        if source_mode == "live_collections":
            raise ValueError(f"Live feature frame validation failed: {message}")
        logger.warning(
            "Feature frame gaps detected source_mode=%s %s domain_gaps=%s",
            source_mode,
            message,
            validation.domain_gaps,
        )
    return score_production_frame(
        scoring_frame,
        top_n=top_n,
        approval_threshold_override=approval_threshold_override,
        scoring_context=scoring_context,
    )
