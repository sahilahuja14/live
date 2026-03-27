from __future__ import annotations

import pandas as pd

from ..logging_config import get_logger
from .risk_main import build_risk_main_scoring_frame
from ..scoring.model import get_active_production_model_family, score_production_frame


logger = get_logger(__name__)



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
    return score_production_frame(
        scoring_frame,
        top_n=top_n,
        approval_threshold_override=approval_threshold_override,
        scoring_context=scoring_context,
    )
