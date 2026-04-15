from __future__ import annotations

from fastapi import Depends, HTTPException

from ...scoring.model import describe_active_production_model
from ..auth import require_api_key
from ..models import ScoreRequest
from ..request_builder import build_manual_request_frame as _build_manual_request_frame
from ..response_builder import (
    _normalize_response_records,
    _shape_response_frame,
    response_from_raw as _response_from_raw,
)
from ...pipeline.runner import score_mongo_frame


def register_scoring_routes(
    *,
    router,
    logger,
    settings,
    api_cache,
    threshold_resolver,
    prepare_history_frame,
    enrich_with_customer_history,
    build_scored_frame,
    build_scored_dataset,
):
    @router.post("/score/{segment}")
    def score(segment: str, payload: ScoreRequest, _auth: None = Depends(require_api_key)):
        logger.debug("Score request started segment=%s", segment)
        try:
            history_df = prepare_history_frame(api_cache)
            raw_df = _build_manual_request_frame(segment, payload)
            raw_df = enrich_with_customer_history(api_cache, raw_df)
            result = score_mongo_frame(
                raw_df,
                history_df=history_df,
                top_n=5,
                approval_threshold_override=threshold_resolver(segment),
                scoring_context=f"live_manual:{segment.lower()}",
            )
            merged = _response_from_raw(raw_df, result)
            response = _normalize_response_records(_shape_response_frame(merged, response_mode="lean"))[0]
            logger.debug("Score request completed segment=%s", segment)
            return response
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Input/schema validation failed: {exc}")
        except Exception:
            logger.exception("Scoring failed for segment=%s", segment)
            raise HTTPException(status_code=500, detail="Scoring failed. Check server logs.")

    @router.get("/score-all/{segment}")
    def score_all(
        segment: str,
        limit: int = settings.score_all_page_default,
        cursor: str | None = None,
        refresh: bool = False,
        _auth: None = Depends(require_api_key),
    ):
        logger.debug("Score-all request started segment=%s limit=%s cursor=%s refresh=%s", segment, limit, bool(cursor), refresh)
        try:
            if limit < 1:
                raise HTTPException(status_code=400, detail="limit must be >= 1")
            limit = min(limit, settings.score_all_page_max)
            descriptor = describe_active_production_model()
            payload = api_cache.get_scored_page(
                segment=segment,
                page_size=limit,
                cursor=cursor,
                refresh=refresh,
            )
            response = {
                "segment": segment.lower(),
                "model_type": descriptor["model_type"],
                "model_family": descriptor["model_family"],
                "model_version": descriptor["version"],
                "count": payload["count"],
                "limit": payload["limit"],
                "summary": payload["summary"],
                "snapshot_summary": payload["snapshot_summary"],
                "total_available": payload["total_available"],
                "pagination": payload["pagination"],
                "records": payload["records"],
            }
            logger.debug("Score-all request completed segment=%s returned=%s", segment, response["count"])
            return response
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Input/schema validation failed: {exc}")
        except Exception:
            logger.exception("Bulk scoring failed for segment=%s", segment)
            raise HTTPException(status_code=500, detail="Bulk scoring failed. Check server logs.")

    return {
        "score": score,
        "score_all": score_all,
        "build_scored_frame": build_scored_frame,
        "build_scored_dataset": build_scored_dataset,
    }
