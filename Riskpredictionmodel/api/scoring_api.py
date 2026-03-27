from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import time

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import get_database_name, get_mongo_uri, init_env

init_env()

from .auth import require_api_key
from .cache import ApiCache
from .models import CustomerScoreRequest, ScoreRequest
from .request_builder import build_manual_request_frame as _build_manual_request_frame
from .response_builder import (
    _build_customer_summary_payload,
    _filter_customer_rows,
    _normalize_response_records,
    _shape_response_frame,
    _ts_to_iso,
    response_from_raw as _response_from_raw,
)
from .settings import load_api_settings
from ..data.segment_filters import filter_segment as _segment_filter
from ..dbconnect import get_database
from ..logging_config import get_logger
from ..pipeline.runner import score_mongo_frame
from ..scoring.model import PRODUCTION_RISK_REGISTRY_PATH, describe_active_production_model, load_production_artifacts
from ..scoring.performance import build_model_performance_payload


logger = get_logger(__name__)
SETTINGS = load_api_settings()
router = APIRouter(tags=["risk"])



def _resolve_threshold_override(segment: str) -> float | None:
    segment = (segment or "").strip().lower()
    if segment == "air" and SETTINGS.threshold_override_air is not None:
        return SETTINGS.threshold_override_air
    if segment == "ocean" and SETTINGS.threshold_override_ocean is not None:
        return SETTINGS.threshold_override_ocean
    return SETTINGS.threshold_override_global


_api_cache = ApiCache(
    dataset_ttl_seconds=SETTINGS.dataset_cache_ttl_seconds,
    history_ttl_seconds=SETTINGS.history_cache_ttl_seconds,
    auto_refresh_enabled=SETTINGS.auto_refresh_enabled,
    auto_refresh_interval_seconds=SETTINGS.auto_refresh_interval_seconds,
    scored_snapshot_retention_seconds=SETTINGS.scored_snapshot_retention_seconds,
    threshold_resolver=_resolve_threshold_override,
)



def _startup_checks() -> None:
    if not get_mongo_uri():
        raise RuntimeError("MONGO_URI is required but not set.")
    if not get_database_name():
        raise RuntimeError("DATABASE_NAME is required but not set.")
    load_production_artifacts()
    logger.info("Startup checks completed successfully")



def _check_mongo_live() -> bool:
    try:
        get_database().command("ping")
        return True
    except Exception:
        logger.warning("MongoDB ping failed", exc_info=True)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_checks()
    _api_cache.start()
    try:
        yield
    finally:
        _api_cache.stop()


def _prepare_history_frame(force_refresh: bool = False) -> pd.DataFrame:
    return _api_cache.load_full_dataset(force_refresh=force_refresh)



def _enrich_with_customer_history(df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    return _api_cache.enrich_with_customer_history(df, force_refresh=force_refresh)



def build_scored_frame(segment: str, limit: int | None = None, customer_id: str | None = None, force_refresh: bool = False) -> pd.DataFrame:
    full_df = _prepare_history_frame(force_refresh=force_refresh)
    if full_df.empty:
        return pd.DataFrame()

    segment_df = _segment_filter(full_df, segment, allow_all=True, missing="input")
    segment_df = _filter_customer_rows(segment_df, customer_id)
    if limit is not None and not segment_df.empty:
        segment_df = segment_df.iloc[:limit].copy()
    if segment_df.empty:
        return pd.DataFrame()

    segment_df = _enrich_with_customer_history(segment_df, force_refresh=force_refresh)
    scored = score_mongo_frame(
        segment_df,
        history_df=full_df,
        top_n=5,
        approval_threshold_override=_resolve_threshold_override(segment),
        scoring_context=f"live_mongo:{segment.lower()}",
    )
    return _response_from_raw(segment_df, scored)



def build_scored_dataset(segment: str, limit: int | None = None, customer_id: str | None = None, force_refresh: bool = False):
    frame = build_scored_frame(segment=segment, limit=limit, customer_id=customer_id, force_refresh=force_refresh)
    if frame.empty:
        return []
    return _normalize_response_records(_shape_response_frame(frame, response_mode="lean"))


@router.post("/score/{segment}")
def score(segment: str, payload: ScoreRequest, _auth: None = Depends(require_api_key)):
    logger.debug("Score request started segment=%s", segment)
    try:
        history_df = _prepare_history_frame()
        raw_df = _build_manual_request_frame(segment, payload)
        raw_df = _enrich_with_customer_history(raw_df)
        result = score_mongo_frame(
            raw_df,
            history_df=history_df,
            top_n=5,
            approval_threshold_override=_resolve_threshold_override(segment),
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
    limit: int = SETTINGS.score_all_page_default,
    cursor: str | None = None,
    refresh: bool = False,
    _auth: None = Depends(require_api_key),
):
    logger.debug("Score-all request started segment=%s limit=%s cursor=%s refresh=%s", segment, limit, bool(cursor), refresh)
    try:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")
        limit = min(limit, SETTINGS.score_all_page_max)
        descriptor = describe_active_production_model()
        payload = _api_cache.get_scored_page(
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


@router.post("/score-customer/{segment}")
def score_customer(segment: str, payload: CustomerScoreRequest, _auth: None = Depends(require_api_key)):
    logger.debug("Customer scoring request started segment=%s snapshot=%s", segment, bool(payload.snapshotId))
    try:
        customer_id = str(payload.customerId).strip()
        if not customer_id:
            raise HTTPException(status_code=400, detail="customer_id must be provided")
        if payload.limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")

        limit = min(int(payload.limit), 5000)
        descriptor = describe_active_production_model()
        snapshot_id = str(payload.snapshotId or "").strip() or None
        if snapshot_id:
            snapshot_df = _api_cache.get_snapshot_customer_frame(
                snapshot_id=snapshot_id,
                segment=segment,
                customer_id=customer_id,
                limit=limit,
            )
            records = _normalize_response_records(snapshot_df)
        else:
            records = build_scored_dataset(segment=segment, limit=limit, customer_id=customer_id)
        response = _build_customer_summary_payload(
            records,
            segment=segment,
            customer_id=customer_id,
            limit=limit,
            model_type=descriptor["model_type"],
        )
        logger.debug("Customer scoring request completed segment=%s rows=%s", segment, response.get("invoice_rows_scored"))
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Customer snapshot validation failed: {exc}")
    except Exception:
        logger.exception("Customer scoring failed for segment=%s", segment)
        raise HTTPException(status_code=500, detail="Customer scoring failed. Check server logs.")


@router.post("/cache/refresh")
def refresh_cache(_auth: None = Depends(require_api_key)):
    logger.debug("Cache refresh requested")
    try:
        _api_cache.refresh()
        snapshot = _api_cache.snapshot(now=time())
        return {
            "status": "ok",
            "message": "API cache refreshed from Mongo.",
            "dataset_ttl_seconds": SETTINGS.dataset_cache_ttl_seconds,
            "history_ttl_seconds": SETTINGS.history_cache_ttl_seconds,
            "active_snapshot_id": snapshot["scored_snapshot"]["snapshot_id"],
        }
    except Exception:
        logger.exception("Cache refresh failed")
        raise HTTPException(status_code=500, detail="Cache refresh failed. Check server logs.")


@router.get("/model-performance/{segment}")
def model_performance(
    segment: str,
    refresh: bool = False,
    _auth: None = Depends(require_api_key),
):
    logger.debug("Model-performance request started segment=%s refresh=%s", segment, refresh)
    try:
        descriptor = describe_active_production_model()
        artifacts = load_production_artifacts()
        live_status = "ok"
        live_error = None
        try:
            scored_frame, snapshot_meta = _api_cache.get_scored_segment_frame(
                segment=segment,
                refresh=refresh,
            )
        except Exception:
            logger.warning(
                "Model performance could not load live snapshot metrics for segment=%s; falling back to registry-only view",
                segment,
                exc_info=True,
            )
            scored_frame = pd.DataFrame()
            snapshot_meta = {
                "snapshot_id": None,
                "snapshot_generated_at": None,
                "rows": 0,
                "total_available": 0,
                "summary": {},
            }
            live_status = "unavailable"
            live_error = "MongoDB is unreachable or no scored snapshot is currently available. Showing registry metrics only."
        response = build_model_performance_payload(
            scored_df=scored_frame,
            descriptor=descriptor,
            registry_entry=artifacts.get("registry_entry", {}),
            segment=segment,
            snapshot_meta=snapshot_meta,
        )
        response["status"] = "ok" if live_status == "ok" else "degraded"
        response["live_status"] = live_status
        response["live_error"] = live_error
        logger.debug(
            "Model-performance request completed segment=%s rows=%s",
            segment,
            response.get("live_metrics", {}).get("rows"),
        )
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Model performance validation failed: {exc}")
    except Exception:
        logger.exception("Model performance failed for segment=%s", segment)
        raise HTTPException(status_code=500, detail="Model performance failed. Check server logs.")


@router.get("/health")
def health():
    try:
        snapshot = _api_cache.snapshot(now=time())
        descriptor = describe_active_production_model()
        mongo_live = _check_mongo_live()
        payload = {
            "status": "ok" if mongo_live else "degraded",
            "mongo": "ok" if mongo_live else "unreachable",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "production_model": {
                "model_family": descriptor["model_family"],
                "model_type": descriptor["model_type"],
                "model_version": descriptor["version"],
                "feature_count": descriptor["feature_count"],
                "threshold": descriptor["threshold"],
                "threshold_policy": descriptor["threshold_policy"],
                "registry_path": PRODUCTION_RISK_REGISTRY_PATH,
                "artifact_source": descriptor["artifact_source"],
                "threshold_overrides": {
                    "global": SETTINGS.threshold_override_global,
                    "air": SETTINGS.threshold_override_air,
                    "ocean": SETTINGS.threshold_override_ocean,
                },
            },
            "dataset_cache": {
                "ttl_seconds": SETTINGS.dataset_cache_ttl_seconds,
                "cached": bool(snapshot["dataset"]["ts"]),
                "last_refresh_utc": _ts_to_iso(snapshot["dataset"]["ts"]),
                "age_seconds": None if snapshot["dataset"]["age_seconds"] is None else round(float(snapshot["dataset"]["age_seconds"]), 3),
                "rows": snapshot["dataset"]["rows"],
                "model_key": snapshot["dataset"].get("model_key"),
            },
            "history_cache": {
                "ttl_seconds": SETTINGS.history_cache_ttl_seconds,
                "entries": snapshot["history"]["entries"],
                "max_age_seconds": None if snapshot["history"]["max_age_seconds"] is None else round(float(snapshot["history"]["max_age_seconds"]), 3),
            },
            "scored_snapshot": {
                "snapshot_id": snapshot["scored_snapshot"]["snapshot_id"],
                "generated_at": snapshot["scored_snapshot"]["generated_at"],
                "age_seconds": None if snapshot["scored_snapshot"]["age_seconds"] is None else round(float(snapshot["scored_snapshot"]["age_seconds"]), 3),
                "rows": snapshot["scored_snapshot"]["rows"],
                "raw_rows": snapshot["scored_snapshot"]["raw_rows"],
                "segment_counts": snapshot["scored_snapshot"]["segment_counts"],
                "previous_snapshot_id": snapshot["scored_snapshot"]["previous_snapshot_id"],
                "previous_snapshot_age_seconds": None
                if snapshot["scored_snapshot"]["previous_snapshot_age_seconds"] is None
                else round(float(snapshot["scored_snapshot"]["previous_snapshot_age_seconds"]), 3),
                "page_cache_entries": snapshot["scored_snapshot"]["page_cache_entries"],
            },
            "mongo_indexes": snapshot["mongo_indexes"],
            "auto_refresh": {
                "enabled": SETTINGS.auto_refresh_enabled,
                "interval_seconds": SETTINGS.auto_refresh_interval_seconds,
                "running": bool(snapshot["auto_refresh"].get("running", False)),
                "last_run_utc": _ts_to_iso(float(snapshot["auto_refresh"].get("last_run_ts", 0.0) or 0.0)),
                "last_error": snapshot["auto_refresh"].get("last_error"),
                "consecutive_errors": int(snapshot["auto_refresh"].get("consecutive_errors", 0) or 0),
                "max_consecutive_errors": _api_cache.max_consecutive_refresh_errors,
            },
        }
        if not mongo_live:
            return JSONResponse(status_code=503, content=payload)
        return payload
    except HTTPException:
        raise
    except Exception:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail="Health check failed. Check server logs.")


app = FastAPI(lifespan=lifespan)
if SETTINGS.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.cors_allowed_origins),
        allow_credentials=SETTINGS.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(router)
