from __future__ import annotations

from datetime import datetime, timezone
from time import time

import pandas as pd
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse

from ...config import get_live_db_name, get_live_invoice_collection, get_live_mongo_uri
from ...pipeline.risk_canonical import get_live_diagnostics
from ...scoring.model import PRODUCTION_RISK_REGISTRY_PATH, describe_active_production_model, load_production_artifacts
from ...scoring.performance import build_model_performance_payload
from ..auth import require_api_key
from ..response_builder import _ts_to_iso


def register_system_routes(
    *,
    router,
    logger,
    settings,
    api_cache,
    score_broadcaster,
    check_live_mongo_live,
):
    @router.post("/cache/refresh")
    def refresh_cache(_auth: None = Depends(require_api_key)):
        logger.debug("Cache refresh requested")
        try:
            score_broadcaster.notify_refresh_started_threadsafe("http_refresh")
            api_cache.refresh(trigger="http_refresh")
            snapshot = api_cache.snapshot(now=time())
            return {
                "status": "ok",
                "message": "API cache refreshed from Mongo.",
                "dataset_ttl_seconds": settings.dataset_cache_ttl_seconds,
                "history_ttl_seconds": settings.history_cache_ttl_seconds,
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
                scored_frame, snapshot_meta = api_cache.get_scored_segment_frame(
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
            snapshot = api_cache.snapshot(now=time())
            descriptor = describe_active_production_model()
            live_ping = bool(get_live_mongo_uri() and get_live_db_name()) and check_live_mongo_live()
            active_ping = live_ping
            live_diag = get_live_diagnostics()
            payload = {
                "status": "ok" if active_ping else "degraded",
                "mongo": "ok" if active_ping else "unreachable",
                "source_mode": "live_collections",
                "risk_main_ping": False,
                "live_ping": live_ping,
                "live_collections_found": live_diag.get("live_collections_found", []),
                "live_coverage": live_diag.get("coverage", {}),
                "live_invoice_collection": get_live_invoice_collection(),
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
                        "global": settings.threshold_override_global,
                        "air": settings.threshold_override_air,
                        "ocean": settings.threshold_override_ocean,
                    },
                },
                "dataset_cache": {
                    "ttl_seconds": settings.dataset_cache_ttl_seconds,
                    "cached": bool(snapshot["dataset"]["ts"]),
                    "last_refresh_utc": _ts_to_iso(snapshot["dataset"]["ts"]),
                    "age_seconds": None if snapshot["dataset"]["age_seconds"] is None else round(float(snapshot["dataset"]["age_seconds"]), 3),
                    "rows": snapshot["dataset"]["rows"],
                    "model_key": snapshot["dataset"].get("model_key"),
                },
                "history_cache": {
                    "ttl_seconds": settings.history_cache_ttl_seconds,
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
                    "enabled": settings.auto_refresh_enabled,
                    "interval_seconds": settings.auto_refresh_interval_seconds,
                    "running": bool(snapshot["auto_refresh"].get("running", False)),
                    "last_run_utc": _ts_to_iso(float(snapshot["auto_refresh"].get("last_run_ts", 0.0) or 0.0)),
                    "last_error": snapshot["auto_refresh"].get("last_error"),
                    "consecutive_errors": int(snapshot["auto_refresh"].get("consecutive_errors", 0) or 0),
                    "max_consecutive_errors": api_cache.max_consecutive_refresh_errors,
                },
            }
            if not active_ping:
                return JSONResponse(status_code=503, content=payload)
            return payload
        except HTTPException:
            raise
        except Exception:
            logger.exception("Health check failed")
            raise HTTPException(status_code=500, detail="Health check failed. Check server logs.")

    return {
        "refresh_cache": refresh_cache,
        "model_performance": model_performance,
        "health": health,
    }
