from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import time

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import (
    get_live_db_name,
    get_live_invoice_collection,
    get_live_mongo_uri,
    init_env,
)

init_env()

from .auth import require_api_key
from .cache import ApiCache
from .models import CustomerScoreRequest, ScoreRequest
from .request_builder import build_manual_request_frame as _build_manual_request_frame
from .response_builder import (
    _build_customer_summary_payload,
    _build_customer_page_summary,
    _build_scored_summary,
    _filter_customer_rows,
    _normalize_response_records,
    _shape_response_frame,
    _ts_to_iso,
    build_customer_history_payload,
    build_customer_portfolio_frame,
    response_from_raw as _response_from_raw,
)
from .settings import load_api_settings
from ..data.segment_filters import filter_segment as _segment_filter
from ..dbconnect import get_live_database
from ..logging_config import get_logger
from ..pipeline.risk_canonical import canonicalize_risk_main_frame, get_live_diagnostics
from ..pipeline.runner import score_mongo_frame, score_mongo_frame_with_details
from ..scoring.model import PRODUCTION_RISK_REGISTRY_PATH, describe_active_production_model, load_production_artifacts
from ..scoring.performance import build_model_performance_payload


logger = get_logger(__name__)
SETTINGS = load_api_settings()
router = APIRouter(tags=["risk"])

# Keep a compatibility alias so existing tests and callers that still patch
# get_database do not break while the runtime stays live-db only.
get_database = get_live_database


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
    if not get_live_mongo_uri():
        raise RuntimeError("LIVE_MONGO_URI is required but not set.")
    if not get_live_db_name():
        raise RuntimeError("LIVE_DB_NAME is required but not set.")
    load_production_artifacts()
    logger.info("Startup checks completed successfully source_mode=live_collections")


def _check_live_mongo_live() -> bool:
    try:
        get_database().command("ping")
        return True
    except Exception:
        logger.warning("Live MongoDB ping failed", exc_info=True)
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


def _encode_cursor(payload: dict) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("utf-8")


def _decode_cursor(cursor: str) -> dict:
    try:
        decoded = base64.urlsafe_b64decode(str(cursor).encode("utf-8")).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Invalid cursor payload.")
    return payload


def _feature_quality_payload(details, input_rows: int, *, scoring_context: str | None = None) -> dict:
    validation = details.validation
    return {
        "feature_validation_passed": validation.is_valid,
        "scored_invoice_rows": int(len(details.scored_frame)),
        "dropped_invoice_rows": max(int(input_rows) - int(len(details.scored_frame)), 0),
        "missing_feature_count": int(len(validation.missing_columns) + len(validation.missing_features)),
        "invalid_object_feature_count": int(len(validation.invalid_object_features)),
        "invalid_datetime_feature_count": int(len(validation.invalid_datetime_features)),
        "scoring_context": scoring_context,
    }


def _history_preview_limit(payload: CustomerScoreRequest) -> int:
    if payload.limit is not None:
        return min(int(payload.limit), 100)
    return int(payload.historyPreviewLimit)


def _load_segment_history_frame(segment: str, *, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_df = _prepare_history_frame(force_refresh=force_refresh)
    segment_df = _segment_filter(full_df, segment, allow_all=True, missing="input")
    return full_df, segment_df


def _load_customer_invoice_frame(
    segment: str,
    customer_id: str,
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    full_df = _prepare_history_frame(force_refresh=force_refresh)
    customer_df = _filter_customer_rows(full_df, customer_id)
    if customer_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customer_id}' was not found in any segment.",
        )

    segment_customer_df = _segment_filter(customer_df, segment, allow_all=True, missing="input")
    if segment_customer_df.empty:
        available_segments = (
            customer_df.get("shipmentDetails.queryFor", pd.Series(dtype=object))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        available_segments = sorted(value for value in available_segments.unique().tolist() if value)
        segment_text = ", ".join(available_segments) if available_segments else "none"
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customer_id}' has no invoices in segment '{segment}'. Available segments: {segment_text}.",
        )

    customer_df = _enrich_with_customer_history(customer_df, force_refresh=force_refresh)
    return full_df, customer_df.reset_index(drop=True), int(len(segment_customer_df))


def _feature_snapshot_for_rows(scoring_frame: pd.DataFrame, page_indices: list[int], expected_features: list[str]) -> list[dict]:
    if scoring_frame.empty or not page_indices:
        return []
    columns = [column for column in ["invoice_key", *expected_features] if column in scoring_frame.columns]
    if not columns:
        return []
    snapshot = scoring_frame.iloc[page_indices][columns].copy().reset_index(drop=True)
    return _normalize_response_records(snapshot)


def _canonical_snapshot_for_rows(raw_frame: pd.DataFrame, page_indices: list[int]) -> list[dict]:
    if raw_frame.empty or not page_indices:
        return []
    canonical = canonicalize_risk_main_frame(raw_frame.copy())
    if canonical.empty:
        return []
    canonical = canonical.iloc[page_indices].copy().reset_index(drop=True)
    return _normalize_response_records(canonical)


def _score_customer_history(
    *,
    segment: str,
    customer_id: str,
    force_refresh: bool,
) -> dict:
    scoring_context = f"live_customer:{segment.lower()}:{customer_id}"
    history_df, customer_df, segment_invoice_rows = _load_customer_invoice_frame(
        segment,
        customer_id,
        force_refresh=force_refresh,
    )
    details = score_mongo_frame_with_details(
        customer_df,
        history_df=history_df,
        top_n=5,
        approval_threshold_override=_resolve_threshold_override(segment),
        scoring_context=scoring_context,
    )
    merged = _response_from_raw(customer_df, details.scored_frame)
    shaped = _shape_response_frame(merged, response_mode="lean")
    records = _normalize_response_records(shaped)
    if not records:
        raise HTTPException(
            status_code=500,
            detail="Customer scoring failed because no invoice rows passed feature validation.",
        )
    return {
        "history_df": history_df,
        "customer_df": customer_df,
        "segment_invoice_rows": int(segment_invoice_rows),
        "records": records,
        "scoring_frame": details.scoring_frame.reset_index(drop=True),
        "feature_quality": _feature_quality_payload(
            details,
            input_rows=len(customer_df),
            scoring_context=scoring_context,
        ),
    }


def _slice_page(items, *, page_size: int, offset: int):
    total_available = int(len(items))
    end_offset = offset + int(page_size)
    if isinstance(items, pd.DataFrame):
        page_items = items.iloc[offset:end_offset].copy().reset_index(drop=True)
    else:
        page_items = list(items[offset:end_offset])
    returned = int(len(page_items))
    next_offset = offset + returned
    return page_items, total_available, returned, next_offset


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
    logger.debug("Customer scoring request started segment=%s", segment)
    try:
        customer_id = str(payload.customerId).strip()
        if not customer_id:
            raise HTTPException(status_code=400, detail="customer_id must be provided")
        descriptor = describe_active_production_model()
        preview_limit = _history_preview_limit(payload)
        customer_result = _score_customer_history(
            segment=segment,
            customer_id=customer_id,
            force_refresh=bool(payload.refresh),
        )
        response = _build_customer_summary_payload(
            customer_result["records"],
            segment=segment,
            customer_id=customer_id,
            history_preview_limit=preview_limit,
            model_type=descriptor["model_type"],
            feature_quality=customer_result["feature_quality"],
            include_history_preview=payload.includeHistoryPreview,
            include_invoice_top_features=payload.includeTopInvoiceFeatures,
            segment_invoice_rows=customer_result["segment_invoice_rows"],
            approval_threshold_override=_resolve_threshold_override(segment),
        )
        pd_trace = response.get("customer_summary", {}).get("pd_computation_trace", {})
        logger.info(
            "Customer scoring completed segment=%s customer_id=%s invoices=%s pd=%.6f path=%s",
            segment,
            customer_id,
            response.get("customer_summary", {}).get("invoice_rows_scored"),
            float(response.get("customer_summary", {}).get("pd", 0.0) or 0.0),
            pd_trace.get("path"),
        )
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Customer scoring validation failed: {exc}")
    except Exception:
        logger.exception("Customer scoring failed for segment=%s", segment)
        raise HTTPException(status_code=500, detail="Customer scoring failed. Check server logs.")


@router.get("/score-customers/{segment}")
def score_customers(
    segment: str,
    limit: int = SETTINGS.score_all_page_default,
    cursor: str | None = None,
    refresh: bool = False,
    _auth: None = Depends(require_api_key),
):
    logger.debug("Customer list request started segment=%s limit=%s cursor=%s refresh=%s", segment, limit, bool(cursor), refresh)
    try:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")
        if cursor and refresh:
            raise HTTPException(status_code=400, detail="refresh cannot be combined with cursor")

        # limit here is explicitly a customer page size, not an invoice row limit.
        customer_page_size = min(int(limit), SETTINGS.score_all_page_max)
        offset = 0
        expected_snapshot_id = None
        if cursor:
            cursor_payload = _decode_cursor(cursor)
            if str(cursor_payload.get("segment") or "").strip().lower() != str(segment or "").strip().lower():
                raise HTTPException(status_code=422, detail="Cursor segment does not match request segment.")
            if int(cursor_payload.get("page_size") or 0) != customer_page_size:
                raise HTTPException(status_code=422, detail="Cursor page size does not match request limit.")
            offset = max(int(cursor_payload.get("offset") or 0), 0)
            expected_snapshot_id = str(cursor_payload.get("snapshot_id") or "").strip() or None

        snapshot_frame, snapshot_meta = _api_cache.get_scored_segment_frame(segment=segment, refresh=refresh and not cursor)
        snapshot_id = str(snapshot_meta.get("snapshot_id") or "").strip()
        if expected_snapshot_id and snapshot_id != expected_snapshot_id:
            raise HTTPException(status_code=422, detail="Snapshot changed. Restart customer pagination without a cursor.")

        segment_invoice_counts: dict[str, int] | None = None
        customer_source_frame = snapshot_frame
        if "customer.customerId" in snapshot_frame.columns:
            segment_customer_frame = snapshot_frame.copy()
            segment_customer_frame["customer.customerId"] = (
                segment_customer_frame["customer.customerId"].fillna("").astype(str).str.strip()
            )
            segment_customer_frame = segment_customer_frame[
                segment_customer_frame["customer.customerId"] != ""
            ].copy()
            if not segment_customer_frame.empty:
                segment_invoice_counts = (
                    segment_customer_frame.groupby("customer.customerId").size().astype(int).to_dict()
                )
                if str(segment or "").strip().lower() != "all":
                    all_frame, all_snapshot_meta = _api_cache.get_scored_segment_frame(segment="all", refresh=False)
                    all_snapshot_id = str(all_snapshot_meta.get("snapshot_id") or "").strip()
                    if all_snapshot_id and snapshot_id and all_snapshot_id != snapshot_id:
                        logger.warning(
                            "Customer list snapshot ids differ between segment=%s and all snapshot segment_id=%s all_id=%s",
                            segment,
                            snapshot_id,
                            all_snapshot_id,
                        )
                    all_customer_frame = all_frame.copy()
                    if "customer.customerId" in all_customer_frame.columns:
                        all_customer_frame["customer.customerId"] = (
                            all_customer_frame["customer.customerId"].fillna("").astype(str).str.strip()
                        )
                        customer_ids = list(segment_invoice_counts.keys())
                        customer_source_frame = all_customer_frame[
                            all_customer_frame["customer.customerId"].isin(customer_ids)
                        ].copy()
                    else:
                        customer_source_frame = pd.DataFrame(columns=snapshot_frame.columns)

        customer_frame = _api_cache.get_customer_portfolio(
            segment=segment,
            snapshot_id=snapshot_id,
            builder=lambda: build_customer_portfolio_frame(
                customer_source_frame,
                segment=segment,
                segment_invoice_counts=segment_invoice_counts,
                approval_threshold_override=_resolve_threshold_override(segment),
            ),
        )
        page_frame, total_available, returned, next_offset = _slice_page(
            customer_frame,
            page_size=customer_page_size,
            offset=offset,
        )
        next_cursor = None
        if next_offset < total_available:
            next_cursor = _encode_cursor(
                {
                    "snapshot_id": snapshot_id,
                    "segment": str(segment or "").strip().lower(),
                    "page_size": customer_page_size,
                    "offset": next_offset,
                }
            )

        descriptor = describe_active_production_model()
        response = {
            "segment": str(segment or "").strip().lower(),
            "model_type": descriptor["model_type"],
            "model_family": descriptor["model_family"],
            "model_version": descriptor["version"],
            "count": returned,
            "limit": customer_page_size,
            "customer_limit_applied": customer_page_size,
            "customers_returned": returned,
            "total_customers_available": total_available,
            "summary": _build_customer_page_summary(page_frame),
            "snapshot_summary": _build_customer_page_summary(customer_frame),
            "total_available": total_available,
            "pagination": {
                "snapshot_id": snapshot_id,
                "snapshot_generated_at": snapshot_meta.get("snapshot_generated_at"),
                "returned": returned,
                "has_more": next_cursor is not None,
                "next_cursor": next_cursor,
            },
            "records": _normalize_response_records(page_frame),
        }
        logger.info(
            "Customer list completed segment=%s customers=%s total_customers=%s snapshot_id=%s",
            segment,
            returned,
            total_available,
            snapshot_id,
        )
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Customer list validation failed: {exc}")
    except Exception:
        logger.exception("Customer list scoring failed for segment=%s", segment)
        raise HTTPException(status_code=500, detail="Customer list scoring failed. Check server logs.")


@router.get("/customer-history/{segment}")
def customer_history(
    segment: str,
    customer_id: str,
    limit: int = 100,
    cursor: str | None = None,
    include_features: bool = False,
    include_canonical: bool = False,
    refresh: bool = False,
    _auth: None = Depends(require_api_key),
):
    logger.debug(
        "Customer history request started segment=%s customer_id=%s limit=%s cursor=%s refresh=%s",
        segment,
        customer_id,
        limit,
        bool(cursor),
        refresh,
    )
    try:
        customer_key = str(customer_id).strip()
        if not customer_key:
            raise HTTPException(status_code=400, detail="customer_id must be provided")
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1")
        if cursor and refresh:
            raise HTTPException(status_code=400, detail="refresh cannot be combined with cursor")

        page_size = min(int(limit), 200)
        offset = 0
        if cursor:
            cursor_payload = _decode_cursor(cursor)
            if str(cursor_payload.get("segment") or "").strip().lower() != str(segment or "").strip().lower():
                raise HTTPException(status_code=422, detail="Cursor segment does not match request segment.")
            if str(cursor_payload.get("customer_id") or "").strip() != customer_key:
                raise HTTPException(status_code=422, detail="Cursor customer_id does not match request customer_id.")
            if int(cursor_payload.get("page_size") or 0) != page_size:
                raise HTTPException(status_code=422, detail="Cursor page size does not match request limit.")
            offset = max(int(cursor_payload.get("offset") or 0), 0)

        customer_result = _score_customer_history(
            segment=segment,
            customer_id=customer_key,
            force_refresh=refresh and not cursor,
        )
        page_records, total_available, returned, next_offset = _slice_page(
            customer_result["records"],
            page_size=page_size,
            offset=offset,
        )
        next_cursor = None
        if next_offset < total_available:
            next_cursor = _encode_cursor(
                {
                    "segment": str(segment or "").strip().lower(),
                    "customer_id": customer_key,
                    "page_size": page_size,
                    "offset": next_offset,
                }
            )

        descriptor = describe_active_production_model()
        customer_payload = _build_customer_summary_payload(
            customer_result["records"],
            segment=segment,
            customer_id=customer_key,
            history_preview_limit=page_size,
            model_type=descriptor["model_type"],
            feature_quality=customer_result["feature_quality"],
            include_history_preview=False,
            segment_invoice_rows=customer_result["segment_invoice_rows"],
            approval_threshold_override=_resolve_threshold_override(segment),
        )

        artifacts = load_production_artifacts()
        page_indices = list(range(offset, offset + returned))
        feature_snapshot = (
            _feature_snapshot_for_rows(customer_result["scoring_frame"], page_indices, artifacts["features"])
            if include_features
            else []
        )
        canonical_snapshot = (
            _canonical_snapshot_for_rows(customer_result["customer_df"], page_indices)
            if include_canonical
            else []
        )

        response = build_customer_history_payload(
            segment=segment,
            customer_id=customer_key,
            customer_summary=customer_payload["customer_summary"],
            records=page_records,
            total_available=total_available,
            returned=returned,
            next_cursor=next_cursor,
            snapshot_meta={
                "source_mode": "live_customer_history",
                "segment": str(segment or "").strip().lower(),
                "model_type": descriptor["model_type"],
                "model_family": descriptor["model_family"],
                "model_version": descriptor["version"],
                "refresh_applied": bool(refresh and not cursor),
                "feature_quality": customer_result["feature_quality"],
            },
            feature_snapshot=feature_snapshot,
            canonical_snapshot=canonical_snapshot,
        )
        logger.info(
            "Customer history completed segment=%s customer_id=%s returned=%s total=%s",
            segment,
            customer_key,
            returned,
            total_available,
        )
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Customer history validation failed: {exc}")
    except Exception:
        logger.exception("Customer history failed for segment=%s customer_id=%s", segment, customer_id)
        raise HTTPException(status_code=500, detail="Customer history failed. Check server logs.")


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
        live_ping = bool(get_live_mongo_uri() and get_live_db_name()) and _check_live_mongo_live()
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
        if not active_ping:
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
