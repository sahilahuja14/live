from __future__ import annotations

from fastapi import Depends, HTTPException, Response

from ...scoring.model import describe_active_production_model
from ..auth import require_api_key
from ..models import CustomerScoreRequest
from ..pagination import decode_cursor, encode_cursor
from ..response_builder import (
    _build_customer_page_summary,
    _build_customer_summary_payload,
    _normalize_response_records,
    build_customer_profile_payload,
)


def register_customer_routes(
    *,
    router,
    logger,
    settings,
    api_cache,
    customer_service,
    threshold_resolver,
    history_preview_limit,
    load_customer_portfolio_page_from_store,
    load_customer_summary_or_bootstrap,
    resolve_customer_lookup_input,
    resolve_customer_lookup_key,
    clean_customer_portfolio_frame,
    build_customer_history_response,
):
    @router.get("/customer/{segment}")
    def customer(
        segment: str,
        customer_id: str | None = None,
        customer_name: str | None = None,
        invoice_no: str | None = None,
        query: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        include_features: bool = False,
        include_canonical: bool = False,
        include_history: bool = False,
        refresh: bool = False,
        _auth: None = Depends(require_api_key),
    ):
        logger.debug(
            "Unified customer request started segment=%s customer_id=%s customer_name=%s invoice_no=%s query=%s limit=%s cursor=%s refresh=%s",
            segment,
            customer_id,
            customer_name,
            invoice_no,
            query,
            limit,
            bool(cursor),
            refresh,
        )
        try:
            if limit < 1:
                raise HTTPException(status_code=400, detail="limit must be >= 1")
            if cursor and refresh:
                raise HTTPException(status_code=400, detail="refresh cannot be combined with cursor")

            requested_refresh = bool(refresh and not cursor)
            lookup_value, lookup_type = resolve_customer_lookup_input(
                customer_id=customer_id,
                customer_name=customer_name,
                invoice_no=invoice_no,
                query=query,
            )
            resolved_lookup = resolve_customer_lookup_key(
                customer_service=customer_service,
                segment=segment,
                lookup_value=lookup_value,
                refresh=requested_refresh,
            )
            customer_key = str(resolved_lookup.get("customer_id") or "").strip()
            if not customer_key:
                raise HTTPException(status_code=404, detail=f"No customer matched lookup '{lookup_value}'.")

            page_size = min(int(limit), 200)
            offset = 0
            if cursor:
                cursor_payload = decode_cursor(cursor)
                if str(cursor_payload.get("segment") or "").strip().lower() != str(segment or "").strip().lower():
                    raise HTTPException(status_code=422, detail="Cursor segment does not match request segment.")
                if str(cursor_payload.get("customer_id") or "").strip() != customer_key:
                    raise HTTPException(status_code=422, detail="Cursor customer_id does not match request customer_id.")
                if int(cursor_payload.get("page_size") or 0) != page_size:
                    raise HTTPException(status_code=422, detail="Cursor page size does not match request limit.")
                offset = max(int(cursor_payload.get("offset") or 0), 0)

            requires_history_payload = bool(cursor or include_history or include_features or include_canonical)

            if not requires_history_payload:
                customer_summary = load_customer_summary_or_bootstrap(
                    api_cache=api_cache,
                    customer_service=customer_service,
                    segment=segment,
                    customer_id=customer_key,
                    refresh=requested_refresh,
                )
                response = build_customer_profile_payload(
                    segment=segment,
                    customer_id=customer_key,
                    customer_summary=customer_summary,
                    refresh_applied=requested_refresh,
                )
                response["lookup"] = {
                    "input_type": lookup_type,
                    "input_value": lookup_value,
                    "resolved_customer_id": customer_key,
                    "resolved_customer_name": resolved_lookup.get("customer_name"),
                    "matched_by": resolved_lookup.get("matched_by"),
                }
                logger.info(
                    "customer_profile_complete segment=%s customer_id=%s lookup_type=%s matched_by=%s pd=%.6f risk_band=%s approval=%s source=%s",
                    segment,
                    customer_key,
                    lookup_type,
                    resolved_lookup.get("matched_by"),
                    float(customer_summary.get("pd", 0.0) or 0.0),
                    customer_summary.get("risk_band", "unknown"),
                    customer_summary.get("approval", "unknown"),
                    response.get("source", {}).get("summary_source", "unknown"),
                )
                return response

            customer_summary_override = load_customer_summary_or_bootstrap(
                api_cache=api_cache,
                customer_service=customer_service,
                segment=segment,
                customer_id=customer_key,
                refresh=False,
            )

            response = build_customer_history_response(
                customer_service=customer_service,
                threshold_resolver=threshold_resolver,
                segment=segment,
                customer_key=customer_key,
                page_size=page_size,
                offset=offset,
                include_features=include_features,
                include_canonical=include_canonical,
                refresh=requested_refresh,
                customer_summary_override=customer_summary_override,
            )
            response["lookup"] = {
                "input_type": lookup_type,
                "input_value": lookup_value,
                "resolved_customer_id": customer_key,
                "resolved_customer_name": resolved_lookup.get("customer_name"),
                "matched_by": resolved_lookup.get("matched_by"),
            }
            customer_summary = response.get("customer_summary", {})
            feature_quality = response.get("feature_quality", {})
            logger.info(
                "customer_get_complete segment=%s customer_id=%s lookup_type=%s matched_by=%s returned=%s total=%s pd=%.6f risk_band=%s approval=%s validation_passed=%s features=%s canonical=%s",
                segment,
                customer_key,
                lookup_type,
                resolved_lookup.get("matched_by"),
                response.get("count"),
                response.get("total_available"),
                float(customer_summary.get("pd", 0.0) or 0.0),
                customer_summary.get("risk_band", "unknown"),
                customer_summary.get("approval", "unknown"),
                feature_quality.get("feature_validation_passed", "unknown"),
                include_features,
                include_canonical,
            )
            return response
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Customer validation failed: {exc}")
        except Exception:
            logger.exception(
                "Unified customer endpoint failed for segment=%s customer_id=%s customer_name=%s invoice_no=%s query=%s",
                segment,
                customer_id,
                customer_name,
                invoice_no,
                query,
            )
            raise HTTPException(status_code=500, detail="Customer retrieval failed. Check server logs.")

    @router.post("/score-customer/{segment}")
    def score_customer(
        segment: str,
        payload: CustomerScoreRequest,
        response: Response,
        _auth: None = Depends(require_api_key),
    ):
        logger.debug("Customer scoring request started segment=%s", segment)
        try:
            response.headers["Deprecation"] = "true"
            response.headers["Warning"] = '299 - "POST /api/risk/score-customer/{segment} is deprecated; use GET /api/risk/customer/{segment}"'
            customer_id = str(payload.customerId).strip()
            if not customer_id:
                raise HTTPException(status_code=400, detail="customer_id must be provided")
            descriptor = describe_active_production_model()
            preview_limit = history_preview_limit(payload)
            customer_result = customer_service.score_customer(
                segment=segment,
                customer_id=customer_id,
                force_refresh=bool(payload.refresh),
            )
            response_payload = _build_customer_summary_payload(
                customer_result.records,
                segment=segment,
                customer_id=customer_id,
                history_preview_limit=preview_limit,
                model_type=descriptor["model_type"],
                feature_quality=customer_result.feature_quality,
                include_history_preview=payload.includeHistoryPreview,
                include_invoice_top_features=payload.includeTopInvoiceFeatures,
                segment_invoice_rows=customer_result.segment_invoice_rows,
                approval_threshold_override=threshold_resolver(segment),
            )
            pd_trace = response_payload.get("customer_summary", {}).get("pd_computation_trace", {})
            logger.info(
                "customer_score_complete segment=%s customer_id=%s invoices=%s pd=%.6f risk_band=%s approval=%s pd_path=%s",
                segment,
                customer_id,
                response_payload.get("customer_summary", {}).get("invoice_rows_scored"),
                float(response_payload.get("customer_summary", {}).get("pd", 0.0) or 0.0),
                response_payload.get("customer_summary", {}).get("risk_band", "unknown"),
                response_payload.get("customer_summary", {}).get("approval", "unknown"),
                pd_trace.get("path", "unknown"),
            )
            return response_payload
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
        limit: int = settings.score_all_page_default,
        cursor: str | None = None,
        search: str | None = None,
        refresh: bool = False,
        _auth: None = Depends(require_api_key),
    ):
        logger.debug(
            "Customer list request started segment=%s limit=%s cursor=%s search=%s refresh=%s",
            segment,
            limit,
            bool(cursor),
            bool(str(search or "").strip()),
            refresh,
        )
        try:
            if limit < 1:
                raise HTTPException(status_code=400, detail="limit must be >= 1")
            if cursor and refresh:
                raise HTTPException(status_code=400, detail="refresh cannot be combined with cursor")

            customer_page_size = min(int(limit), settings.score_all_page_max)
            offset = 0
            if cursor:
                cursor_payload = decode_cursor(cursor)
                if str(cursor_payload.get("segment") or "").strip().lower() != str(segment or "").strip().lower():
                    raise HTTPException(status_code=422, detail="Cursor segment does not match request segment.")
                if int(cursor_payload.get("page_size") or 0) != customer_page_size:
                    raise HTTPException(status_code=422, detail="Cursor page size does not match request limit.")
                if str(cursor_payload.get("search") or "").strip() != str(search or "").strip():
                    raise HTTPException(status_code=422, detail="Cursor search does not match request search.")
                offset = max(int(cursor_payload.get("offset") or 0), 0)
            store_payload = load_customer_portfolio_page_from_store(
                api_cache=api_cache,
                segment=segment,
                search=search,
                refresh=bool(refresh and not cursor),
            )
            filtered_summary = store_payload["filtered_summary"]
            page_frame, total_available = api_cache.get_persisted_customer_portfolio_page(
                segment=segment,
                page_size=customer_page_size,
                offset=offset,
                search=search,
            )
            page_frame = clean_customer_portfolio_frame(page_frame)
            returned = int(len(page_frame))
            next_offset = min(offset + returned, total_available)
            next_cursor = None
            if next_offset < total_available:
                next_cursor = encode_cursor(
                    {
                        "segment": str(segment or "").strip().lower(),
                        "page_size": customer_page_size,
                        "search": str(search or "").strip(),
                        "offset": next_offset,
                    }
                )

            descriptor = describe_active_production_model()
            response_payload = {
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
                "snapshot_summary": filtered_summary,
                "total_available": total_available,
                "pagination": {
                    "snapshot_id": None,
                    "snapshot_generated_at": None,
                    "returned": returned,
                    "has_more": next_cursor is not None,
                    "next_cursor": next_cursor,
                },
                "records": _normalize_response_records(page_frame),
            }
            logger.info(
                "Customer list completed segment=%s customers=%s total_customers=%s source=customerriskmasters",
                segment,
                returned,
                total_available,
            )
            return response_payload
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Customer list validation failed: {exc}")
        except Exception:
            logger.exception("Customer list scoring failed for segment=%s", segment)
            raise HTTPException(status_code=500, detail="Customer list scoring failed. Check server logs.")

    @router.get("/customer-dashboard/{segment}")
    def customer_dashboard(
        segment: str,
        refresh: bool = False,
        _auth: None = Depends(require_api_key),
    ):
        logger.debug("Customer dashboard request started segment=%s refresh=%s", segment, refresh)
        try:
            store_payload = load_customer_portfolio_page_from_store(
                api_cache=api_cache,
                segment=segment,
                search=None,
                refresh=bool(refresh),
            )
            summary = dict(store_payload.get("filtered_summary") or {})
            descriptor = describe_active_production_model()
            response_payload = {
                "segment": str(segment or "").strip().lower(),
                "model_type": descriptor["model_type"],
                "model_family": descriptor["model_family"],
                "model_version": descriptor["version"],
                "total_available": int(store_payload.get("segment_customer_count") or summary.get("customers") or 0),
                "summary": summary,
                "source": {
                    "data_source": "customerriskmasters",
                    "refresh_applied": bool(refresh),
                    "latest_snapshot_id": summary.get("latest_snapshot_id"),
                    "latest_persisted_at_utc": summary.get("latest_persisted_at_utc"),
                },
            }
            logger.info(
                "Customer dashboard completed segment=%s customers=%s source=customerriskmasters",
                segment,
                response_payload["total_available"],
            )
            return response_payload
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Customer dashboard validation failed: {exc}")
        except Exception:
            logger.exception("Customer dashboard failed for segment=%s", segment)
            raise HTTPException(status_code=500, detail="Customer dashboard failed. Check server logs.")

    @router.get("/customer-history/{segment}")
    def customer_history(
        segment: str,
        customer_id: str | None = None,
        customer_name: str | None = None,
        invoice_no: str | None = None,
        query: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        include_features: bool = False,
        include_canonical: bool = False,
        refresh: bool = False,
        _auth: None = Depends(require_api_key),
    ):
        logger.debug(
            "Customer history request started segment=%s customer_id=%s customer_name=%s invoice_no=%s query=%s limit=%s cursor=%s refresh=%s",
            segment,
            customer_id,
            customer_name,
            invoice_no,
            query,
            limit,
            bool(cursor),
            refresh,
        )
        try:
            if limit < 1:
                raise HTTPException(status_code=400, detail="limit must be >= 1")
            if cursor and refresh:
                raise HTTPException(status_code=400, detail="refresh cannot be combined with cursor")

            requested_refresh = bool(refresh and not cursor)
            lookup_value, lookup_type = resolve_customer_lookup_input(
                customer_id=customer_id,
                customer_name=customer_name,
                invoice_no=invoice_no,
                query=query,
            )
            resolved_lookup = resolve_customer_lookup_key(
                customer_service=customer_service,
                segment=segment,
                lookup_value=lookup_value,
                refresh=requested_refresh,
            )
            customer_key = str(resolved_lookup.get("customer_id") or "").strip()
            if not customer_key:
                raise HTTPException(status_code=404, detail=f"No customer matched lookup '{lookup_value}'.")

            page_size = min(int(limit), 200)
            offset = 0
            if cursor:
                cursor_payload = decode_cursor(cursor)
                if str(cursor_payload.get("segment") or "").strip().lower() != str(segment or "").strip().lower():
                    raise HTTPException(status_code=422, detail="Cursor segment does not match request segment.")
                if str(cursor_payload.get("customer_id") or "").strip() != customer_key:
                    raise HTTPException(status_code=422, detail="Cursor customer_id does not match request customer_id.")
                if int(cursor_payload.get("page_size") or 0) != page_size:
                    raise HTTPException(status_code=422, detail="Cursor page size does not match request limit.")
                offset = max(int(cursor_payload.get("offset") or 0), 0)

            response_payload = build_customer_history_response(
                customer_service=customer_service,
                threshold_resolver=threshold_resolver,
                segment=segment,
                customer_key=customer_key,
                page_size=page_size,
                offset=offset,
                include_features=include_features,
                include_canonical=include_canonical,
                refresh=requested_refresh,
            )
            response_payload["lookup"] = {
                "input_type": lookup_type,
                "input_value": lookup_value,
                "resolved_customer_id": customer_key,
                "resolved_customer_name": resolved_lookup.get("customer_name"),
                "matched_by": resolved_lookup.get("matched_by"),
            }
            customer_summary = response_payload.get("customer_summary", {})
            feature_quality = response_payload.get("feature_quality", {})
            logger.info(
                "customer_history_complete segment=%s customer_id=%s lookup_type=%s matched_by=%s returned=%s total=%s pd=%.6f risk_band=%s approval=%s validation_passed=%s features=%s canonical=%s",
                segment,
                customer_key,
                lookup_type,
                resolved_lookup.get("matched_by"),
                response_payload.get("count"),
                response_payload.get("total_available"),
                float(customer_summary.get("pd", 0.0) or 0.0),
                customer_summary.get("risk_band", "unknown"),
                customer_summary.get("approval", "unknown"),
                feature_quality.get("feature_validation_passed", "unknown"),
                include_features,
                include_canonical,
            )
            return response_payload
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Customer history validation failed: {exc}")
        except Exception:
            logger.exception(
                "Customer history failed for segment=%s customer_id=%s customer_name=%s invoice_no=%s query=%s",
                segment,
                customer_id,
                customer_name,
                invoice_no,
                query,
            )
            raise HTTPException(status_code=500, detail="Customer history failed. Check server logs.")

    return {
        "customer": customer,
        "score_customer": score_customer,
        "score_customers": score_customers,
        "customer_dashboard": customer_dashboard,
        "customer_history": customer_history,
    }
