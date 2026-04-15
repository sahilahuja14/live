from __future__ import annotations

from fastapi import HTTPException

from ...scoring.model import describe_active_production_model, load_production_artifacts
from ..pagination import encode_cursor
from ..response_builder import (
    _build_customer_summary_payload,
    build_customer_history_payload,
)
from .scoring_helpers import canonical_snapshot_for_rows, feature_snapshot_for_rows


def load_customer_portfolio_page_from_store(
    *,
    api_cache,
    segment: str,
    search: str | None,
    refresh: bool,
) -> dict:
    normalized_segment = str(segment or "").strip().lower()
    normalized_search = str(search or "").strip() or None

    if refresh:
        api_cache.refresh(trigger=f"customer_list_refresh:{normalized_segment}")

    total_segment_customers = api_cache.get_persisted_customer_portfolio_count(
        segment=normalized_segment,
        search=None,
    )
    if total_segment_customers > 0:
        return {
            "segment_customer_count": total_segment_customers,
            "filtered_summary": api_cache.get_persisted_customer_portfolio_summary(
                segment=normalized_segment,
                search=normalized_search,
            ),
        }

    api_cache.refresh(trigger=f"customer_collection_bootstrap:{normalized_segment}")
    warmed_segment_customers = api_cache.get_persisted_customer_portfolio_count(
        segment=normalized_segment,
        search=None,
    )
    if warmed_segment_customers > 0:
        return {
            "segment_customer_count": warmed_segment_customers,
            "filtered_summary": api_cache.get_persisted_customer_portfolio_summary(
                segment=normalized_segment,
                search=normalized_search,
            ),
        }

    raise HTTPException(
        status_code=503,
        detail=(
            "Customer risk data is warming up. A refresh was triggered for "
            f"segment '{normalized_segment}'. Retry shortly."
        ),
    )


def load_customer_summary_from_store(
    *,
    api_cache,
    segment: str,
    customer_id: str,
    refresh: bool,
) -> dict | None:
    normalized_segment = str(segment or "").strip().lower()
    normalized_customer_id = str(customer_id or "").strip()
    if not normalized_customer_id:
        return None

    if refresh:
        api_cache.refresh(trigger=f"customer_summary_refresh:{normalized_segment}:{normalized_customer_id}")

    return api_cache.get_persisted_customer_record(
        segment=normalized_segment,
        customer_id=normalized_customer_id,
    )


def load_customer_summary_or_bootstrap(
    *,
    api_cache,
    customer_service,
    segment: str,
    customer_id: str,
    refresh: bool,
) -> dict:
    normalized_segment = str(segment or "").strip().lower()
    normalized_customer_id = str(customer_id or "").strip()

    customer_summary = load_customer_summary_from_store(
        api_cache=api_cache,
        segment=normalized_segment,
        customer_id=normalized_customer_id,
        refresh=refresh,
    )
    if customer_summary:
        return customer_summary

    api_cache.refresh(trigger=f"customer_summary_bootstrap:{normalized_segment}:{normalized_customer_id}")
    warmed_summary = load_customer_summary_from_store(
        api_cache=api_cache,
        segment=normalized_segment,
        customer_id=normalized_customer_id,
        refresh=False,
    )
    if warmed_summary:
        return warmed_summary

    customer_service._load_customer_invoice_frame(
        segment=normalized_segment,
        customer_id=normalized_customer_id,
        force_refresh=False,
    )
    raise HTTPException(
        status_code=503,
        detail=(
            "Customer risk profile is warming up. A refresh was triggered for "
            f"customer '{normalized_customer_id}' in segment '{normalized_segment}'. Retry shortly."
        ),
    )


def resolve_customer_lookup_input(
    *,
    customer_id: str | None = None,
    customer_name: str | None = None,
    invoice_no: str | None = None,
    query: str | None = None,
) -> tuple[str, str]:
    for lookup_type, raw_value in (
        ("customer_id", customer_id),
        ("customer_name", customer_name),
        ("invoice_no", invoice_no),
        ("query", query),
    ):
        normalized_value = str(raw_value or "").strip()
        if normalized_value:
            return normalized_value, lookup_type
    raise HTTPException(status_code=400, detail="Provide customer_id, customer_name, invoice_no, or query.")


def resolve_customer_lookup_key(
    *,
    customer_service,
    segment: str,
    lookup_value: str,
    refresh: bool,
) -> dict:
    return customer_service.resolve_customer_lookup(
        segment=segment,
        lookup_value=lookup_value,
        force_refresh=refresh,
    )


def build_customer_history_response(
    *,
    customer_service,
    threshold_resolver,
    segment: str,
    customer_key: str,
    page_size: int,
    offset: int,
    include_features: bool,
    include_canonical: bool,
    refresh: bool,
    customer_summary_override: dict | None = None,
) -> dict:
    history_page = customer_service.get_history_page(
        segment=segment,
        customer_id=customer_key,
        page_size=page_size,
        offset=offset,
        force_refresh=refresh,
    )
    customer_result = history_page["result"]
    page_records = history_page["page_records"]
    total_available = history_page["total_available"]
    returned = history_page["returned"]
    next_offset = history_page["next_offset"]

    next_cursor = None
    if next_offset < total_available:
        next_cursor = encode_cursor(
            {
                "segment": str(segment or "").strip().lower(),
                "customer_id": customer_key,
                "page_size": page_size,
                "offset": next_offset,
            }
        )

    descriptor = describe_active_production_model()
    customer_payload = _build_customer_summary_payload(
        customer_result.records,
        segment=segment,
        customer_id=customer_key,
        history_preview_limit=page_size,
        model_type=descriptor["model_type"],
        feature_quality=customer_result.feature_quality,
        include_history_preview=False,
        segment_invoice_rows=customer_result.segment_invoice_rows,
        approval_threshold_override=threshold_resolver(segment),
    )
    customer_summary = (
        dict(customer_summary_override)
        if customer_summary_override
        else dict(customer_payload["customer_summary"])
    )

    artifacts = load_production_artifacts()
    page_indices = list(range(offset, offset + returned))
    feature_snapshot = (
        feature_snapshot_for_rows(customer_result.scoring_frame, page_indices, artifacts["features"])
        if include_features
        else []
    )
    canonical_snapshot = (
        canonical_snapshot_for_rows(customer_result.customer_df, page_indices)
        if include_canonical
        else []
    )

    return build_customer_history_payload(
        segment=segment,
        customer_id=customer_key,
        customer_summary=customer_summary,
        records=page_records,
        total_available=total_available,
        returned=returned,
        next_cursor=next_cursor,
        snapshot_meta={
            "source_mode": "customerriskmasters+live_customer_history" if customer_summary_override else "live_customer_history",
            "segment": str(segment or "").strip().lower(),
            "model_type": descriptor["model_type"],
            "model_family": descriptor["model_family"],
            "model_version": descriptor["version"],
            "refresh_applied": bool(refresh),
            "feature_quality": customer_result.feature_quality,
        },
        feature_snapshot=feature_snapshot,
        canonical_snapshot=canonical_snapshot,
    )
