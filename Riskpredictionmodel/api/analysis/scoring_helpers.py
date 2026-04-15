from __future__ import annotations

import pandas as pd

from ...data.segment_filters import filter_segment as _segment_filter
from ...pipeline.risk_canonical import canonicalize_risk_main_frame
from ...pipeline.runner import score_mongo_frame
from ..response_builder import (
    _normalize_response_records,
    _shape_response_frame,
    response_from_raw as _response_from_raw,
)


def prepare_history_frame(api_cache, force_refresh: bool = False) -> pd.DataFrame:
    return api_cache.load_full_dataset(force_refresh=force_refresh)


def enrich_with_customer_history(api_cache, df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    return api_cache.enrich_with_customer_history(df, force_refresh=force_refresh)


def history_preview_limit(payload) -> int:
    if payload.limit is not None:
        return min(int(payload.limit), 100)
    return int(payload.historyPreviewLimit)


def feature_snapshot_for_rows(scoring_frame: pd.DataFrame, page_indices: list[int], expected_features: list[str]) -> list[dict]:
    if scoring_frame.empty or not page_indices:
        return []
    columns = [column for column in ["invoice_key", *expected_features] if column in scoring_frame.columns]
    if not columns:
        return []
    snapshot = scoring_frame.iloc[page_indices][columns].copy().reset_index(drop=True)
    return _normalize_response_records(snapshot)


def canonical_snapshot_for_rows(raw_frame: pd.DataFrame, page_indices: list[int]) -> list[dict]:
    if raw_frame.empty or not page_indices:
        return []
    canonical = canonicalize_risk_main_frame(raw_frame.copy())
    if canonical.empty:
        return []
    canonical = canonical.iloc[page_indices].copy().reset_index(drop=True)
    return _normalize_response_records(canonical)


def clean_customer_portfolio_frame(customer_frame: pd.DataFrame) -> pd.DataFrame:
    if customer_frame.empty:
        return customer_frame.copy()
    return customer_frame.drop(
        columns=[
            "portfolio_key",
            "store_version",
            "snapshot_id",
            "persisted_at_utc",
            "customer.customerId",
            "customer.customerName",
            "shipmentDetails.queryFor",
            "grossAmount",
            "max_pd",
            "customer_total_invoices",
            "customer_avg_delay_days",
            "customer_avg_invoice",
            "customer_delay_rate",
            "historical_customer_delay_rate",
        ],
        errors="ignore",
    ).copy()


def build_scored_frame(
    *,
    api_cache,
    threshold_resolver,
    segment: str,
    limit: int | None = None,
    customer_id: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    full_df = prepare_history_frame(api_cache, force_refresh=force_refresh)
    if full_df.empty:
        return pd.DataFrame()

    segment_df = _segment_filter(full_df, segment, allow_all=True, missing="input")
    if customer_id:
        customer_key = str(customer_id).strip()
        if customer_key:
            customer_series = segment_df.get("customer.customerId")
            if customer_series is not None:
                segment_df = segment_df.loc[customer_series.astype(str).eq(customer_key)].copy()
            elif "customerId" in segment_df.columns:
                segment_df = segment_df.loc[segment_df["customerId"].astype(str).eq(customer_key)].copy()
    if limit is not None and not segment_df.empty:
        segment_df = segment_df.iloc[:limit].copy()
    if segment_df.empty:
        return pd.DataFrame()

    segment_df = enrich_with_customer_history(api_cache, segment_df, force_refresh=force_refresh)
    scored = score_mongo_frame(
        segment_df,
        history_df=full_df,
        top_n=5,
        approval_threshold_override=threshold_resolver(segment),
        scoring_context=f"live_mongo:{segment.lower()}",
    )
    return _response_from_raw(segment_df, scored)


def build_scored_dataset(
    *,
    api_cache,
    threshold_resolver,
    segment: str,
    limit: int | None = None,
    customer_id: str | None = None,
    force_refresh: bool = False,
):
    frame = build_scored_frame(
        api_cache=api_cache,
        threshold_resolver=threshold_resolver,
        segment=segment,
        limit=limit,
        customer_id=customer_id,
        force_refresh=force_refresh,
    )
    if frame.empty:
        return []
    return _normalize_response_records(_shape_response_frame(frame, response_mode="lean"))
