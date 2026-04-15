from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from fastapi import HTTPException

from ...data.segment_filters import filter_segment as segment_filter
from ...features.registry import SEGMENT_COL
from ...pipeline.runner import score_mongo_frame_with_details
from ..cache import ApiCache
from ..pagination import slice_page
from ..response_builder import _filter_customer_rows, _normalize_response_records, _shape_response_frame, response_from_raw


@dataclass
class CustomerScoringResult:
    history_df: pd.DataFrame
    customer_df: pd.DataFrame
    segment_invoice_rows: int
    records: list[dict]
    scoring_frame: pd.DataFrame
    feature_quality: dict


class CustomerService:
    def __init__(
        self,
        *,
        api_cache: ApiCache,
        threshold_resolver: Callable[[str], float | None],
    ) -> None:
        self._api_cache = api_cache
        self._threshold_resolver = threshold_resolver

    def _feature_quality_payload(self, details, input_rows: int, *, scoring_context: str | None = None) -> dict:
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

    def _load_customer_invoice_frame(
        self,
        *,
        segment: str,
        customer_id: str,
        force_refresh: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        full_df = self._api_cache.load_full_dataset(force_refresh=force_refresh)
        customer_df = _filter_customer_rows(full_df, customer_id)
        if customer_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{customer_id}' was not found in any segment.",
            )

        segment_customer_df = segment_filter(customer_df, segment, allow_all=True, missing="input")
        if segment_customer_df.empty:
            available_segments = customer_df.get("shipmentDetails.queryFor", pd.Series(dtype=object))
            if available_segments.dropna().empty:
                available_segments = customer_df.get(SEGMENT_COL, pd.Series(dtype=object))
            available_segments = (
                available_segments.fillna("")
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

        enriched_customer_df = self._api_cache.enrich_with_customer_history(customer_df, force_refresh=force_refresh)
        return full_df, enriched_customer_df.reset_index(drop=True), int(len(segment_customer_df))

    def resolve_customer_lookup(
        self,
        *,
        segment: str,
        lookup_value: str,
        force_refresh: bool = False,
    ) -> dict:
        normalized_lookup = str(lookup_value or "").strip()
        if not normalized_lookup:
            raise HTTPException(status_code=400, detail="A customer lookup value must be provided.")

        persisted_match = self._api_cache.resolve_persisted_customer_lookup(
            segment=segment,
            lookup_value=normalized_lookup,
        )
        if persisted_match:
            return {
                "customer_id": str(persisted_match.get("customerId") or "").strip(),
                "customer_name": str(persisted_match.get("customerName") or "").strip() or None,
                "matched_by": "customer_store",
                "lookup_value": normalized_lookup,
            }

        full_df = self._api_cache.load_full_dataset(force_refresh=force_refresh)
        if full_df.empty:
            raise HTTPException(status_code=404, detail=f"No customer matched lookup '{normalized_lookup}'.")

        resolved = self._resolve_lookup_from_frame(full_df, normalized_lookup)
        if resolved is None:
            raise HTTPException(status_code=404, detail=f"No customer matched lookup '{normalized_lookup}'.")
        return resolved

    def _resolve_lookup_from_frame(self, frame: pd.DataFrame, lookup_value: str) -> dict | None:
        normalized_lookup = str(lookup_value or "").strip()
        if not normalized_lookup or frame.empty:
            return None

        working = frame.copy()
        customer_id_series = working.get("customer.customerId", pd.Series("", index=working.index)).fillna("").astype(str).str.strip()
        customer_name_series = working.get("customer.customerName", pd.Series("", index=working.index)).fillna("").astype(str).str.strip()
        invoice_key_series = working.get("invoice_key", pd.Series("", index=working.index)).fillna("").astype(str).str.strip()
        invoice_no_series = working.get("invoiceNo", pd.Series("", index=working.index)).fillna("").astype(str).str.strip()
        invoice_ref_series = working.get("legacy.invoice_ref_raw", pd.Series("", index=working.index)).fillna("").astype(str).str.strip()

        lookup_lower = normalized_lookup.lower()

        exact_customer_mask = customer_id_series.str.lower() == lookup_lower
        if exact_customer_mask.any():
            row = working.loc[exact_customer_mask].iloc[0]
            return {
                "customer_id": str(row.get("customer.customerId") or "").strip(),
                "customer_name": str(row.get("customer.customerName") or "").strip() or None,
                "matched_by": "customer_id",
                "lookup_value": normalized_lookup,
            }

        exact_name_mask = customer_name_series.str.lower() == lookup_lower
        if exact_name_mask.any():
            row = working.loc[exact_name_mask].iloc[0]
            return {
                "customer_id": str(row.get("customer.customerId") or "").strip(),
                "customer_name": str(row.get("customer.customerName") or "").strip() or None,
                "matched_by": "customer_name",
                "lookup_value": normalized_lookup,
            }

        invoice_mask = (
            (invoice_key_series.str.lower() == lookup_lower)
            | (invoice_no_series.str.lower() == lookup_lower)
            | (invoice_ref_series.str.lower() == lookup_lower)
        )
        if invoice_mask.any():
            row = working.loc[invoice_mask].iloc[0]
            return {
                "customer_id": str(row.get("customer.customerId") or "").strip(),
                "customer_name": str(row.get("customer.customerName") or "").strip() or None,
                "matched_by": "invoice",
                "lookup_value": normalized_lookup,
            }

        contains_name_mask = customer_name_series.str.lower().str.contains(lookup_lower, regex=False)
        if contains_name_mask.any():
            row = working.loc[contains_name_mask].iloc[0]
            return {
                "customer_id": str(row.get("customer.customerId") or "").strip(),
                "customer_name": str(row.get("customer.customerName") or "").strip() or None,
                "matched_by": "customer_name_contains",
                "lookup_value": normalized_lookup,
            }
        return None

    def score_customer(
        self,
        *,
        segment: str,
        customer_id: str,
        force_refresh: bool = False,
    ) -> CustomerScoringResult:
        scoring_context = f"live_customer:{segment.lower()}:{customer_id}"
        history_df, customer_df, segment_invoice_rows = self._load_customer_invoice_frame(
            segment=segment,
            customer_id=customer_id,
            force_refresh=force_refresh,
        )
        details = score_mongo_frame_with_details(
            customer_df,
            history_df=history_df,
            top_n=5,
            approval_threshold_override=self._threshold_resolver(segment),
            scoring_context=scoring_context,
        )
        merged = response_from_raw(customer_df, details.scored_frame)
        shaped = _shape_response_frame(merged, response_mode="lean")
        records = _normalize_response_records(shaped)
        if not records:
            raise HTTPException(
                status_code=500,
                detail="Customer scoring failed because no invoice rows passed feature validation.",
            )

        return CustomerScoringResult(
            history_df=history_df,
            customer_df=customer_df,
            segment_invoice_rows=int(segment_invoice_rows),
            records=records,
            scoring_frame=details.scoring_frame.reset_index(drop=True),
            feature_quality=self._feature_quality_payload(
                details,
                input_rows=len(customer_df),
                scoring_context=scoring_context,
            ),
        )

    def get_history_page(
        self,
        *,
        segment: str,
        customer_id: str,
        page_size: int,
        offset: int,
        force_refresh: bool = False,
    ) -> dict:
        result = self.score_customer(
            segment=segment,
            customer_id=customer_id,
            force_refresh=force_refresh,
        )
        page_records, total_available, returned, next_offset = slice_page(
            result.records,
            page_size=page_size,
            offset=offset,
        )
        return {
            "result": result,
            "page_records": page_records,
            "total_available": total_available,
            "returned": returned,
            "next_offset": next_offset,
        }
