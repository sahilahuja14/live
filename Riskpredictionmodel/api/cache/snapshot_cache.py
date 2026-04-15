from __future__ import annotations

from threading import Condition, Lock
from time import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from ..pagination import decode_cursor, encode_cursor
from ..response_builder import (
    _build_scored_summary,
    _filter_customer_rows,
    _normalize_response_records,
    _shape_response_frame,
    _ts_to_iso,
    build_customer_portfolio_frame,
    response_from_raw,
)
from ...logging_config import get_logger
from ...pipeline.runner import score_mongo_frame
from .dataset_cache import SNAPSHOT_ID_SORT_COL, SNAPSHOT_INVOICE_TS_COL, SNAPSHOT_SEGMENT_COL


logger = get_logger(__name__)


def empty_scored_summary() -> dict:
    return {
        "rows": 0,
        "actual_delay_rate": None,
        "average_pd": None,
        "average_score": None,
        "approval_mix": {},
        "risk_band_mix": {},
        "top_pd_customers": [],
    }


class SnapshotCache:
    def __init__(
        self,
        *,
        lock: Lock,
        condition: Condition,
        dataset_ttl_seconds: int,
        scored_snapshot_retention_seconds: int,
        model_key_loader: Callable[[], str],
        load_full_dataset: Callable[[bool], pd.DataFrame],
        enrich_snapshot_with_customer_history: Callable[[pd.DataFrame, bool], pd.DataFrame],
        resolve_threshold: Callable[[str], float | None],
        broadcaster: Any = None,
        persist_customer_portfolio: Callable[[str, str, pd.DataFrame], None] | None = None,
    ) -> None:
        self._lock = lock
        self._condition = condition
        self._dataset_ttl_seconds = int(dataset_ttl_seconds)
        self._scored_snapshot_retention_seconds = int(scored_snapshot_retention_seconds)
        self._model_key_loader = model_key_loader
        self._load_full_dataset = load_full_dataset
        self._enrich_snapshot_with_customer_history = enrich_snapshot_with_customer_history
        self._resolve_threshold = resolve_threshold
        self._broadcaster = broadcaster
        self._persist_customer_portfolio = persist_customer_portfolio

        self._scored_snapshot: dict | None = None
        self._previous_scored_snapshot: dict | None = None
        self._scored_loading = False
        self._page_cache: dict[tuple[str, str, int, int], dict] = {}

    def _apply_segment_thresholds(self, merged: pd.DataFrame) -> pd.DataFrame:
        if merged.empty:
            return merged

        segment_series = (
            merged.get(SNAPSHOT_SEGMENT_COL, pd.Series("unknown", index=merged.index))
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown"})
        )
        default_threshold = pd.to_numeric(
            merged.get("approval_threshold", pd.Series(dtype=object)),
            errors="coerce",
        ).fillna(0.30)
        threshold_series = default_threshold.copy()
        policy_series = (
            merged.get("approval_threshold_policy", pd.Series("active", index=merged.index))
            .fillna("active")
            .astype(str)
        )

        for segment in sorted(segment_series.unique().tolist()):
            override_threshold = self._resolve_threshold(segment)
            if override_threshold is None:
                continue
            segment_mask = segment_series == segment
            threshold_series.loc[segment_mask] = float(override_threshold)
            policy_series.loc[segment_mask] = "override"

        prob_array = pd.to_numeric(merged.get("pd", pd.Series(dtype=object)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        threshold_array = threshold_series.to_numpy(dtype=float)
        merged["approval_threshold"] = threshold_series.astype(float)
        merged["approval_threshold_policy"] = policy_series
        merged["approval"] = np.where(prob_array <= threshold_array, "Approve", "Reject").astype(object)
        return merged

    def _build_scored_snapshot(self, full_df: pd.DataFrame, model_key: str) -> dict:
        prepared_full = full_df.copy()
        snapshot_ts = time()
        snapshot_id = f"snapshot_{uuid4().hex}"

        if prepared_full.empty:
            return {
                "snapshot_id": snapshot_id,
                "ts": snapshot_ts,
                "generated_at": _ts_to_iso(snapshot_ts),
                "model_key": model_key,
                "records": [],
                "segment_positions": {"all": []},
                "segment_summaries": {"all": empty_scored_summary()},
                "segment_counts": {"all": 0},
                "rows": 0,
                "raw_rows": 0,
            }

        # Score the full snapshot once, then apply per-segment approval thresholds.
        enriched_full = self._enrich_snapshot_with_customer_history(prepared_full, force_refresh=True)
        scored = score_mongo_frame(
            enriched_full,
            history_df=enriched_full,
            top_n=5,
            approval_threshold_override=None,
            scoring_context="bulk_snapshot:all",
        )
        merged = response_from_raw(enriched_full, scored)
        merged = self._apply_segment_thresholds(merged)

        merged = merged.sort_values(
            [SNAPSHOT_INVOICE_TS_COL, SNAPSHOT_ID_SORT_COL],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        ordered_segments = (
            merged.get(SNAPSHOT_SEGMENT_COL, pd.Series("unknown", index=merged.index))
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown"})
        )
        shaped = _shape_response_frame(merged, response_mode="lean")
        records = _normalize_response_records(shaped)

        segment_positions: dict[str, list[int]] = {"all": list(range(len(records)))}
        segment_counts: dict[str, int] = {"all": int(len(records))}
        segment_summaries: dict[str, dict] = {"all": _build_scored_summary(shaped)}

        for segment in sorted(ordered_segments.unique().tolist()):
            positions = ordered_segments[ordered_segments == segment].index.tolist()
            segment_positions[segment] = positions
            segment_counts[segment] = len(positions)
            if positions:
                segment_summaries[segment] = _build_scored_summary(shaped.iloc[positions].copy())
            else:
                segment_summaries[segment] = empty_scored_summary()

        self._persist_customer_portfolios(
            snapshot_id=snapshot_id,
            merged=merged,
            ordered_segments=ordered_segments,
        )

        return {
            "snapshot_id": snapshot_id,
            "ts": snapshot_ts,
            "generated_at": _ts_to_iso(snapshot_ts),
            "model_key": model_key,
            "records": records,
            "segment_positions": segment_positions,
            "segment_summaries": segment_summaries,
            "segment_counts": segment_counts,
            "rows": int(len(records)),
            "raw_rows": int(len(prepared_full)),
        }

    def _persist_customer_portfolios(
        self,
        *,
        snapshot_id: str,
        merged: pd.DataFrame,
        ordered_segments: pd.Series,
    ) -> None:
        if self._persist_customer_portfolio is None or merged.empty:
            return
        if "customer.customerId" not in merged.columns:
            return

        all_frame = merged.copy()
        all_customer_ids = all_frame["customer.customerId"].fillna("").astype(str).str.strip()
        all_frame = all_frame[all_customer_ids != ""].copy()
        if all_frame.empty:
            return

        try:
            all_portfolio = build_customer_portfolio_frame(
                all_frame,
                segment="all",
                approval_threshold_override=self._resolve_threshold("all"),
            )
            self._persist_customer_portfolio(
                segment="all",
                snapshot_id=snapshot_id,
                portfolio_frame=all_portfolio,
            )

            for segment in sorted(ordered_segments.dropna().astype(str).str.strip().str.lower().unique().tolist()):
                if not segment or segment == "all":
                    continue

                segment_frame = merged.loc[ordered_segments == segment].copy()
                if segment_frame.empty or "customer.customerId" not in segment_frame.columns:
                    continue

                segment_ids = segment_frame["customer.customerId"].fillna("").astype(str).str.strip()
                segment_frame = segment_frame[segment_ids != ""].copy()
                if segment_frame.empty:
                    continue

                segment_invoice_counts = segment_frame.groupby("customer.customerId").size().astype(int).to_dict()
                source_frame = all_frame[
                    all_frame["customer.customerId"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .isin(segment_invoice_counts.keys())
                ].copy()
                segment_portfolio = build_customer_portfolio_frame(
                    source_frame,
                    segment=segment,
                    segment_invoice_counts=segment_invoice_counts,
                    approval_threshold_override=self._resolve_threshold(segment),
                )
                self._persist_customer_portfolio(
                    segment=segment,
                    snapshot_id=snapshot_id,
                    portfolio_frame=segment_portfolio,
                )
        except Exception:
            logger.warning(
                "Customer portfolio persistence skipped because snapshot export failed snapshot_id=%s",
                snapshot_id,
                exc_info=True,
            )

    def _prune_previous_snapshot_locked(self, now: float) -> None:
        if self._previous_scored_snapshot is None:
            return
        age = now - float(self._previous_scored_snapshot.get("ts", 0.0) or 0.0)
        if age > self._scored_snapshot_retention_seconds:
            self._previous_scored_snapshot = None

    def clear_runtime_caches(self) -> None:
        with self._lock:
            self._page_cache.clear()

    def load_scored_snapshot(self, force_refresh: bool = False, source_df: pd.DataFrame | None = None) -> dict:
        model_key = self._model_key_loader()
        with self._condition:
            while self._scored_loading:
                self._condition.wait()

            now = time()
            self._prune_previous_snapshot_locked(now)
            current = self._scored_snapshot
            if current is not None:
                age = now - float(current.get("ts", 0.0) or 0.0)
                if (
                    not force_refresh
                    and str(current.get("model_key")) == model_key
                    and age <= self._dataset_ttl_seconds
                ):
                    return current

            stale_snapshot = current
            self._scored_loading = True

        try:
            full_df = source_df.copy() if source_df is not None else self._load_full_dataset(force_refresh=force_refresh)
            snapshot = self._build_scored_snapshot(full_df, model_key=model_key)
        except Exception:
            with self._condition:
                self._scored_loading = False
                self._condition.notify_all()
            if not force_refresh and stale_snapshot is not None:
                logger.warning(
                    "Scored snapshot rebuild failed; serving stale snapshot snapshot_id=%s",
                    stale_snapshot.get("snapshot_id"),
                    exc_info=True,
                )
                return stale_snapshot
            raise

        with self._condition:
            previous = self._scored_snapshot
            if previous is not None:
                self._previous_scored_snapshot = previous
            self._scored_snapshot = snapshot
            self._scored_loading = False
            self._page_cache.clear()
            self._condition.notify_all()

        if self._broadcaster is not None:
            self._broadcaster.notify_snapshot_ready_threadsafe(snapshot)

        return snapshot

    def _resolve_scored_snapshot(self, snapshot_id: str | None = None) -> dict | None:
        now = time()
        with self._lock:
            self._prune_previous_snapshot_locked(now)
            current = self._scored_snapshot
            previous = self._previous_scored_snapshot

        if snapshot_id is None:
            return current
        if current is not None and str(current.get("snapshot_id")) == str(snapshot_id):
            return current
        if previous is not None and str(previous.get("snapshot_id")) == str(snapshot_id):
            return previous
        return None

    def _positions_for_segment(self, snapshot: dict, segment: str) -> list[int]:
        normalized_segment = str(segment or "").strip().lower()
        if normalized_segment == "all":
            return list(snapshot.get("segment_positions", {}).get("all", []))
        return list(snapshot.get("segment_positions", {}).get(normalized_segment, []))

    def get_scored_page(
        self,
        *,
        segment: str,
        page_size: int,
        cursor: str | None = None,
        refresh: bool = False,
    ) -> dict:
        segment_value = str(segment or "").strip().lower()
        if cursor and refresh:
            raise ValueError("refresh cannot be combined with cursor.")

        if cursor:
            payload = decode_cursor(cursor)
            cursor_segment = str(payload.get("segment") or "").strip().lower()
            cursor_page_size = int(payload.get("page_size") or 0)
            if cursor_segment != segment_value:
                raise ValueError("Cursor segment does not match request segment.")
            if cursor_page_size != int(page_size):
                raise ValueError("Cursor page size does not match request limit.")
            offset = max(int(payload.get("offset") or 0), 0)
            snapshot_id = str(payload.get("snapshot_id") or "").strip()
            self.load_scored_snapshot(force_refresh=False)
            snapshot = self._resolve_scored_snapshot(snapshot_id=snapshot_id)
            if snapshot is None:
                raise ValueError("Snapshot is stale or unavailable. Restart pagination without a cursor.")
        else:
            snapshot = self.load_scored_snapshot(force_refresh=refresh)
            snapshot_id = str(snapshot.get("snapshot_id") or "")
            offset = 0

        page_key = (snapshot_id, segment_value, int(page_size), int(offset))
        with self._lock:
            cached = self._page_cache.get(page_key)
            if cached is not None:
                return dict(cached)

        positions = self._positions_for_segment(snapshot, segment_value)
        total_available = len(positions)
        page_positions = positions[offset : offset + int(page_size)]
        all_records = list(snapshot.get("records", []))
        page_records = [all_records[pos] for pos in page_positions]
        page_df = pd.DataFrame(page_records) if page_records else pd.DataFrame()
        returned = int(len(page_records))
        next_offset = offset + returned
        has_more = next_offset < total_available
        next_cursor = None

        if has_more:
            next_cursor = encode_cursor(
                {
                    "snapshot_id": snapshot_id,
                    "segment": segment_value,
                    "page_size": int(page_size),
                    "offset": int(next_offset),
                }
            )

        payload = {
            "count": returned,
            "limit": int(page_size),
            "summary": _build_scored_summary(page_df),
            "snapshot_summary": dict(
                snapshot.get("segment_summaries", {}).get(segment_value, empty_scored_summary())
            ),
            "total_available": total_available,
            "records": page_records,
            "pagination": {
                "snapshot_id": snapshot_id,
                "snapshot_generated_at": snapshot.get("generated_at"),
                "returned": returned,
                "has_more": has_more,
                "next_cursor": next_cursor,
            },
        }

        with self._lock:
            self._page_cache[page_key] = dict(payload)

        return payload

    def get_snapshot_customer_frame(
        self,
        *,
        snapshot_id: str,
        segment: str,
        customer_id: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        self.load_scored_snapshot(force_refresh=False)
        snapshot = self._resolve_scored_snapshot(snapshot_id=snapshot_id)
        if snapshot is None:
            raise ValueError("Snapshot is stale or unavailable.")

        positions = self._positions_for_segment(snapshot, segment)
        all_records = list(snapshot.get("records", []))
        scoped_records = [all_records[pos] for pos in positions]
        scoped_df = pd.DataFrame(scoped_records) if scoped_records else pd.DataFrame()
        customer_df = _filter_customer_rows(scoped_df, customer_id)
        if limit is not None and not customer_df.empty:
            customer_df = customer_df.iloc[: int(limit)].copy()
        return customer_df.reset_index(drop=True)

    def get_scored_segment_frame(self, *, segment: str, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
        snapshot = self.load_scored_snapshot(force_refresh=refresh)
        segment_value = str(segment or "").strip().lower()
        positions = self._positions_for_segment(snapshot, segment_value)
        all_records = list(snapshot.get("records", []))
        segment_records = [all_records[pos] for pos in positions]
        frame = pd.DataFrame(segment_records) if segment_records else pd.DataFrame()
        return frame, {
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_generated_at": snapshot.get("generated_at"),
            "rows": int(len(segment_records)),
            "total_available": int(len(positions)),
            "summary": dict(
                snapshot.get("segment_summaries", {}).get(segment_value, empty_scored_summary())
            ),
        }

    def snapshot_state(self, *, now: float) -> dict:
        with self._lock:
            self._prune_previous_snapshot_locked(now)
            current_snapshot = self._scored_snapshot
            previous_snapshot = self._previous_scored_snapshot
            page_cache_entries = len(self._page_cache)

        scored_snapshot_age = None
        previous_snapshot_age = None
        if current_snapshot is not None:
            scored_snapshot_age = now - float(current_snapshot.get("ts", 0.0) or 0.0)
        if previous_snapshot is not None:
            previous_snapshot_age = now - float(previous_snapshot.get("ts", 0.0) or 0.0)

        return {
            "snapshot_id": None if current_snapshot is None else current_snapshot.get("snapshot_id"),
            "generated_at": None if current_snapshot is None else current_snapshot.get("generated_at"),
            "age_seconds": scored_snapshot_age,
            "rows": 0 if current_snapshot is None else int(current_snapshot.get("rows", 0) or 0),
            "raw_rows": 0 if current_snapshot is None else int(current_snapshot.get("raw_rows", 0) or 0),
            "segment_counts": {} if current_snapshot is None else dict(current_snapshot.get("segment_counts", {})),
            "previous_snapshot_id": None if previous_snapshot is None else previous_snapshot.get("snapshot_id"),
            "previous_snapshot_age_seconds": previous_snapshot_age,
            "page_cache_entries": page_cache_entries,
        }
