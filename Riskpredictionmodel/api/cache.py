from __future__ import annotations

import base64
import json
import os
from threading import Condition, Event, Lock, Thread
from time import perf_counter, time
from typing import Callable
from uuid import uuid4

import pandas as pd

from .response_builder import (
    _build_scored_summary,
    _filter_customer_rows,
    _normalize_response_records,
    _shape_response_frame,
    _ts_to_iso,
    response_from_raw,
)
from ..features.customer_aggregates import customer_ids_from_frame, merge_customer_history_aggregates
from ..features.registry import CUSTOMER_ID_COL, SEGMENT_COL
from ..logging_config import get_logger
from ..pipeline.parsing import parse_main_date
from ..pipeline.risk_canonical import inspect_risk_main_indexes
from ..pipeline.risk_main import build_risk_main_customer_aggregates, fetch_production_risk_main_dataset
from ..pipeline.runner import score_mongo_frame
from ..scoring.model import describe_active_production_model


logger = get_logger(__name__)
_SNAPSHOT_SEGMENT_COL = "__snapshot_segment"
_SNAPSHOT_INVOICE_TS_COL = "__snapshot_invoice_ts"
_SNAPSHOT_ID_SORT_COL = "__snapshot_id_sort"



def _empty_summary() -> dict:
    return {
        "rows": 0,
        "actual_delay_rate": None,
        "average_pd": None,
        "average_score": None,
        "approval_mix": {},
        "risk_band_mix": {},
    }


class ApiCache:
    def __init__(
        self,
        *,
        dataset_ttl_seconds: int,
        history_ttl_seconds: int,
        auto_refresh_enabled: bool,
        auto_refresh_interval_seconds: int,
        scored_snapshot_retention_seconds: int,
        threshold_resolver: Callable[[str], float | None] | None = None,
    ) -> None:
        self.dataset_ttl_seconds = dataset_ttl_seconds
        self.history_ttl_seconds = history_ttl_seconds
        self.auto_refresh_enabled = auto_refresh_enabled
        self.auto_refresh_interval_seconds = auto_refresh_interval_seconds
        self.scored_snapshot_retention_seconds = scored_snapshot_retention_seconds
        self.threshold_resolver = threshold_resolver
        self.max_consecutive_refresh_errors = max(int(os.getenv("MAX_REFRESH_ERRORS", "5")), 1)

        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._dataset_cache = {"ts": 0.0, "df": pd.DataFrame(), "model_key": None}
        self._dataset_loading = False
        self._history_cache: dict[tuple[str, ...], tuple[float, pd.DataFrame]] = {}
        self._scored_snapshot: dict | None = None
        self._previous_scored_snapshot: dict | None = None
        self._scored_loading = False
        self._page_cache: dict[tuple[str, str, int, int], dict] = {}
        self._index_report: dict | None = None
        self._auto_refresh_state = {
            "running": False,
            "last_run_ts": 0.0,
            "last_error": None,
            "consecutive_errors": 0,
        }
        self._stop_event = Event()
        self._bg_thread: Thread | None = None

    def _model_key(self) -> str:
        descriptor = describe_active_production_model()
        return f"{descriptor['model_family']}::{descriptor['version']}"

    def _prepare_snapshot_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        prepared = df.copy()
        if SEGMENT_COL in prepared.columns:
            segment_values = prepared[SEGMENT_COL]
        elif "shipmentDetails.queryFor" in prepared.columns:
            segment_values = prepared["shipmentDetails.queryFor"]
        else:
            segment_values = pd.Series("unknown", index=prepared.index, dtype=object)
        prepared[_SNAPSHOT_SEGMENT_COL] = (
            segment_values.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown"})
        )

        if "invoiceDate" in prepared.columns:
            prepared[_SNAPSHOT_INVOICE_TS_COL] = prepared["invoiceDate"].apply(parse_main_date)
        else:
            prepared[_SNAPSHOT_INVOICE_TS_COL] = pd.Series(pd.NaT, index=prepared.index)

        if "_id" in prepared.columns:
            prepared[_SNAPSHOT_ID_SORT_COL] = prepared["_id"].fillna("").astype(str)
        else:
            prepared[_SNAPSHOT_ID_SORT_COL] = pd.Series(
                [f"row_{idx}" for idx in range(len(prepared))],
                index=prepared.index,
                dtype=object,
            )
        return prepared

    def _load_index_report(self) -> dict | None:
        try:
            return inspect_risk_main_indexes()
        except Exception:
            logger.exception("Risk.Main index inspection failed")
            return {"error": "Index inspection failed. Check server logs."}

    def load_full_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        model_key = self._model_key()
        with self._condition:
            while self._dataset_loading:
                self._condition.wait()

            now = time()
            age = now - self._dataset_cache["ts"]
            if (
                not force_refresh
                and not self._dataset_cache["df"].empty
                and self._dataset_cache.get("model_key") == model_key
                and age <= self.dataset_ttl_seconds
            ):
                return self._dataset_cache["df"].copy()

            stale_dataset = (
                self._dataset_cache["df"].copy()
                if not self._dataset_cache["df"].empty and self._dataset_cache.get("model_key") == model_key
                else pd.DataFrame()
            )
            self._dataset_loading = True

        try:
            fresh = fetch_production_risk_main_dataset()
            fresh = self._prepare_snapshot_frame(fresh)
            index_report = self._load_index_report()
        except Exception:
            with self._condition:
                self._dataset_loading = False
                self._condition.notify_all()
            if not force_refresh and not stale_dataset.empty:
                logger.warning(
                    "Dataset refresh failed; serving stale cached dataset rows=%d",
                    len(stale_dataset),
                    exc_info=True,
                )
                return stale_dataset
            raise

        with self._condition:
            self._dataset_cache["ts"] = time()
            self._dataset_cache["df"] = fresh.copy()
            self._dataset_cache["model_key"] = model_key
            self._dataset_loading = False
            self._history_cache.clear()
            self._index_report = index_report
            self._condition.notify_all()
            return fresh.copy()

    def fetch_customer_aggregates(self, customer_ids: list[str], force_refresh: bool = False) -> pd.DataFrame:
        if not customer_ids:
            return pd.DataFrame()

        normalized_ids = tuple(sorted(set(str(value) for value in customer_ids if value is not None and str(value).strip())))
        if not normalized_ids:
            return pd.DataFrame()

        model_key = self._model_key()
        key = (model_key,) + normalized_ids
        now = time()
        with self._lock:
            cached = self._history_cache.get(key)
            if cached is not None:
                ts, cached_df = cached
                if not force_refresh and (now - ts) <= self.history_ttl_seconds:
                    return cached_df.copy()

        full_df = self.load_full_dataset(force_refresh=force_refresh)
        grouped = build_risk_main_customer_aggregates(full_df, list(normalized_ids))

        with self._lock:
            self._history_cache[key] = (now, grouped.copy())
        return grouped

    def enrich_with_customer_history(self, df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
        aggregates = self.fetch_customer_aggregates(
            customer_ids_from_frame(df),
            force_refresh=force_refresh,
        )
        return merge_customer_history_aggregates(df, aggregates)

    def _encode_cursor(self, payload: dict) -> str:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(encoded).decode("utf-8")

    def _decode_cursor(self, cursor: str) -> dict:
        try:
            decoded = base64.urlsafe_b64decode(str(cursor).encode("utf-8")).decode("utf-8")
            payload = json.loads(decoded)
        except Exception as exc:
            raise ValueError(f"Invalid cursor: {type(exc).__name__}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Invalid cursor payload.")
        return payload

    def _resolve_threshold(self, segment: str) -> float | None:
        if self.threshold_resolver is None:
            return None
        return self.threshold_resolver(str(segment or "").strip().lower())

    def _build_scored_snapshot(self, full_df: pd.DataFrame, model_key: str) -> dict:
        prepared_full = self._prepare_snapshot_frame(full_df)
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
                "segment_summaries": {"all": _empty_summary()},
                "segment_counts": {"all": 0},
                "rows": 0,
                "raw_rows": 0,
            }

        enriched_full = self.enrich_with_customer_history(prepared_full, force_refresh=False)
        segment_values = (
            enriched_full[_SNAPSHOT_SEGMENT_COL]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown"})
        )

        merged_frames: list[pd.DataFrame] = []
        unique_segments = sorted(value for value in segment_values.unique().tolist() if value)
        if not unique_segments:
            unique_segments = ["unknown"]

        for segment in unique_segments:
            current_df = enriched_full[segment_values == segment].copy()
            if current_df.empty:
                continue
            scored = score_mongo_frame(
                current_df,
                history_df=enriched_full,
                top_n=5,
                approval_threshold_override=self._resolve_threshold(segment),
                scoring_context=f"bulk_snapshot:{segment}",
            )
            merged_frames.append(response_from_raw(current_df, scored))

        merged = pd.concat(merged_frames, ignore_index=True, sort=False) if merged_frames else enriched_full.iloc[0:0].copy()
        merged = merged.sort_values(
            [_SNAPSHOT_INVOICE_TS_COL, _SNAPSHOT_ID_SORT_COL],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        ordered_segments = (
            merged.get(_SNAPSHOT_SEGMENT_COL, pd.Series("unknown", index=merged.index))
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
            segment_summaries[segment] = _build_scored_summary(shaped.iloc[positions].copy()) if positions else _empty_summary()

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

    def _prune_previous_snapshot_locked(self, now: float) -> None:
        previous = self._previous_scored_snapshot
        if previous is None:
            return
        age = now - float(previous.get("ts", 0.0) or 0.0)
        if age > self.scored_snapshot_retention_seconds:
            self._previous_scored_snapshot = None

    def load_scored_snapshot(self, force_refresh: bool = False, source_df: pd.DataFrame | None = None) -> dict:
        model_key = self._model_key()
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
                    and age <= self.dataset_ttl_seconds
                ):
                    return current

            stale_snapshot = current
            self._scored_loading = True

        try:
            full_df = source_df.copy() if source_df is not None else self.load_full_dataset(force_refresh=force_refresh)
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
        normalized = (segment or "").strip().lower()
        if normalized == "all":
            return list(snapshot.get("segment_positions", {}).get("all", []))
        return list(snapshot.get("segment_positions", {}).get(normalized, []))

    def get_scored_page(
        self,
        *,
        segment: str,
        page_size: int,
        cursor: str | None = None,
        refresh: bool = False,
    ) -> dict:
        segment_value = (segment or "").strip().lower()
        if cursor and refresh:
            raise ValueError("refresh cannot be combined with cursor.")

        if cursor:
            payload = self._decode_cursor(cursor)
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
            next_cursor = self._encode_cursor(
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
            "snapshot_summary": dict(snapshot.get("segment_summaries", {}).get(segment_value, _empty_summary())),
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
        segment_value = (segment or "").strip().lower()
        positions = self._positions_for_segment(snapshot, segment_value)
        all_records = list(snapshot.get("records", []))
        segment_records = [all_records[pos] for pos in positions]
        frame = pd.DataFrame(segment_records) if segment_records else pd.DataFrame()
        return frame, {
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_generated_at": snapshot.get("generated_at"),
            "rows": int(len(segment_records)),
            "total_available": int(len(positions)),
            "summary": dict(snapshot.get("segment_summaries", {}).get(segment_value, _empty_summary())),
        }

    def refresh(self) -> None:
        refresh_start = perf_counter()
        logger.info("API cache refresh started model_key=%s", self._model_key())
        fresh = self.load_full_dataset(force_refresh=True)
        snapshot = self.load_scored_snapshot(force_refresh=True, source_df=fresh)
        with self._lock:
            self._history_cache.clear()
        logger.info(
            "API cache refresh completed raw_rows=%d scored_rows=%d duration_ms=%.1f",
            len(fresh),
            int(snapshot.get("rows", 0) or 0),
            (perf_counter() - refresh_start) * 1000.0,
        )

    def snapshot(self, now: float | None = None) -> dict:
        current_time = time() if now is None else now
        with self._lock:
            dataset_ts = float(self._dataset_cache.get("ts", 0.0) or 0.0)
            dataset_df = self._dataset_cache.get("df", pd.DataFrame())
            history_entries = len(self._history_cache)
            history_ages = [current_time - ts for ts, _ in self._history_cache.values()] if self._history_cache else []
            auto_state = self._auto_refresh_state.copy()
            model_key = self._dataset_cache.get("model_key")
            self._prune_previous_snapshot_locked(current_time)
            current_snapshot = self._scored_snapshot
            previous_snapshot = self._previous_scored_snapshot
            page_cache_entries = len(self._page_cache)
            index_report = None if self._index_report is None else dict(self._index_report)

        dataset_age = (current_time - dataset_ts) if dataset_ts else None
        scored_snapshot_age = None
        previous_snapshot_age = None
        if current_snapshot is not None:
            scored_snapshot_age = current_time - float(current_snapshot.get("ts", 0.0) or 0.0)
        if previous_snapshot is not None:
            previous_snapshot_age = current_time - float(previous_snapshot.get("ts", 0.0) or 0.0)

        return {
            "dataset": {
                "ts": dataset_ts,
                "age_seconds": dataset_age,
                "rows": int(len(dataset_df)) if isinstance(dataset_df, pd.DataFrame) else 0,
                "model_key": model_key,
            },
            "history": {
                "entries": history_entries,
                "max_age_seconds": None if not history_ages else max(history_ages),
            },
            "scored_snapshot": {
                "snapshot_id": None if current_snapshot is None else current_snapshot.get("snapshot_id"),
                "generated_at": None if current_snapshot is None else current_snapshot.get("generated_at"),
                "age_seconds": scored_snapshot_age,
                "rows": 0 if current_snapshot is None else int(current_snapshot.get("rows", 0) or 0),
                "raw_rows": 0 if current_snapshot is None else int(current_snapshot.get("raw_rows", 0) or 0),
                "segment_counts": {} if current_snapshot is None else dict(current_snapshot.get("segment_counts", {})),
                "previous_snapshot_id": None if previous_snapshot is None else previous_snapshot.get("snapshot_id"),
                "previous_snapshot_age_seconds": previous_snapshot_age,
                "page_cache_entries": page_cache_entries,
            },
            "mongo_indexes": index_report,
            "auto_refresh": auto_state,
        }

    def _worker(self) -> None:
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                self.refresh()
                consecutive_errors = 0
                with self._lock:
                    self._auto_refresh_state["last_run_ts"] = time()
                    self._auto_refresh_state["last_error"] = None
                    self._auto_refresh_state["consecutive_errors"] = 0
            except Exception as exc:  # pragma: no cover
                consecutive_errors += 1
                logger.exception("Cache refresh failed (%d consecutive): %s", consecutive_errors, exc)
                if consecutive_errors >= self.max_consecutive_refresh_errors:
                    logger.critical(
                        "Cache refresh has failed %d times in a row. Data is stale. Manual intervention required.",
                        consecutive_errors,
                    )
                with self._lock:
                    self._auto_refresh_state["last_error"] = "Cache refresh failed. Check server logs."
                    self._auto_refresh_state["consecutive_errors"] = consecutive_errors
            self._stop_event.wait(max(self.auto_refresh_interval_seconds, 60))

    def start(self) -> None:
        if not self.auto_refresh_enabled:
            with self._lock:
                self._auto_refresh_state["running"] = False
            return

        with self._lock:
            if self._auto_refresh_state["running"]:
                return

        self._stop_event.clear()
        self._bg_thread = Thread(target=self._worker, name="api-cache-refresh", daemon=True)
        self._bg_thread.start()
        with self._lock:
            self._auto_refresh_state["running"] = True
        logger.info("API cache background refresh started interval_seconds=%d", self.auto_refresh_interval_seconds)

    def stop(self) -> None:
        self._stop_event.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=1.0)
        with self._lock:
            self._auto_refresh_state["running"] = False
        logger.info("API cache background refresh stopped")
