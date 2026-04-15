from __future__ import annotations

from threading import Condition, Lock
from time import time
from typing import Callable

import pandas as pd

from ...features.customer_aggregates import customer_ids_from_frame, merge_customer_history_aggregates
from ...features.registry import SEGMENT_COL
from ...logging_config import get_logger
from ...pipeline.parsing import parse_main_date
from ...pipeline.risk_canonical import inspect_risk_main_indexes
from ...pipeline.risk_main import (
    build_risk_main_customer_aggregates,
    fetch_production_risk_main_customer_aggregates,
    fetch_production_risk_main_dataset,
)


logger = get_logger(__name__)

SNAPSHOT_SEGMENT_COL = "__snapshot_segment"
SNAPSHOT_INVOICE_TS_COL = "__snapshot_invoice_ts"
SNAPSHOT_ID_SORT_COL = "__snapshot_id_sort"
_NULL_TEXT_MARKERS = {"", "nan", "none", "<na>", "nat"}


class DatasetCache:
    def __init__(
        self,
        *,
        lock: Lock,
        condition: Condition,
        dataset_ttl_seconds: int,
        history_ttl_seconds: int,
        model_key_loader: Callable[[], str],
        on_dataset_refresh: Callable[[], None] | None = None,
    ) -> None:
        self._lock = lock
        self._condition = condition
        self._dataset_ttl_seconds = int(dataset_ttl_seconds)
        self._history_ttl_seconds = int(history_ttl_seconds)
        self._model_key_loader = model_key_loader
        self._on_dataset_refresh = on_dataset_refresh

        self._dataset_cache = {"ts": 0.0, "df": pd.DataFrame(), "model_key": None}
        self._dataset_loading = False
        self._history_cache: dict[tuple[str, ...], tuple[float, pd.DataFrame]] = {}
        self._index_report: dict | None = None

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

        prepared[SNAPSHOT_SEGMENT_COL] = (
            segment_values.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown"})
        )

        if "invoiceDate" in prepared.columns:
            prepared[SNAPSHOT_INVOICE_TS_COL] = prepared["invoiceDate"].apply(parse_main_date)
        else:
            prepared[SNAPSHOT_INVOICE_TS_COL] = pd.Series(pd.NaT, index=prepared.index)

        if "_id" in prepared.columns:
            prepared[SNAPSHOT_ID_SORT_COL] = prepared["_id"].fillna("").astype(str)
        else:
            prepared[SNAPSHOT_ID_SORT_COL] = pd.Series(
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
        model_key = self._model_key_loader()
        with self._condition:
            while self._dataset_loading:
                self._condition.wait()

            now = time()
            age = now - float(self._dataset_cache["ts"] or 0.0)
            if (
                not force_refresh
                and not self._dataset_cache["df"].empty
                and self._dataset_cache.get("model_key") == model_key
                and age <= self._dataset_ttl_seconds
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

        if self._on_dataset_refresh is not None:
            self._on_dataset_refresh()

        return fresh.copy()

    def fetch_customer_aggregates(self, customer_ids: list[str], force_refresh: bool = False) -> pd.DataFrame:
        if not customer_ids:
            return pd.DataFrame()

        normalized_ids = tuple(
            sorted(
                {
                    cleaned
                    for value in customer_ids
                    for cleaned in [str(value).strip()]
                    if cleaned and cleaned.lower() not in _NULL_TEXT_MARKERS
                }
            )
        )
        if not normalized_ids:
            return pd.DataFrame()

        model_key = self._model_key_loader()
        cache_key = (model_key,) + normalized_ids
        now = time()
        with self._lock:
            cached = self._history_cache.get(cache_key)
            if cached is not None:
                cached_ts, cached_df = cached
                if not force_refresh and (now - cached_ts) <= self._history_ttl_seconds:
                    return cached_df.copy()

        try:
            grouped = fetch_production_risk_main_customer_aggregates(list(normalized_ids))
        except Exception:
            logger.warning(
                "Targeted customer aggregate fetch failed; falling back to cached dataset customer_count=%d",
                len(normalized_ids),
                exc_info=True,
            )
            full_df = self.load_full_dataset(force_refresh=force_refresh)
            grouped = build_risk_main_customer_aggregates(full_df, list(normalized_ids))

        with self._lock:
            self._history_cache[cache_key] = (now, grouped.copy())

        return grouped

    def fetch_all_customer_aggregates_from_frame(
        self,
        full_df: pd.DataFrame,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if full_df is None or full_df.empty:
            return pd.DataFrame()

        customer_ids = customer_ids_from_frame(full_df)
        if not customer_ids:
            return pd.DataFrame()

        model_key = self._model_key_loader()
        cache_key = (model_key, "__all_customers__")
        now = time()
        with self._lock:
            cached = self._history_cache.get(cache_key)
            if cached is not None:
                cached_ts, cached_df = cached
                if not force_refresh and (now - cached_ts) <= self._history_ttl_seconds:
                    return cached_df.copy()

        grouped = build_risk_main_customer_aggregates(full_df, customer_ids)
        with self._lock:
            self._history_cache[cache_key] = (now, grouped.copy())

        return grouped

    def enrich_with_customer_history(self, df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
        aggregates = self.fetch_customer_aggregates(
            customer_ids_from_frame(df),
            force_refresh=force_refresh,
        )
        return merge_customer_history_aggregates(df, aggregates)

    def enrich_snapshot_with_customer_history(self, df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
        aggregates = self.fetch_all_customer_aggregates_from_frame(
            df,
            force_refresh=force_refresh,
        )
        return merge_customer_history_aggregates(df, aggregates)

    def clear_history_cache(self) -> None:
        with self._lock:
            self._history_cache.clear()

    def snapshot_state(self, *, now: float) -> dict:
        with self._lock:
            dataset_ts = float(self._dataset_cache.get("ts", 0.0) or 0.0)
            dataset_df = self._dataset_cache.get("df", pd.DataFrame())
            history_entries = len(self._history_cache)
            history_ages = [now - ts for ts, _ in self._history_cache.values()] if self._history_cache else []
            model_key = self._dataset_cache.get("model_key")
            index_report = None if self._index_report is None else dict(self._index_report)

        dataset_age = (now - dataset_ts) if dataset_ts else None
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
            "mongo_indexes": index_report,
        }
