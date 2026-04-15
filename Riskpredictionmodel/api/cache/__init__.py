from __future__ import annotations

import os
from threading import Condition, Event, Lock, Thread
from time import perf_counter, time
from typing import Callable

import pandas as pd

from ...logging_config import get_logger
from ...scoring.model import describe_active_production_model
from .customer_risk_store import CustomerRiskStore
from .dataset_cache import DatasetCache
from .portfolio_cache import PortfolioCache
from .snapshot_cache import SnapshotCache


logger = get_logger(__name__)


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
        broadcaster=None,
    ) -> None:
        self.dataset_ttl_seconds = int(dataset_ttl_seconds)
        self.history_ttl_seconds = int(history_ttl_seconds)
        self.auto_refresh_enabled = bool(auto_refresh_enabled)
        self.auto_refresh_interval_seconds = int(auto_refresh_interval_seconds)
        self.scored_snapshot_retention_seconds = int(scored_snapshot_retention_seconds)
        self.threshold_resolver = threshold_resolver
        self._broadcaster = broadcaster
        self.max_consecutive_refresh_errors = max(int(os.getenv("MAX_REFRESH_ERRORS", "5")), 1)

        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._stop_event = Event()
        self._bg_thread: Thread | None = None
        self._auto_refresh_state = {
            "running": False,
            "last_run_ts": 0.0,
            "last_error": None,
            "consecutive_errors": 0,
            "refresh_in_progress": False,
            "refresh_trigger": None,
        }

        self._customer_risk_store = CustomerRiskStore()
        self._portfolio_cache = PortfolioCache(lock=self._lock, store=self._customer_risk_store)
        self._dataset_cache = DatasetCache(
            lock=self._lock,
            condition=self._condition,
            dataset_ttl_seconds=self.dataset_ttl_seconds,
            history_ttl_seconds=self.history_ttl_seconds,
            model_key_loader=self._model_key,
            on_dataset_refresh=self._portfolio_cache.clear,
        )
        self._snapshot_cache = SnapshotCache(
            lock=self._lock,
            condition=self._condition,
            dataset_ttl_seconds=self.dataset_ttl_seconds,
            scored_snapshot_retention_seconds=self.scored_snapshot_retention_seconds,
            model_key_loader=self._model_key,
            load_full_dataset=self._dataset_cache.load_full_dataset,
            enrich_snapshot_with_customer_history=self._dataset_cache.enrich_snapshot_with_customer_history,
            resolve_threshold=self._resolve_threshold,
            broadcaster=self._broadcaster,
            persist_customer_portfolio=self._portfolio_cache.persist_customer_portfolio,
        )

    def _model_key(self) -> str:
        descriptor = describe_active_production_model()
        return f"{descriptor['model_family']}::{descriptor['version']}"

    def _resolve_threshold(self, segment: str) -> float | None:
        if self.threshold_resolver is None:
            return None
        return self.threshold_resolver(str(segment or "").strip().lower())

    def load_full_dataset(self, force_refresh: bool = False) -> pd.DataFrame:
        return self._dataset_cache.load_full_dataset(force_refresh=force_refresh)

    def fetch_customer_aggregates(self, customer_ids: list[str], force_refresh: bool = False) -> pd.DataFrame:
        return self._dataset_cache.fetch_customer_aggregates(customer_ids, force_refresh=force_refresh)

    def enrich_with_customer_history(self, df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
        return self._dataset_cache.enrich_with_customer_history(df, force_refresh=force_refresh)

    def load_scored_snapshot(self, force_refresh: bool = False, source_df: pd.DataFrame | None = None) -> dict:
        return self._snapshot_cache.load_scored_snapshot(force_refresh=force_refresh, source_df=source_df)

    def get_scored_page(
        self,
        *,
        segment: str,
        page_size: int,
        cursor: str | None = None,
        refresh: bool = False,
    ) -> dict:
        return self._snapshot_cache.get_scored_page(
            segment=segment,
            page_size=page_size,
            cursor=cursor,
            refresh=refresh,
        )

    def get_snapshot_customer_frame(
        self,
        *,
        snapshot_id: str,
        segment: str,
        customer_id: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        return self._snapshot_cache.get_snapshot_customer_frame(
            snapshot_id=snapshot_id,
            segment=segment,
            customer_id=customer_id,
            limit=limit,
        )

    def get_scored_segment_frame(self, *, segment: str, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
        return self._snapshot_cache.get_scored_segment_frame(segment=segment, refresh=refresh)

    def get_customer_portfolio(
        self,
        *,
        segment: str,
        snapshot_id: str,
        builder: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        return self._portfolio_cache.get_customer_portfolio(
            segment=segment,
            snapshot_id=snapshot_id,
            builder=builder,
        )

    def get_persisted_customer_portfolio_frame(self, *, segment: str, search: str | None = None) -> pd.DataFrame:
        return self._customer_risk_store.load_portfolio(segment=segment, search=search)

    def get_persisted_customer_portfolio_count(self, *, segment: str, search: str | None = None) -> int:
        return self._customer_risk_store.count_portfolio(segment=segment, search=search)

    def get_persisted_customer_portfolio_page(
        self,
        *,
        segment: str,
        page_size: int,
        offset: int,
        search: str | None = None,
    ) -> tuple[pd.DataFrame, int]:
        return self._customer_risk_store.load_portfolio_page(
            segment=segment,
            page_size=page_size,
            offset=offset,
            search=search,
        )

    def get_persisted_customer_portfolio_summary(self, *, segment: str, search: str | None = None) -> dict:
        return self._customer_risk_store.summarize_portfolio(segment=segment, search=search)

    def get_persisted_customer_record(self, *, segment: str, customer_id: str) -> dict | None:
        return self._customer_risk_store.load_customer_record(segment=segment, customer_id=customer_id)

    def resolve_persisted_customer_lookup(self, *, segment: str, lookup_value: str) -> dict | None:
        return self._customer_risk_store.resolve_customer_lookup(segment=segment, lookup_value=lookup_value)

    def refresh(self, *, trigger: str = "manual") -> None:
        with self._condition:
            if self._auto_refresh_state["refresh_in_progress"]:
                active_trigger = self._auto_refresh_state.get("refresh_trigger")
                logger.info(
                    "API cache refresh join trigger=%s active_trigger=%s",
                    trigger,
                    active_trigger,
                )
                while self._auto_refresh_state["refresh_in_progress"]:
                    self._condition.wait()
                return

            self._auto_refresh_state["refresh_in_progress"] = True
            self._auto_refresh_state["refresh_trigger"] = trigger

        refresh_start = perf_counter()
        logger.info("API cache refresh started model_key=%s trigger=%s", self._model_key(), trigger)
        try:
            fresh = self.load_full_dataset(force_refresh=True)
            snapshot = self.load_scored_snapshot(force_refresh=True, source_df=fresh)
            self._dataset_cache.clear_history_cache()
            self._portfolio_cache.clear()
            self._snapshot_cache.clear_runtime_caches()
            logger.info(
                "API cache refresh completed raw_rows=%d scored_rows=%d duration_ms=%.1f trigger=%s",
                len(fresh),
                int(snapshot.get("rows", 0) or 0),
                (perf_counter() - refresh_start) * 1000.0,
                trigger,
            )
        finally:
            with self._condition:
                self._auto_refresh_state["refresh_in_progress"] = False
                self._auto_refresh_state["refresh_trigger"] = None
                self._condition.notify_all()

    def snapshot(self, now: float | None = None) -> dict:
        current_time = time() if now is None else now
        dataset_state = self._dataset_cache.snapshot_state(now=current_time)
        snapshot_state = self._snapshot_cache.snapshot_state(now=current_time)
        with self._lock:
            auto_state = dict(self._auto_refresh_state)

        snapshot_state["customer_portfolio_cache_entries"] = self._portfolio_cache.size()
        return {
            "dataset": dataset_state["dataset"],
            "history": dataset_state["history"],
            "scored_snapshot": snapshot_state,
            "mongo_indexes": dataset_state["mongo_indexes"],
            "auto_refresh": auto_state,
        }

    def _worker(self) -> None:
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                if self._broadcaster is not None:
                    self._broadcaster.notify_refresh_started_threadsafe("auto_refresh")
                self.refresh(trigger="auto_refresh")
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
