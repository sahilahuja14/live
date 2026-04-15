from __future__ import annotations

from threading import Lock
from typing import Callable

import pandas as pd

from ...logging_config import get_logger
from .customer_risk_store import CustomerRiskStore


logger = get_logger(__name__)


class PortfolioCache:
    def __init__(self, *, lock: Lock, store: CustomerRiskStore | None = None) -> None:
        self._lock = lock
        self._store = store
        self._portfolio_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_customer_portfolio(
        self,
        *,
        segment: str,
        snapshot_id: str,
        builder: Callable[[], pd.DataFrame],
    ) -> pd.DataFrame:
        normalized_segment = str(segment or "").strip().lower()
        normalized_snapshot_id = str(snapshot_id or "").strip()

        if not normalized_snapshot_id:
            logger.warning(
                "Customer portfolio cache skipped because snapshot_id is empty segment=%s",
                normalized_segment,
            )
            return builder()

        cache_key = (normalized_segment, normalized_snapshot_id)
        with self._lock:
            cached = self._portfolio_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        stored = self._load_from_store(segment=normalized_segment, snapshot_id=normalized_snapshot_id)
        if not stored.empty:
            with self._lock:
                self._portfolio_cache[cache_key] = stored.copy()
            return stored.copy()

        portfolio_frame = builder()
        with self._lock:
            self._portfolio_cache[cache_key] = portfolio_frame.copy()

        return portfolio_frame.copy()

    def persist_customer_portfolio(
        self,
        *,
        segment: str,
        snapshot_id: str,
        portfolio_frame: pd.DataFrame,
    ) -> None:
        normalized_segment = str(segment or "").strip().lower()
        normalized_snapshot_id = str(snapshot_id or "").strip()
        if not normalized_snapshot_id or portfolio_frame.empty:
            return

        with self._lock:
            self._portfolio_cache[(normalized_segment, normalized_snapshot_id)] = portfolio_frame.copy()

        if self._store is None or not self._store.enabled:
            return

        try:
            self._store.persist_portfolio(
                segment=normalized_segment,
                snapshot_id=normalized_snapshot_id,
                portfolio_frame=portfolio_frame,
            )
        except Exception:
            logger.warning(
                "Persisting customer portfolio failed segment=%s snapshot_id=%s",
                normalized_segment,
                normalized_snapshot_id,
                exc_info=True,
            )

    def clear(self) -> None:
        with self._lock:
            self._portfolio_cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._portfolio_cache)

    def _load_from_store(self, *, segment: str, snapshot_id: str) -> pd.DataFrame:
        if self._store is None or not self._store.enabled:
            return pd.DataFrame()

        try:
            return self._store.load_portfolio(segment=segment, snapshot_id=snapshot_id)
        except Exception:
            logger.warning(
                "Loading customer portfolio from store failed segment=%s snapshot_id=%s",
                segment,
                snapshot_id,
                exc_info=True,
            )
            return pd.DataFrame()
