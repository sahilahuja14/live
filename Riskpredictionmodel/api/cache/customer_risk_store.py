from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Iterable

import pandas as pd
from pymongo import ReplaceOne

from ...config import get_live_customer_risk_collection
from ...dbconnect import get_live_database
from ...logging_config import get_logger
from ...pipeline.utils import json_safe


logger = get_logger(__name__)


class CustomerRiskStore:
    STORE_VERSION = 1
    _PD_BUCKET_LABELS = {
        0: "0-10%",
        0.1: "10-20%",
        0.2: "20-30%",
        0.3: "30-40%",
        0.4: "40-50%",
        0.5: "50%+",
        "overflow": "50%+",
    }
    _EXCLUDED_PORTFOLIO_FIELDS = {
        "_id": 0,
        "portfolio_key": 0,
        "store_version": 0,
        "snapshot_id": 0,
        "persisted_at_utc": 0,
        "customer.customerId": 0,
        "customer.customerName": 0,
        "shipmentDetails.queryFor": 0,
        "grossAmount": 0,
        "max_pd": 0,
        "customer_total_invoices": 0,
        "customer_avg_delay_days": 0,
        "customer_avg_invoice": 0,
        "customer_delay_rate": 0,
        "historical_customer_delay_rate": 0,
    }

    def __init__(self, *, collection_name: str | None = None) -> None:
        self._collection_name = str(collection_name or get_live_customer_risk_collection()).strip()

    @property
    def enabled(self) -> bool:
        return bool(self._collection_name)

    def load_portfolio(self, *, segment: str, snapshot_id: str | None = None, search: str | None = None) -> pd.DataFrame:
        normalized_segment = str(segment or "").strip().lower()
        if not self.enabled:
            return pd.DataFrame()

        collection = get_live_database()[self._collection_name]
        query = self._portfolio_query(segment=normalized_segment, search=search)
        rows = list(
            collection.find(
                query,
                self._EXCLUDED_PORTFOLIO_FIELDS,
            ).sort(
                [
                    ("pd", -1),
                    ("average_delay_days", -1),
                    ("score", 1),
                    ("customerId", 1),
                ]
            )
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def count_portfolio(self, *, segment: str, search: str | None = None) -> int:
        normalized_segment = str(segment or "").strip().lower()
        if not self.enabled:
            return 0

        collection = get_live_database()[self._collection_name]
        query = self._portfolio_query(segment=normalized_segment, search=search)
        return int(collection.count_documents(query))

    def load_portfolio_page(
        self,
        *,
        segment: str,
        page_size: int,
        offset: int,
        search: str | None = None,
    ) -> tuple[pd.DataFrame, int]:
        normalized_segment = str(segment or "").strip().lower()
        if not self.enabled:
            return pd.DataFrame(), 0

        collection = get_live_database()[self._collection_name]
        query = self._portfolio_query(segment=normalized_segment, search=search)
        total = int(collection.count_documents(query))
        if total <= 0:
            return pd.DataFrame(), 0

        cursor = (
            collection.find(
                query,
                self._EXCLUDED_PORTFOLIO_FIELDS,
            )
            .sort(
                [
                    ("pd", -1),
                    ("average_delay_days", -1),
                    ("score", 1),
                    ("customerId", 1),
                ]
            )
            .skip(max(int(offset), 0))
            .limit(max(int(page_size), 1))
        )
        rows = list(cursor)
        if not rows:
            return pd.DataFrame(), total
        return pd.DataFrame(rows), total

    def summarize_portfolio(self, *, segment: str, search: str | None = None) -> dict:
        normalized_segment = str(segment or "").strip().lower()
        if not self.enabled:
            return self._empty_summary()

        collection = get_live_database()[self._collection_name]
        query = self._portfolio_query(segment=normalized_segment, search=search)
        pipeline = [
            {"$match": query},
            {
                "$facet": {
                    "stats": [
                        {
                            "$group": {
                                "_id": None,
                                "customers": {"$sum": 1},
                                "avg_customer_pd": {"$avg": "$pd"},
                                "avg_customer_score": {"$avg": "$score"},
                                "avg_delay_days": {"$avg": "$average_delay_days"},
                                "avg_actual_delay_rate": {"$avg": "$actual_delay_rate"},
                                "total_invoice_rows": {"$sum": {"$ifNull": ["$invoice_rows_scored", 0]}},
                                "total_open_invoices": {"$sum": {"$ifNull": ["$open_invoices", 0]}},
                                "total_paid_invoices": {"$sum": {"$ifNull": ["$paid_invoices", 0]}},
                                "total_amount": {"$sum": {"$ifNull": ["$totalAmountB", 0]}},
                                "avg_invoice_amount": {"$avg": "$average_invoice_amount"},
                            }
                        }
                    ],
                    "approval_mix": [
                        {
                            "$group": {
                                "_id": {"$ifNull": ["$approval", "Unknown"]},
                                "count": {"$sum": 1},
                            }
                        }
                    ],
                    "risk_band_mix": [
                        {
                            "$group": {
                                "_id": {"$ifNull": ["$risk_band", "Unknown"]},
                                "count": {"$sum": 1},
                            }
                        }
                    ],
                    "top_customers_by_pd": [
                        {"$sort": {"pd": -1, "average_delay_days": -1, "score": 1, "customerId": 1}},
                        {"$limit": 5},
                        {
                            "$project": {
                                "_id": 0,
                                "portfolio_key": 0,
                                "store_version": 0,
                                "snapshot_id": 0,
                                "persisted_at_utc": 0,
                                "customer.customerId": 0,
                                "customer.customerName": 0,
                                "shipmentDetails.queryFor": 0,
                                "grossAmount": 0,
                                "max_pd": 0,
                                "customer_total_invoices": 0,
                                "customer_avg_delay_days": 0,
                                "customer_avg_invoice": 0,
                                "customer_delay_rate": 0,
                                "historical_customer_delay_rate": 0,
                            }
                        },
                    ],
                    "pd_histogram": [
                        {
                            "$bucket": {
                                "groupBy": {"$ifNull": ["$pd", 0]},
                                "boundaries": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.000001],
                                "default": "overflow",
                                "output": {"count": {"$sum": 1}},
                            }
                        }
                    ],
                    "latest_meta": [
                        {"$sort": {"persisted_at_utc": -1, "snapshot_id": -1}},
                        {"$limit": 1},
                        {
                            "$project": {
                                "_id": 0,
                                "latest_snapshot_id": "$snapshot_id",
                                "latest_persisted_at_utc": "$persisted_at_utc",
                            }
                        },
                    ],
                }
            },
        ]

        result = list(collection.aggregate(pipeline, allowDiskUse=False))
        if not result:
            return self._empty_summary()

        payload = result[0]
        stats = (payload.get("stats") or [{}])[0]
        if not stats:
            return self._empty_summary()
        latest_meta = (payload.get("latest_meta") or [{}])[0]

        return {
            "customers": int(stats.get("customers") or 0),
            "avg_customer_pd": self._round_or_none(stats.get("avg_customer_pd"), 6),
            "avg_customer_score": self._round_or_none(stats.get("avg_customer_score"), 2),
            "avg_delay_days": self._round_or_none(stats.get("avg_delay_days"), 2),
            "avg_actual_delay_rate": self._round_or_none(stats.get("avg_actual_delay_rate"), 6),
            "total_invoice_rows": int(stats.get("total_invoice_rows") or 0),
            "total_open_invoices": int(stats.get("total_open_invoices") or 0),
            "total_paid_invoices": int(stats.get("total_paid_invoices") or 0),
            "total_amount": self._round_or_none(stats.get("total_amount"), 2),
            "avg_invoice_amount": self._round_or_none(stats.get("avg_invoice_amount"), 2),
            "approval_mix": self._count_map(payload.get("approval_mix") or []),
            "risk_band_mix": self._count_map(payload.get("risk_band_mix") or []),
            "top_customers_by_pd": list(payload.get("top_customers_by_pd") or []),
            "pd_histogram": self._format_pd_histogram(payload.get("pd_histogram") or []),
            "latest_snapshot_id": latest_meta.get("latest_snapshot_id"),
            "latest_persisted_at_utc": latest_meta.get("latest_persisted_at_utc"),
        }

    def load_customer_record(self, *, segment: str, customer_id: str) -> dict | None:
        normalized_segment = str(segment or "").strip().lower()
        normalized_customer_id = str(customer_id or "").strip()
        if not self.enabled or not normalized_customer_id:
            return None

        collection = get_live_database()[self._collection_name]
        doc = collection.find_one(
            {
                "store_version": self.STORE_VERSION,
                "portfolio_key": self._portfolio_key(normalized_segment, normalized_customer_id),
            },
            self._EXCLUDED_PORTFOLIO_FIELDS,
        )
        if not doc:
            return None
        return dict(doc)

    def resolve_customer_lookup(self, *, segment: str, lookup_value: str) -> dict | None:
        normalized_segment = str(segment or "").strip().lower()
        normalized_lookup = str(lookup_value or "").strip()
        if not self.enabled or not normalized_lookup:
            return None

        collection = get_live_database()[self._collection_name]
        queries = [
            {"customerId": normalized_lookup},
            {"customerId": {"$regex": f"^{re.escape(normalized_lookup)}$", "$options": "i"}},
            {"customerName": {"$regex": f"^{re.escape(normalized_lookup)}$", "$options": "i"}},
            {"customerId": {"$regex": re.escape(normalized_lookup), "$options": "i"}},
            {"customerName": {"$regex": re.escape(normalized_lookup), "$options": "i"}},
        ]
        for candidate in queries:
            cursor = (
                collection.find(
                    {
                        "store_version": self.STORE_VERSION,
                        "segment": normalized_segment,
                        **candidate,
                    },
                    self._EXCLUDED_PORTFOLIO_FIELDS,
                )
                .sort(
                    [
                        ("pd", -1),
                        ("average_delay_days", -1),
                        ("score", 1),
                        ("customerId", 1),
                    ]
                )
                .limit(1)
            )
            rows = list(cursor)
            if rows:
                return dict(rows[0])
        return None

    def persist_portfolio(self, *, segment: str, snapshot_id: str, portfolio_frame: pd.DataFrame) -> None:
        normalized_snapshot_id = str(snapshot_id or "").strip()
        normalized_segment = str(segment or "").strip().lower()
        if not self.enabled or not normalized_snapshot_id or portfolio_frame.empty:
            return

        operations = self._build_upserts(
            segment=normalized_segment,
            snapshot_id=normalized_snapshot_id,
            records=portfolio_frame.to_dict(orient="records"),
        )
        if not operations:
            return

        collection = get_live_database()[self._collection_name]
        collection.bulk_write(operations, ordered=False)
        logger.info(
            "Persisted customer portfolio rows=%d segment=%s snapshot_id=%s collection=%s",
            len(operations),
            normalized_segment,
            normalized_snapshot_id,
            self._collection_name,
        )

    def _build_upserts(
        self,
        *,
        segment: str,
        snapshot_id: str,
        records: Iterable[dict],
    ) -> list[ReplaceOne]:
        persisted_at = datetime.now(timezone.utc).isoformat()
        operations: list[ReplaceOne] = []
        for raw_record in records:
            customer_id = str(raw_record.get("customerId") or raw_record.get("customer.customerId") or "").strip()
            if not customer_id:
                continue

            record = dict(json_safe(raw_record))
            record["customerId"] = customer_id
            record["segment"] = segment
            record["snapshot_id"] = snapshot_id
            record["persisted_at_utc"] = persisted_at
            record["portfolio_key"] = self._portfolio_key(segment, customer_id)
            record["store_version"] = self.STORE_VERSION

            operations.append(
                ReplaceOne(
                    {
                        "store_version": self.STORE_VERSION,
                        "portfolio_key": record["portfolio_key"],
                    },
                    record,
                    upsert=True,
                )
            )
        return operations

    def _portfolio_key(self, segment: str, customer_id: str) -> str:
        return f"{str(segment).strip().lower()}::{str(customer_id).strip()}"

    def _portfolio_query(self, *, segment: str, search: str | None) -> dict:
        query: dict = {
            "store_version": self.STORE_VERSION,
            "segment": segment,
        }

        normalized_search = str(search or "").strip()
        if not normalized_search:
            return query

        pattern = re.escape(normalized_search)
        query["$or"] = [
            {"customerId": {"$regex": pattern, "$options": "i"}},
            {"customerName": {"$regex": pattern, "$options": "i"}},
        ]
        return query

    def _empty_summary(self) -> dict:
        return {
            "customers": 0,
            "avg_customer_pd": None,
            "avg_customer_score": None,
            "avg_delay_days": None,
            "avg_actual_delay_rate": None,
            "total_invoice_rows": 0,
            "total_open_invoices": 0,
            "total_paid_invoices": 0,
            "total_amount": None,
            "avg_invoice_amount": None,
            "approval_mix": {},
            "risk_band_mix": {},
            "top_customers_by_pd": [],
            "pd_histogram": self._empty_pd_histogram(),
            "latest_snapshot_id": None,
            "latest_persisted_at_utc": None,
        }

    def _count_map(self, rows: list[dict]) -> dict[str, int]:
        output: dict[str, int] = {}
        for row in rows:
            key = str(row.get("_id") or "Unknown")
            output[key] = int(row.get("count") or 0)
        return output

    def _round_or_none(self, value, digits: int):
        if value is None:
            return None
        try:
            return round(float(value), digits)
        except Exception:
            return None

    def _empty_pd_histogram(self) -> list[dict]:
        return [
            {"range": "0-10%", "count": 0},
            {"range": "10-20%", "count": 0},
            {"range": "20-30%", "count": 0},
            {"range": "30-40%", "count": 0},
            {"range": "40-50%", "count": 0},
            {"range": "50%+", "count": 0},
        ]

    def _format_pd_histogram(self, rows: list[dict]) -> list[dict]:
        counts = {item["range"]: 0 for item in self._empty_pd_histogram()}
        for row in rows:
            bucket_key = row.get("_id")
            label = self._PD_BUCKET_LABELS.get(bucket_key)
            if not label:
                continue
            counts[label] = int(row.get("count") or 0)
        return [{"range": label, "count": counts[label]} for label in counts]
