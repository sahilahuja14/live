from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from ..config import get_live_db_name, get_live_invoice_collection
from ..dbconnect import get_live_database
from ..logging_config import get_logger
from .parsing import parse_main_date, safe_numeric, safe_text
from .risk_map import (
    LIVE_PASSTHROUGH_FIELDS,
    LIVE_PROFILE_COLLECTIONS,
    LIVE_TO_NORMALIZED,
    NORMALIZED_TO_CANONICAL_FIELD_MAP,
    PRODUCTION_RISK_COLLECTION,
    PRODUCTION_RISK_DB_NAME,
    TARGET_DELAY_DAYS,
)
from .utils import flatten_dict, is_missing, json_safe, safe_ratio, write_json


logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_OUTPUT_DIR = PROJECT_ROOT / "pipeline" / "schema_inventory"
LIVE_FETCH_BATCH_SIZE = max(
    int(os.getenv("API_LIVE_FETCH_BATCH_SIZE") or os.getenv("API_RISK_MAIN_FETCH_BATCH_SIZE", "1000")),
    1,
)

PAYMENT_COLLECTION = "paymenttransactions"
CUSTOMER_COLLECTION = "customermasters"
QUERIES_COLLECTION = "queries"
BOOKINGS_COLLECTION = "bookings"
GATEINS_COLLECTION = "gateins"
CLEARANCE_COLLECTION = "clearencedocs"

_SETTLED_MARKERS = {"settled", "paid", "received", "completed", "success", "captured"}
_REJECTED_MARKERS = {"cancel", "reject", "fail", "reverse", "void"}
_PAYMENT_DATE_FIELDS = (
    "paymentDate",
    "utrDate",
    "payment_date",
    "transactionDate",
    "transaction_date",
    "valueDate",
    "value_date",
    "createdAt",
    "created_at",
    "date",
)
_PAYMENT_STATUS_FIELDS = (
    "paymentstatus",
    "status",
    "paymentStatus",
    "settlementStatus",
    "transactionStatus",
    "paymentServer.status",
)
_PAYMENT_RAW_FIELDS = (
    "paymentReferenceNo",
    "referenceNo",
    "utrId",
    "paymentDateRaw",
    "paymentDate",
    "paymentDetails",
    "remarks",
)

_LAST_LIVE_COVERAGE: dict[str, Any] = {}
_LAST_COLLECTIONS_FOUND: list[str] = []
MIN_VALID_TERMS_DAYS = 0.0
MAX_VALID_TERMS_DAYS = 365.0


def _series_from_frame(df: pd.DataFrame, column: str, default=None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def _coerce_terms_days(value):
    numeric = safe_numeric(value)
    if numeric is not None:
        if MIN_VALID_TERMS_DAYS <= numeric <= MAX_VALID_TERMS_DAYS:
            return numeric
        return None

    text = str(value).strip() if value is not None else ""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        try:
            parsed = float(match.group(0))
        except ValueError:
            return None
        if MIN_VALID_TERMS_DAYS <= parsed <= MAX_VALID_TERMS_DAYS:
            return parsed
    return None


def _build_live_projection() -> dict[str, int]:
    # Parent paths must win over child paths to avoid Mongo projection collisions.
    all_paths = set(LIVE_TO_NORMALIZED) | set(LIVE_PASSTHROUGH_FIELDS)
    sorted_paths = sorted(all_paths, key=lambda path: (path.count("."), path))

    selected: list[str] = []
    for path in sorted_paths:
        already_covered = any(path == parent or path.startswith(f"{parent}.") for parent in selected)
        if already_covered:
            continue
        selected = [existing for existing in selected if not existing.startswith(f"{path}.")]
        selected.append(path)

    return {"_id": 1, **{path: 1 for path in selected}}


def _store_live_diagnostics(*, coverage: dict[str, Any], collections: list[str]) -> None:
    global _LAST_LIVE_COVERAGE, _LAST_COLLECTIONS_FOUND
    _LAST_LIVE_COVERAGE = dict(coverage)
    _LAST_COLLECTIONS_FOUND = list(collections)


def get_live_diagnostics() -> dict[str, Any]:
    return {
        "coverage": dict(_LAST_LIVE_COVERAGE),
        "live_collections_found": list(_LAST_COLLECTIONS_FOUND),
    }


def _normalize_live_doc(doc: dict) -> dict:
    flat = flatten_dict(doc)
    normalized: dict[str, Any] = {}
    if "_id" in doc:
        normalized["_id"] = doc.get("_id")

    for live_path, normalized_path in LIVE_TO_NORMALIZED.items():
        if normalized_path is None or live_path not in flat:
            continue
        value = flat.get(live_path)
        if is_missing(value) or normalized_path in normalized:
            continue
        normalized[normalized_path] = value

    for live_path, normalized_path in LIVE_PASSTHROUGH_FIELDS.items():
        if live_path not in flat:
            continue
        value = flat.get(live_path)
        if is_missing(value) or normalized_path in normalized:
            continue
        normalized[normalized_path] = value

    return normalized


def _normalize_key(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = safe_text(value)
    if text is not None:
        return text
    return str(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if not is_missing(value):
            return value
    return None


def _column_values(frame: pd.DataFrame, column: str) -> list[Any]:
    if column not in frame.columns:
        return []
    return [value for value in frame[column].tolist() if not is_missing(value)]


def _column_value(frame: pd.DataFrame, idx: int, column: str) -> Any:
    if column not in frame.columns:
        return None
    return frame.at[idx, column]


def _collection_present(collection_name: str, available_collections: set[str] | None) -> bool:
    if available_collections is None:
        return True
    return collection_name in available_collections


def _query_from_clauses(clauses: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


def _airport_bits(raw_value: Any) -> dict[str, str | None]:
    text = safe_text(raw_value)
    if text is None:
        return {"city": None, "country": None}

    country = None
    city = text
    if "(" in text and ")" in text:
        country = text.split("(", 1)[1].split(")", 1)[0].strip()
        before = text.split("(", 1)[0].strip()
        if "," in before:
            city = before.split(",", 1)[1].strip()
        else:
            city = before
    elif "," in text:
        city = text.split(",", 1)[1].strip()
    return {"city": city or None, "country": country or None}


def _doc_timestamp(flat: dict[str, Any]) -> pd.Timestamp:
    for field in ("updatedAt", "createdAt", "paymentDate", "utrDate", "invoiceDate", "ClearenceDate", "clearenceDate"):
        parsed = parse_main_date(flat.get(field))
        if pd.notna(parsed):
            return parsed
    return pd.Timestamp.min


def _latest_record(lookup: dict[str, dict[str, Any]], key: Any, flat: dict[str, Any]) -> None:
    normalized = _normalize_key(key)
    if normalized is None:
        return

    current = lookup.get(normalized)
    if current is None or _doc_timestamp(flat) >= _doc_timestamp(current):
        lookup[normalized] = flat


def _fill_missing(frame: pd.DataFrame, idx: int, column: str, value: Any) -> None:
    if is_missing(value):
        return
    if column not in frame.columns or is_missing(frame.at[idx, column]):
        frame.at[idx, column] = value


def _has_value(series: pd.Series) -> pd.Series:
    return series.map(lambda value: not is_missing(value))


def _coverage_ratio(frame: pd.DataFrame, columns: list[str], *, require_all: bool = False) -> float:
    rows = len(frame)
    if rows == 0:
        return 0.0

    if require_all:
        filled = pd.Series(True, index=frame.index)
        for column in columns:
            column_mask = _has_value(frame[column]) if column in frame.columns else pd.Series(False, index=frame.index)
            filled &= column_mask
        return round(float(filled.mean() * 100.0), 2)

    filled = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            filled |= _has_value(frame[column])
    return round(float(filled.mean() * 100.0), 2)


def compute_live_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "invoice_key_pct": 0.0,
            "invoice_date_pct": 0.0,
            "due_date_or_terms_pct": 0.0,
            "payment_date_pct": 0.0,
            "customer_id_pct": 0.0,
            "receivables_complete_pct": 0.0,
            "admin_operational_pct": 0.0,
        }

    admin_columns = [
        "salesPersonName",
        "shipmentDetails.queryFor",
        "shipmentDetails.accountType",
        "shipmentDetails.incoTerms",
        "shipmentDetails.originCity",
        "shipmentDetails.destinationCity",
        "operational.jobNo",
        "operational.bookingNo",
    ]
    receivable_columns = [
        "receivables.notDueAmount",
        "receivables.bucket0To15Amount",
        "receivables.bucket16To30Amount",
        "receivables.bucket31To45Amount",
        "receivables.bucket46To60Amount",
        "receivables.bucket60To90Amount",
        "receivables.bucketAbove90Amount",
    ]
    return {
        "rows": int(len(frame)),
        "invoice_key_pct": _coverage_ratio(frame, ["invoiceNo"]),
        "invoice_date_pct": _coverage_ratio(frame, ["invoiceDate"]),
        "due_date_or_terms_pct": _coverage_ratio(frame, ["invoiceDueDate", "paymentTerms"]),
        "payment_date_pct": _coverage_ratio(frame, ["paymentDate"]),
        "customer_id_pct": _coverage_ratio(frame, ["customer.customerId"]),
        "receivables_complete_pct": _coverage_ratio(frame, receivable_columns, require_all=True),
        "admin_operational_pct": _coverage_ratio(frame, admin_columns),
    }


def _payment_is_rejected(flat: dict[str, Any]) -> bool:
    for field in _PAYMENT_STATUS_FIELDS:
        status = safe_text(flat.get(field))
        if not status:
            continue
        lowered = status.lower()
        if any(marker in lowered for marker in _REJECTED_MARKERS):
            return True
        if any(marker in lowered for marker in _SETTLED_MARKERS):
            return False
    return False


def _payment_date(flat: dict[str, Any]) -> pd.Timestamp | pd.NaT:
    for field in _PAYMENT_DATE_FIELDS:
        parsed = parse_main_date(flat.get(field))
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def _payment_raw(flat: dict[str, Any]) -> str | None:
    for field in _PAYMENT_RAW_FIELDS:
        value = safe_text(flat.get(field))
        if value:
            return value
    return None


def _merge_customer_enrichment(frame: pd.DataFrame, database, available_collections: set[str] | None = None) -> pd.DataFrame:
    if frame.empty or not _collection_present(CUSTOMER_COLLECTION, available_collections):
        return frame
    if "customer.customerId" not in frame.columns:
        return frame

    customer_ids = _column_values(frame, "customer.customerId")
    if not customer_ids:
        return frame

    query = {
        "$or": [
            {"_id": {"$in": customer_ids}},
            {"customerId": {"$in": customer_ids}},
            {"customerCode": {"$in": customer_ids}},
        ]
    }
    projection = {
        "_id": 1,
        "customerId": 1,
        "customerCode": 1,
        "customerName": 1,
        "customerAccountType": 1,
        "customerType": 1,
        "custType": 1,
        "category": 1,
        "createdAt": 1,
        "custCurrency": 1,
        "customerCurrency": 1,
    }

    lookup: dict[str, dict[str, Any]] = {}
    for doc in database[CUSTOMER_COLLECTION].find(query, projection):
        flat = flatten_dict(doc)
        _latest_record(lookup, doc.get("_id"), flat)
        _latest_record(lookup, flat.get("customerId"), flat)
        _latest_record(lookup, flat.get("customerCode"), flat)

    enriched = frame.copy()
    for idx, row in enriched.iterrows():
        record = lookup.get(_normalize_key(row.get("customer.customerId")))
        if not record:
            continue

        category = record.get("category")
        if isinstance(category, list):
            category = _first_present(*category)
        _fill_missing(enriched, idx, "customer.customerName", record.get("customerName"))
        _fill_missing(enriched, idx, "customer.customerAccountType", record.get("customerAccountType"))
        _fill_missing(enriched, idx, "customer.customerType", _first_present(record.get("customerType"), record.get("custType")))
        _fill_missing(enriched, idx, "customer.category", category)
        _fill_missing(enriched, idx, "customer.onboardDate", record.get("createdAt"))
        _fill_missing(enriched, idx, "selectedCustomerCurrency", _first_present(record.get("custCurrency"), record.get("customerCurrency")))
    return enriched


def _build_booking_lookup(
    database,
    frame: pd.DataFrame,
    available_collections: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_booking_no: dict[str, dict[str, Any]] = {}
    by_invoice_no: dict[str, dict[str, Any]] = {}
    if not _collection_present(BOOKINGS_COLLECTION, available_collections):
        return by_id, by_booking_no, by_invoice_no

    clauses = []
    booking_ids = _column_values(frame, "bookingId")
    booking_nos = _column_values(frame, "operational.bookingNo")
    invoice_nos = _column_values(frame, "invoiceNo")
    if booking_ids:
        clauses.append({"_id": {"$in": booking_ids}})
    if booking_nos:
        clauses.append({"bookingNo": {"$in": booking_nos}})
    if invoice_nos:
        clauses.append({"invoiceNo": {"$in": invoice_nos}})

    query = _query_from_clauses(clauses)
    if query is None:
        return by_id, by_booking_no, by_invoice_no

    projection = {
        "_id": 1,
        "bookingNo": 1,
        "invoiceNo": 1,
        "queryFor": 1,
        "accountType": 1,
        "incoTerms": 1,
        "commodity": 1,
        "grossWeight": 1,
        "chargeableWeight": 1,
        "volumeWeight": 1,
        "mblVolumeWeight": 1,
        "noOfContainers": 1,
        "originAirport.name": 1,
        "destinationAirport.name": 1,
    }
    for doc in database[BOOKINGS_COLLECTION].find(query, projection):
        flat = flatten_dict(doc)
        _latest_record(by_id, doc.get("_id"), flat)
        _latest_record(by_booking_no, flat.get("bookingNo"), flat)
        _latest_record(by_invoice_no, flat.get("invoiceNo"), flat)
    return by_id, by_booking_no, by_invoice_no


def _build_query_lookup(
    database,
    frame: pd.DataFrame,
    available_collections: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_booking_id: dict[str, dict[str, Any]] = {}
    by_booking_no: dict[str, dict[str, Any]] = {}
    by_job_no: dict[str, dict[str, Any]] = {}
    if not _collection_present(QUERIES_COLLECTION, available_collections):
        return by_booking_id, by_booking_no, by_job_no

    clauses = []
    booking_ids = _column_values(frame, "bookingId")
    booking_nos = _column_values(frame, "operational.bookingNo")
    if booking_ids:
        clauses.append({"bookingId": {"$in": booking_ids}})
    if booking_nos:
        clauses.append({"bookingNo": {"$in": booking_nos}})

    query = _query_from_clauses(clauses)
    if query is None:
        return by_booking_id, by_booking_no, by_job_no

    projection = {
        "_id": 1,
        "bookingId": 1,
        "bookingNo": 1,
        "customerId": 1,
        "queryFor": 1,
        "jobNo": 1,
        "commodity": 1,
        "chargeableWeight": 1,
        "grossWeight": 1,
        "volumeWeight": 1,
        "noOfContainers": 1,
        "originAirport.name": 1,
        "destinationAirport.name": 1,
        "customerCurrency": 1,
        "shipmentType": 1,
        "IncoTerms": 1,
        "clearenceDate": 1,
        "blNo": 1,
    }
    for doc in database[QUERIES_COLLECTION].find(query, projection):
        flat = flatten_dict(doc)
        _latest_record(by_booking_id, flat.get("bookingId"), flat)
        _latest_record(by_booking_no, flat.get("bookingNo"), flat)
        _latest_record(by_job_no, flat.get("jobNo"), flat)
    return by_booking_id, by_booking_no, by_job_no


def _build_gatein_lookup(
    database,
    frame: pd.DataFrame,
    available_collections: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_invoice_no: dict[str, dict[str, Any]] = {}
    by_job_no: dict[str, dict[str, Any]] = {}
    if not _collection_present(GATEINS_COLLECTION, available_collections):
        return by_invoice_no, by_job_no

    clauses = []
    invoice_nos = _column_values(frame, "invoiceNo")
    job_nos = _column_values(frame, "operational.jobNo")
    if invoice_nos:
        clauses.append({"invoiceNo": {"$in": invoice_nos}})
    if job_nos:
        clauses.append({"jobNo": {"$in": job_nos}})

    query = _query_from_clauses(clauses)
    if query is None:
        return by_invoice_no, by_job_no

    projection = {
        "invoiceNo": 1,
        "jobNo": 1,
        "shippingBillNo": 1,
        "queryFor": 1,
        "grossWeight": 1,
        "volumeWeight": 1,
        "originAirport": 1,
        "destinationAirport": 1,
        "originName": 1,
        "destinationName": 1,
        "dispatch": 1,
        "approvalReceived.status": 1,
        "documentReceived.status": 1,
        "checklistPrepared.status": 1,
    }
    for doc in database[GATEINS_COLLECTION].find(query, projection):
        flat = flatten_dict(doc)
        _latest_record(by_invoice_no, flat.get("invoiceNo"), flat)
        _latest_record(by_job_no, flat.get("jobNo"), flat)
    return by_invoice_no, by_job_no


def _build_clearance_lookup(
    database,
    frame: pd.DataFrame,
    available_collections: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    by_job_no: dict[str, dict[str, Any]] = {}
    if not _collection_present(CLEARANCE_COLLECTION, available_collections):
        return by_job_no

    job_nos = _column_values(frame, "operational.jobNo")
    if not job_nos:
        return by_job_no

    projection = {
        "jobNo": 1,
        "status": 1,
        "shipmentComplete.status": 1,
        "shipmentHandover.status": 1,
        "terminalChallan.status": 1,
        "documentSuperintendent.status": 1,
        "physicalSuperintendent.status": 1,
        "destination": 1,
        "createdAt": 1,
        "updatedAt": 1,
    }
    for doc in database[CLEARANCE_COLLECTION].find({"jobNo": {"$in": job_nos}}, projection):
        flat = flatten_dict(doc)
        _latest_record(by_job_no, flat.get("jobNo"), flat)
    return by_job_no


def _apply_route_bits(frame: pd.DataFrame, idx: int, prefix: str, raw_value: Any) -> None:
    bits = _airport_bits(raw_value)
    _fill_missing(frame, idx, f"shipmentDetails.{prefix}City", bits.get("city"))
    _fill_missing(frame, idx, f"shipmentDetails.{prefix}Country", bits.get("country"))


def _normalize_inco_terms(value: Any) -> Any:
    if not isinstance(value, list):
        return value

    normalized_inco = None
    for item in value:
        if isinstance(item, dict):
            normalized_inco = _first_present(item.get("value"), item.get("label"))
        else:
            normalized_inco = item
        if not is_missing(normalized_inco):
            break
    return normalized_inco


def _assign_payment_column(frame: pd.DataFrame, column: str, new_values: list[Any]) -> None:
    if column not in frame.columns:
        frame[column] = new_values
        return

    frame[column] = [
        existing if not is_missing(existing) else new
        for existing, new in zip(frame[column], new_values)
    ]


def _enrich_with_related_collections(
    frame: pd.DataFrame,
    database,
    available_collections: set[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = _merge_customer_enrichment(frame.copy(), database, available_collections)
    bookings_by_id, bookings_by_booking_no, bookings_by_invoice_no = _build_booking_lookup(database, enriched, available_collections)
    queries_by_booking_id, queries_by_booking_no, queries_by_job_no = _build_query_lookup(database, enriched, available_collections)

    # First pass: customer, booking, and query joins.
    for idx, row in enriched.iterrows():
        booking_id_key = _normalize_key(row.get("bookingId"))
        booking_no_key = _normalize_key(row.get("operational.bookingNo"))
        invoice_no_key = _normalize_key(row.get("invoiceNo"))

        booking_record = (
            bookings_by_id.get(booking_id_key)
            or bookings_by_booking_no.get(booking_no_key)
            or bookings_by_invoice_no.get(invoice_no_key)
        )
        effective_booking_no = booking_no_key
        if effective_booking_no is None and booking_record is not None:
            effective_booking_no = _normalize_key(booking_record.get("bookingNo"))
        query_record = queries_by_booking_id.get(booking_id_key) or queries_by_booking_no.get(effective_booking_no)

        if booking_record:
            _fill_missing(enriched, idx, "operational.bookingNo", booking_record.get("bookingNo"))
            _fill_missing(enriched, idx, "shipmentDetails.queryFor", booking_record.get("queryFor"))
            _fill_missing(enriched, idx, "shipmentDetails.accountType", booking_record.get("accountType"))
            _fill_missing(enriched, idx, "shipmentDetails.incoTerms", booking_record.get("incoTerms"))
            _fill_missing(enriched, idx, "shipmentDetails.commodity", booking_record.get("commodity"))
            _fill_missing(enriched, idx, "shipmentDetails.grossWeight", booking_record.get("grossWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.chargeableWeight", booking_record.get("chargeableWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.volumeWeight", _first_present(booking_record.get("mblVolumeWeight"), booking_record.get("volumeWeight")))
            _fill_missing(enriched, idx, "shipmentDetails.noOfContainers", booking_record.get("noOfContainers"))
            _apply_route_bits(enriched, idx, "origin", booking_record.get("originAirport.name"))
            _apply_route_bits(enriched, idx, "destination", booking_record.get("destinationAirport.name"))

        if query_record:
            _fill_missing(enriched, idx, "operational.bookingNo", query_record.get("bookingNo"))
            _fill_missing(enriched, idx, "operational.jobNo", query_record.get("jobNo"))
            _fill_missing(enriched, idx, "shipmentDetails.queryFor", query_record.get("queryFor"))
            _fill_missing(enriched, idx, "shipmentDetails.accountType", query_record.get("shipmentType"))
            _fill_missing(enriched, idx, "shipmentDetails.incoTerms", _normalize_inco_terms(query_record.get("IncoTerms")))
            _fill_missing(enriched, idx, "shipmentDetails.commodity", query_record.get("commodity"))
            _fill_missing(enriched, idx, "shipmentDetails.grossWeight", query_record.get("grossWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.chargeableWeight", query_record.get("chargeableWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.volumeWeight", query_record.get("volumeWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.noOfContainers", query_record.get("noOfContainers"))
            _fill_missing(enriched, idx, "selectedCustomerCurrency", query_record.get("customerCurrency"))
            _apply_route_bits(enriched, idx, "origin", query_record.get("originAirport.name"))
            _apply_route_bits(enriched, idx, "destination", query_record.get("destinationAirport.name"))

    # Second pass: gateins, clearances, and job-level query fallback.
    gateins_by_invoice_no, gateins_by_job_no = _build_gatein_lookup(database, enriched, available_collections)
    clearances_by_job_no = _build_clearance_lookup(database, enriched, available_collections)
    for idx, row in enriched.iterrows():
        invoice_no_key = _normalize_key(row.get("invoiceNo"))
        job_no_key = _normalize_key(row.get("operational.jobNo"))
        gatein_record = gateins_by_invoice_no.get(invoice_no_key) or gateins_by_job_no.get(job_no_key)
        if gatein_record:
            _fill_missing(enriched, idx, "operational.jobNo", gatein_record.get("jobNo"))
            _fill_missing(enriched, idx, "operational.shippingBillNo", gatein_record.get("shippingBillNo"))
            _fill_missing(enriched, idx, "shipmentDetails.queryFor", gatein_record.get("queryFor"))
            _fill_missing(enriched, idx, "shipmentDetails.grossWeight", gatein_record.get("grossWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.volumeWeight", gatein_record.get("volumeWeight"))
            _fill_missing(
                enriched,
                idx,
                "operational.gateInStatus",
                _first_present(
                    gatein_record.get("dispatch"),
                    gatein_record.get("approvalReceived.status"),
                    gatein_record.get("documentReceived.status"),
                    gatein_record.get("checklistPrepared.status"),
                ),
            )
            _apply_route_bits(enriched, idx, "origin", _first_present(gatein_record.get("originAirport"), gatein_record.get("originName")))
            _apply_route_bits(enriched, idx, "destination", _first_present(gatein_record.get("destinationAirport"), gatein_record.get("destinationName")))

        effective_job_no = _normalize_key(_column_value(enriched, idx, "operational.jobNo"))
        query_record = queries_by_job_no.get(effective_job_no)
        if query_record:
            _fill_missing(enriched, idx, "shipmentDetails.queryFor", query_record.get("queryFor"))
            _fill_missing(enriched, idx, "shipmentDetails.accountType", query_record.get("shipmentType"))
            _fill_missing(enriched, idx, "shipmentDetails.commodity", query_record.get("commodity"))
            _fill_missing(enriched, idx, "shipmentDetails.grossWeight", query_record.get("grossWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.chargeableWeight", query_record.get("chargeableWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.volumeWeight", query_record.get("volumeWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.noOfContainers", query_record.get("noOfContainers"))
            _fill_missing(enriched, idx, "selectedCustomerCurrency", query_record.get("customerCurrency"))
            _apply_route_bits(enriched, idx, "origin", query_record.get("originAirport.name"))
            _apply_route_bits(enriched, idx, "destination", query_record.get("destinationAirport.name"))

        clearance_record = clearances_by_job_no.get(effective_job_no)
        if clearance_record:
            _fill_missing(
                enriched,
                idx,
                "operational.clearanceStatus",
                _first_present(
                    clearance_record.get("status"),
                    clearance_record.get("shipmentComplete.status"),
                    clearance_record.get("shipmentHandover.status"),
                    clearance_record.get("terminalChallan.status"),
                    clearance_record.get("documentSuperintendent.status"),
                    clearance_record.get("physicalSuperintendent.status"),
                ),
            )
            _fill_missing(
                enriched,
                idx,
                "operational.lastTrackingStatus",
                _first_present(
                    clearance_record.get("shipmentComplete.status"),
                    clearance_record.get("shipmentHandover.status"),
                    clearance_record.get("status"),
                ),
            )
            _fill_missing(enriched, idx, "operational.lastTrackingLocation", clearance_record.get("destination"))
    return enriched


def join_payment_transactions(
    frame: pd.DataFrame,
    *,
    payment_rows: list[dict] | None = None,
    db=None,
    available_collections: set[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    prepared = frame.copy()
    rows_by_invoice_id: dict[str, list[int]] = defaultdict(list)
    rows_by_invoice_no: dict[str, list[int]] = defaultdict(list)
    rows_by_invoice_ref: dict[str, list[int]] = defaultdict(list)
    invoice_ids: list[Any] = []
    invoice_nos: list[Any] = []
    invoice_refs: list[Any] = []

    for idx, row in prepared.iterrows():
        invoice_id = row.get("_id")
        invoice_no = row.get("invoiceNo")
        invoice_ref = row.get("legacy.invoice_ref_raw")
        normalized_id = _normalize_key(invoice_id)
        normalized_no = _normalize_key(invoice_no)
        normalized_ref = _normalize_key(invoice_ref)
        if normalized_id is not None:
            rows_by_invoice_id[normalized_id].append(idx)
            invoice_ids.append(invoice_id)
        if normalized_no is not None:
            rows_by_invoice_no[normalized_no].append(idx)
            invoice_nos.append(invoice_no)
        if normalized_ref is not None:
            rows_by_invoice_ref[normalized_ref].append(idx)
            invoice_refs.append(invoice_ref)

    if payment_rows is None:
        database = db if db is not None else get_live_database()
        collection_names = available_collections or set(database.list_collection_names())
        if PAYMENT_COLLECTION not in collection_names:
            return prepared

        clauses = []
        if invoice_ids:
            clauses.append({"performaInvoiceId": {"$in": invoice_ids}})
        if invoice_nos:
            clauses.append({"invoiceNo": {"$in": invoice_nos}})
        if invoice_refs:
            clauses.append({"finalInvoiceId": {"$in": invoice_refs}})

        query = _query_from_clauses(clauses)
        if query is None:
            return prepared
        payment_rows = list(database[PAYMENT_COLLECTION].find(query))

    payments_by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[int, str]] = set()
    for raw_row in payment_rows or []:
        flat = flatten_dict(raw_row)
        candidate_maps = (
            (raw_row.get("performaInvoiceId", flat.get("performaInvoiceId")), rows_by_invoice_id),
            (raw_row.get("invoiceNo", flat.get("invoiceNo")), rows_by_invoice_no),
            (raw_row.get("finalInvoiceId", flat.get("finalInvoiceId")), rows_by_invoice_ref),
        )
        row_token = _normalize_key(raw_row.get("_id", flat.get("_id", flat.get("transactionId"))))
        if row_token is None:
            row_token = json.dumps(flat, sort_keys=True, default=str)

        for candidate, mapping in candidate_maps:
            join_key = _normalize_key(candidate)
            if join_key is None:
                continue
            for idx in mapping.get(join_key, []):
                pair = (idx, row_token)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                payments_by_index[idx].append(flat)

    payment_dates: list[Any] = []
    payment_raw_values: list[Any] = []
    installment_counts: list[int] = []
    partial_flags: list[bool] = []
    payment_details: list[Any] = []

    for idx in prepared.index:
        rows = [row for row in payments_by_index.get(idx, []) if not _payment_is_rejected(row)]
        parsed_dates = sorted([date for date in (_payment_date(row) for row in rows) if pd.notna(date)])
        latest_payment = parsed_dates[-1] if parsed_dates else pd.NaT
        raw_details = [value for value in (_payment_raw(row) for row in rows) if value]
        payment_dates.append(latest_payment)
        payment_raw_values.append(raw_details[-1] if raw_details else None)
        installment_counts.append(len(rows))
        partial_flags.append(len(rows) > 1)
        payment_details.append(json.dumps(raw_details) if raw_details else None)

    _assign_payment_column(prepared, "paymentDate", payment_dates)
    _assign_payment_column(prepared, "paymentDate_raw", payment_raw_values)
    _assign_payment_column(prepared, "paymentInstallmentCount", installment_counts)
    _assign_payment_column(prepared, "partialPaymentFlag", partial_flags)
    _assign_payment_column(prepared, "paymentDetailsRaw", payment_details)
    return prepared


# Main fetch entry point for the pipeline.
# The old Risk.Main collection path is removed. We now build the same normalized
# invoice frame directly from the live collections and keep the downstream
# canonicalization and feature pipeline unchanged.
def fetch_risk_main_frame(query: dict | None = None, limit: int | None = None) -> pd.DataFrame:
    database = get_live_database()
    available_collections = set(database.list_collection_names())
    invoice_collection = database[get_live_invoice_collection()]
    cursor = (
        invoice_collection.find(query or {}, _build_live_projection())
        .sort([("invoiceDate", -1), ("_id", -1)])
        .batch_size(LIVE_FETCH_BATCH_SIZE)
    )
    if limit is not None:
        cursor = cursor.limit(int(limit))

    rows = [_normalize_live_doc(doc) for doc in cursor]
    frame = pd.DataFrame(rows)
    frame = _enrich_with_related_collections(frame, database, available_collections)
    frame = join_payment_transactions(frame, db=database, available_collections=available_collections)

    coverage = compute_live_coverage(frame)
    _store_live_diagnostics(coverage=coverage, collections=sorted(available_collections))
    logger.info(
        "Fetched live invoice frame rows=%d invoice_key_pct=%.2f customer_id_pct=%.2f admin_operational_pct=%.2f",
        len(frame),
        coverage.get("invoice_key_pct", 0.0),
        coverage.get("customer_id_pct", 0.0),
        coverage.get("admin_operational_pct", 0.0),
    )
    return frame


def inspect_risk_main_indexes() -> dict:
    database = get_live_database()
    available_collections = sorted(database.list_collection_names())
    return {
        "mode": "live_collections",
        "database": get_live_db_name() or database.name,
        "collection": get_live_invoice_collection(),
        "legacy_database_name": PRODUCTION_RISK_DB_NAME,
        "legacy_collection_name": PRODUCTION_RISK_COLLECTION,
        "available_collections": available_collections,
        "projection_fields": sorted(_build_live_projection().keys()),
    }


def canonicalize_risk_main_frame(
    df: pd.DataFrame,
    target_delay_days: int = TARGET_DELAY_DAYS,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    as_of_date = as_of_date or pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).normalize()
    canonical = pd.DataFrame(index=df.index)
    for normalized_path, canonical_name in NORMALIZED_TO_CANONICAL_FIELD_MAP.items():
        canonical[canonical_name] = _series_from_frame(df, normalized_path)

    canonical["invoice_key"] = canonical["invoice_key"].fillna(_series_from_frame(df, "BillNo"))
    canonical["invoice_date"] = canonical["invoice_date"].apply(parse_main_date)
    canonical["due_date"] = canonical["due_date"].apply(parse_main_date)
    canonical["payment_date"] = canonical["payment_date"].apply(parse_main_date)
    canonical["execution_date"] = _series_from_frame(canonical, "execution_date").apply(parse_main_date)
    canonical["customer_onboard_date"] = _series_from_frame(canonical, "customer_onboard_date").apply(parse_main_date)

    for column in [
        "invoice_amount",
        "gross_amount",
        "paid_amount",
        "tds_amount",
        "ytd_exposure",
        "aging_not_due",
        "aging_0_15",
        "aging_16_30",
        "aging_31_45",
        "aging_46_60",
        "aging_60_90",
        "aging_above_90",
        "gross_weight",
        "chargeable_weight",
        "volume_weight",
        "container_count",
        "payment_installment_count",
    ]:
        canonical[column] = pd.to_numeric(canonical.get(column), errors="coerce")
    canonical["terms_days"] = canonical.get("terms_days", pd.Series(index=canonical.index)).map(_coerce_terms_days)

    partial_payment = canonical.get("partial_payment_flag")
    if partial_payment is None:
        partial_payment = pd.Series(False, index=canonical.index, dtype="boolean")
    else:
        partial_payment = pd.Series(partial_payment, index=canonical.index, dtype="boolean")
    canonical["partial_payment_flag"] = partial_payment.fillna(False).astype(bool)
    canonical["paid_status"] = canonical.get("paid_status", "Pending").fillna("Pending").astype(str)
    canonical["source_system"] = "risk_main"
    canonical["company"] = _series_from_frame(df, "companyCode").fillna("").astype(str)
    canonical["payment_date_raw"] = _series_from_frame(df, "paymentDate_raw")
    canonical["payment_details_raw"] = _series_from_frame(df, "paymentDetailsRaw")
    canonical["invoice_ref_raw"] = _series_from_frame(df, "legacy.invoice_ref_raw")

    safe_terms_days = canonical["terms_days"].where(
        canonical["terms_days"].between(MIN_VALID_TERMS_DAYS, MAX_VALID_TERMS_DAYS),
        other=pd.NA,
    )
    due_fallback = canonical["invoice_date"] + pd.to_timedelta(safe_terms_days, unit="D", errors="coerce")
    canonical["due_date"] = canonical["due_date"].fillna(due_fallback)
    actual_delay = (canonical["payment_date"] - canonical["due_date"]).dt.days
    inferred_delay = (as_of_date - canonical["due_date"]).dt.days
    canonical["delay_days"] = actual_delay.where(canonical["payment_date"].notna(), inferred_delay).fillna(0).clip(lower=0)
    canonical["target"] = (canonical["delay_days"] > int(target_delay_days)).astype(int)
    canonical["label_quality"] = canonical["payment_date"].notna().map({True: "actual", False: "inferred"})

    canonical["execution_gap_days"] = (canonical["invoice_date"] - canonical["execution_date"]).dt.days
    canonical["days_to_due"] = (canonical["due_date"] - canonical["invoice_date"]).dt.days.fillna(canonical["terms_days"])
    canonical["customer_age_days"] = (canonical["invoice_date"] - canonical["customer_onboard_date"]).dt.days
    canonical["weight_discrepancy"] = (canonical["chargeable_weight"] - canonical["gross_weight"]).abs()
    canonical["aging_total"] = canonical[
        [
            "aging_not_due",
            "aging_0_15",
            "aging_16_30",
            "aging_31_45",
            "aging_46_60",
            "aging_60_90",
            "aging_above_90",
        ]
    ].fillna(0.0).sum(axis=1)
    canonical["unpaid_amount"] = (canonical["invoice_amount"].fillna(0.0) - canonical["paid_amount"].fillna(0.0)).clip(lower=0.0)
    canonical["gross_to_invoice_ratio"] = [
        safe_ratio(gross, amount) for gross, amount in zip(canonical["gross_amount"], canonical["invoice_amount"])
    ]
    canonical["paid_to_invoice_ratio"] = [
        safe_ratio(paid, amount) for paid, amount in zip(canonical["paid_amount"], canonical["invoice_amount"])
    ]
    canonical["tds_to_invoice_ratio"] = [
        safe_ratio(tds, amount) for tds, amount in zip(canonical["tds_amount"], canonical["invoice_amount"])
    ]
    canonical["exposure_to_invoice_ratio"] = [
        safe_ratio(exposure, amount) for exposure, amount in zip(canonical["ytd_exposure"], canonical["invoice_amount"])
    ]
    canonical["aging_total_to_invoice_ratio"] = [
        safe_ratio(age_total, amount) for age_total, amount in zip(canonical["aging_total"], canonical["invoice_amount"])
    ]
    canonical["customer_key"] = canonical["customer_key"].fillna(canonical["customer_name"]).astype(str)
    return canonical


def load_canonical_risk_main_dataset(limit: int | None = None) -> pd.DataFrame:
    raw = fetch_risk_main_frame(limit=limit)
    return canonicalize_risk_main_frame(raw)


def _normalize_join_value(value: Any) -> str | None:
    if is_missing(value):
        return None
    return str(value)


def _sample_documents(collection, sample_size: int) -> list[dict]:
    return list(collection.find({}).limit(sample_size))


def _profile_documents(documents: Iterable[dict], *, sample_size: int) -> dict[str, Any]:
    docs = list(documents)
    field_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"non_missing": 0, "types": set(), "samples": []})
    total = len(docs)
    for raw_doc in docs:
        flat = flatten_dict(raw_doc)
        for path, value in flat.items():
            stats = field_stats[path]
            if not is_missing(value):
                stats["non_missing"] += 1
                stats["types"].add(type(value).__name__)
                if len(stats["samples"]) < 3:
                    stats["samples"].append(json_safe(value))
            elif len(stats["samples"]) < 3:
                stats["samples"].append(None)

    return {
        "sample_size": int(min(total, sample_size)),
        "fields": {
            path: {
                "types": sorted(stats["types"]),
                "non_missing_rate": round(safe_ratio(stats["non_missing"], total) * 100.0, 2),
                "samples": stats["samples"],
            }
            for path, stats in sorted(field_stats.items())
        },
    }


def profile_collection(collection_name: str, *, sample_size: int = 500, db=None) -> dict[str, Any]:
    database = db if db is not None else get_live_database()
    collection = database[collection_name]
    documents = _sample_documents(collection, sample_size=sample_size)
    profile = _profile_documents(documents, sample_size=sample_size)
    profile["collection"] = collection_name
    profile["database"] = database.name
    profile["document_count_profiled"] = len(documents)
    return profile


def find_join_keys(
    *,
    invoice_collection: str = "invoicemasters",
    payment_collection: str = PAYMENT_COLLECTION,
    sample_size: int = 500,
    db=None,
) -> dict[str, Any]:
    database = db if db is not None else get_live_database()
    invoice_docs = _sample_documents(database[invoice_collection], sample_size=sample_size)
    payment_docs = _sample_documents(database[payment_collection], sample_size=sample_size)

    invoice_flats = [flatten_dict(doc) for doc in invoice_docs]
    payment_flats = [flatten_dict(doc) for doc in payment_docs]
    invoice_candidates = ("_id", "invoiceNo", "legacy.invoice_ref_raw")
    payment_candidates = ("performaInvoiceId", "finalInvoiceId", "invoiceId", "invoiceNo")

    comparisons: list[dict[str, Any]] = []
    for invoice_field in invoice_candidates:
        invoice_values = {
            normalized
            for flat in invoice_flats
            if (normalized := _normalize_join_value(flat.get(invoice_field))) is not None
        }
        for payment_field in payment_candidates:
            payment_values = [
                normalized
                for flat in payment_flats
                if (normalized := _normalize_join_value(flat.get(payment_field))) is not None
            ]
            match_count = sum(1 for value in payment_values if value in invoice_values)
            comparisons.append(
                {
                    "invoice_field": invoice_field,
                    "payment_field": payment_field,
                    "payment_rows": len(payment_values),
                    "match_pct": round(safe_ratio(match_count, len(payment_values)) * 100.0, 2),
                }
            )

    best = max(comparisons, key=lambda item: item["match_pct"]) if comparisons else None
    return {
        "invoice_collection": invoice_collection,
        "payment_collection": payment_collection,
        "best_match": best,
        "comparisons": comparisons,
    }


def build_live_profile(
    *,
    sample_size: int = 500,
    collection_names: list[str] | None = None,
    db=None,
) -> dict[str, Any]:
    database = db if db is not None else get_live_database()
    names = collection_names or sorted(database.list_collection_names())
    return {
        "database": get_live_db_name() or database.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "collections": {name: profile_collection(name, sample_size=sample_size, db=database) for name in names},
        "join_keys": find_join_keys(sample_size=sample_size, db=database),
    }


def profile_live_database(
    *,
    sample_size: int = 500,
    collection_names: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    payload = build_live_profile(sample_size=sample_size, collection_names=collection_names or list(LIVE_PROFILE_COLLECTIONS))
    target_dir = output_dir or DEFAULT_PROFILE_OUTPUT_DIR
    write_json(target_dir / "live_schema_inventory.json", payload)
    logger.info("Live schema inventory written to %s", target_dir)
    return payload
