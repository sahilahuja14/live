from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

import pandas as pd

from ..config import get_live_invoice_collection
from ..dbconnect import get_live_database
from ..logging_config import get_logger
from .live_field_map import LIVE_PASSTHROUGH_FIELDS, LIVE_TO_NORMALIZED
from .parsing import parse_main_date, safe_text
from .utils import flatten_dict, is_missing


logger = get_logger(__name__)
LIVE_FETCH_BATCH_SIZE = max(int(os.getenv("API_RISK_MAIN_FETCH_BATCH_SIZE", "1000")), 1)
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


def _projection_from_live_map() -> dict[str, int]:
    raw_paths = sorted(
        set(LIVE_TO_NORMALIZED) | set(LIVE_PASSTHROUGH_FIELDS),
        key=lambda value: (value.count("."), value),
    )
    selected: list[str] = []
    for path in raw_paths:
        if any(path == parent or path.startswith(f"{parent}.") for parent in selected):
            continue
        selected = [existing for existing in selected if not existing.startswith(f"{path}.")]
        selected.append(path)

    projection = {"_id": 1}
    for key in selected:
        projection[key] = 1
    return projection


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


def _coverage_ratio(frame: pd.DataFrame, columns: list[str], *, require_all: bool = False) -> float:
    rows = len(frame)
    if rows == 0:
        return 0.0
    if require_all:
        available = pd.Series(True, index=frame.index)
        for column in columns:
            if column not in frame.columns:
                available &= False
            else:
                available &= frame[column].map(lambda value: not is_missing(value))
        return round(float(available.mean() * 100.0), 2)

    available = pd.Series(False, index=frame.index)
    for column in columns:
        if column in frame.columns:
            available |= frame[column].map(lambda value: not is_missing(value))
    return round(float(available.mean() * 100.0), 2)


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
        value = flat.get(field)
        parsed = parse_main_date(value)
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def _payment_raw(flat: dict[str, Any]) -> str | None:
    for field in _PAYMENT_RAW_FIELDS:
        value = safe_text(flat.get(field))
        if value:
            return value
    return None


def _merge_customer_enrichment(frame: pd.DataFrame, database) -> pd.DataFrame:
    if frame.empty or CUSTOMER_COLLECTION not in database.list_collection_names():
        return frame
    if "customer.customerId" not in frame.columns:
        return frame

    customer_ids = [value for value in frame["customer.customerId"].tolist() if not is_missing(value)]
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
        _fill_missing(
            enriched,
            idx,
            "selectedCustomerCurrency",
            _first_present(record.get("custCurrency"), record.get("customerCurrency")),
        )
    return enriched


def _build_booking_lookup(database, frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_booking_no: dict[str, dict[str, Any]] = {}
    by_invoice_no: dict[str, dict[str, Any]] = {}
    if BOOKINGS_COLLECTION not in database.list_collection_names():
        return by_id, by_booking_no, by_invoice_no

    booking_ids = [value for value in frame.get("bookingId", pd.Series(dtype=object)).tolist() if not is_missing(value)]
    booking_nos = [
        value
        for value in frame.get("operational.bookingNo", pd.Series(dtype=object)).tolist()
        if not is_missing(value)
    ]
    invoice_nos = [value for value in frame.get("invoiceNo", pd.Series(dtype=object)).tolist() if not is_missing(value)]
    clauses = []
    if booking_ids:
        clauses.append({"_id": {"$in": booking_ids}})
    if booking_nos:
        clauses.append({"bookingNo": {"$in": booking_nos}})
    if invoice_nos:
        clauses.append({"invoiceNo": {"$in": invoice_nos}})
    if not clauses:
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
    query = clauses[0] if len(clauses) == 1 else {"$or": clauses}
    for doc in database[BOOKINGS_COLLECTION].find(query, projection):
        flat = flatten_dict(doc)
        _latest_record(by_id, doc.get("_id"), flat)
        _latest_record(by_booking_no, flat.get("bookingNo"), flat)
        _latest_record(by_invoice_no, flat.get("invoiceNo"), flat)
    return by_id, by_booking_no, by_invoice_no


def _build_query_lookup(database, frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_booking_id: dict[str, dict[str, Any]] = {}
    by_booking_no: dict[str, dict[str, Any]] = {}
    by_job_no: dict[str, dict[str, Any]] = {}
    if QUERIES_COLLECTION not in database.list_collection_names():
        return by_booking_id, by_booking_no, by_job_no

    booking_ids = [value for value in frame.get("bookingId", pd.Series(dtype=object)).tolist() if not is_missing(value)]
    booking_nos = [
        value
        for value in frame.get("operational.bookingNo", pd.Series(dtype=object)).tolist()
        if not is_missing(value)
    ]
    clauses = []
    if booking_ids:
        clauses.append({"bookingId": {"$in": booking_ids}})
    if booking_nos:
        clauses.append({"bookingNo": {"$in": booking_nos}})
    if not clauses:
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
    query = clauses[0] if len(clauses) == 1 else {"$or": clauses}
    for doc in database[QUERIES_COLLECTION].find(query, projection):
        flat = flatten_dict(doc)
        _latest_record(by_booking_id, flat.get("bookingId"), flat)
        _latest_record(by_booking_no, flat.get("bookingNo"), flat)
        _latest_record(by_job_no, flat.get("jobNo"), flat)
    return by_booking_id, by_booking_no, by_job_no


def _build_gatein_lookup(database, frame: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_invoice_no: dict[str, dict[str, Any]] = {}
    by_job_no: dict[str, dict[str, Any]] = {}
    if GATEINS_COLLECTION not in database.list_collection_names():
        return by_invoice_no, by_job_no

    invoice_nos = [value for value in frame.get("invoiceNo", pd.Series(dtype=object)).tolist() if not is_missing(value)]
    job_nos = [
        value
        for value in frame.get("operational.jobNo", pd.Series(dtype=object)).tolist()
        if not is_missing(value)
    ]
    clauses = []
    if invoice_nos:
        clauses.append({"invoiceNo": {"$in": invoice_nos}})
    if job_nos:
        clauses.append({"jobNo": {"$in": job_nos}})
    if not clauses:
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
    query = clauses[0] if len(clauses) == 1 else {"$or": clauses}
    for doc in database[GATEINS_COLLECTION].find(query, projection):
        flat = flatten_dict(doc)
        _latest_record(by_invoice_no, flat.get("invoiceNo"), flat)
        _latest_record(by_job_no, flat.get("jobNo"), flat)
    return by_invoice_no, by_job_no


def _build_clearance_lookup(database, frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    by_job_no: dict[str, dict[str, Any]] = {}
    if CLEARANCE_COLLECTION not in database.list_collection_names():
        return by_job_no

    job_nos = [
        value
        for value in frame.get("operational.jobNo", pd.Series(dtype=object)).tolist()
        if not is_missing(value)
    ]
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


def _enrich_with_related_collections(frame: pd.DataFrame, database) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = _merge_customer_enrichment(frame.copy(), database)
    bookings_by_id, bookings_by_booking_no, bookings_by_invoice_no = _build_booking_lookup(database, enriched)
    queries_by_booking_id, queries_by_booking_no, queries_by_job_no = _build_query_lookup(database, enriched)

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
        query_record = (
            queries_by_booking_id.get(booking_id_key)
            or queries_by_booking_no.get(effective_booking_no)
        )

        if booking_record:
            _fill_missing(enriched, idx, "operational.bookingNo", booking_record.get("bookingNo"))
            _fill_missing(enriched, idx, "shipmentDetails.queryFor", booking_record.get("queryFor"))
            _fill_missing(enriched, idx, "shipmentDetails.accountType", booking_record.get("accountType"))
            _fill_missing(enriched, idx, "shipmentDetails.incoTerms", booking_record.get("incoTerms"))
            _fill_missing(enriched, idx, "shipmentDetails.commodity", booking_record.get("commodity"))
            _fill_missing(enriched, idx, "shipmentDetails.grossWeight", booking_record.get("grossWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.chargeableWeight", booking_record.get("chargeableWeight"))
            _fill_missing(
                enriched,
                idx,
                "shipmentDetails.volumeWeight",
                _first_present(booking_record.get("mblVolumeWeight"), booking_record.get("volumeWeight")),
            )
            _fill_missing(enriched, idx, "shipmentDetails.noOfContainers", booking_record.get("noOfContainers"))
            _apply_route_bits(enriched, idx, "origin", booking_record.get("originAirport.name"))
            _apply_route_bits(enriched, idx, "destination", booking_record.get("destinationAirport.name"))

        if query_record:
            inco_terms = query_record.get("IncoTerms")
            if isinstance(inco_terms, list):
                normalized_inco = None
                for item in inco_terms:
                    if isinstance(item, dict):
                        normalized_inco = _first_present(item.get("value"), item.get("label"))
                    else:
                        normalized_inco = item
                    if not is_missing(normalized_inco):
                        break
                inco_terms = normalized_inco

            _fill_missing(enriched, idx, "operational.bookingNo", query_record.get("bookingNo"))
            _fill_missing(enriched, idx, "operational.jobNo", query_record.get("jobNo"))
            _fill_missing(enriched, idx, "shipmentDetails.queryFor", query_record.get("queryFor"))
            _fill_missing(enriched, idx, "shipmentDetails.accountType", query_record.get("shipmentType"))
            _fill_missing(enriched, idx, "shipmentDetails.incoTerms", inco_terms)
            _fill_missing(enriched, idx, "shipmentDetails.commodity", query_record.get("commodity"))
            _fill_missing(enriched, idx, "shipmentDetails.grossWeight", query_record.get("grossWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.chargeableWeight", query_record.get("chargeableWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.volumeWeight", query_record.get("volumeWeight"))
            _fill_missing(enriched, idx, "shipmentDetails.noOfContainers", query_record.get("noOfContainers"))
            _fill_missing(enriched, idx, "selectedCustomerCurrency", query_record.get("customerCurrency"))
            _apply_route_bits(enriched, idx, "origin", query_record.get("originAirport.name"))
            _apply_route_bits(enriched, idx, "destination", query_record.get("destinationAirport.name"))

    gateins_by_invoice_no, gateins_by_job_no = _build_gatein_lookup(database, enriched)
    clearances_by_job_no = _build_clearance_lookup(database, enriched)

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
            _apply_route_bits(
                enriched,
                idx,
                "origin",
                _first_present(gatein_record.get("originAirport"), gatein_record.get("originName")),
            )
            _apply_route_bits(
                enriched,
                idx,
                "destination",
                _first_present(gatein_record.get("destinationAirport"), gatein_record.get("destinationName")),
            )

        effective_job_no = _normalize_key(enriched.get("operational.jobNo", pd.Series(index=enriched.index)).get(idx))
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
        collection_names = set(database.list_collection_names())
        if PAYMENT_COLLECTION not in collection_names:
            return prepared
        clauses = []
        if invoice_ids:
            clauses.append({"performaInvoiceId": {"$in": invoice_ids}})
        if invoice_nos:
            clauses.append({"invoiceNo": {"$in": invoice_nos}})
        if invoice_refs:
            clauses.append({"finalInvoiceId": {"$in": invoice_refs}})
        if not clauses:
            return prepared
        query = clauses[0] if len(clauses) == 1 else {"$or": clauses}
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
        row_token = _normalize_key(raw_row.get("_id", flat.get("_id", flat.get("transactionId")))) or json.dumps(flat, sort_keys=True, default=str)
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
        parsed_dates = [date for date in (_payment_date(row) for row in rows) if pd.notna(date)]
        parsed_dates = sorted(parsed_dates)
        latest_payment = parsed_dates[-1] if parsed_dates else pd.NaT
        raw_details = [value for value in (_payment_raw(row) for row in rows) if value]
        payment_dates.append(latest_payment)
        payment_raw_values.append(raw_details[-1] if raw_details else None)
        installment_counts.append(len(rows))
        partial_flags.append(len(rows) > 1)
        payment_details.append(json.dumps(raw_details) if raw_details else None)

    if "paymentDate" not in prepared.columns:
        prepared["paymentDate"] = payment_dates
    else:
        prepared["paymentDate"] = [
            existing if not is_missing(existing) else new
            for existing, new in zip(prepared["paymentDate"], payment_dates)
        ]
    if "paymentDate_raw" not in prepared.columns:
        prepared["paymentDate_raw"] = payment_raw_values
    else:
        prepared["paymentDate_raw"] = [
            existing if not is_missing(existing) else new
            for existing, new in zip(prepared["paymentDate_raw"], payment_raw_values)
        ]
    if "paymentInstallmentCount" not in prepared.columns:
        prepared["paymentInstallmentCount"] = installment_counts
    else:
        prepared["paymentInstallmentCount"] = [
            int(existing) if not is_missing(existing) else new
            for existing, new in zip(prepared["paymentInstallmentCount"], installment_counts)
        ]
    if "partialPaymentFlag" not in prepared.columns:
        prepared["partialPaymentFlag"] = partial_flags
    else:
        prepared["partialPaymentFlag"] = [
            bool(existing) if not is_missing(existing) else new
            for existing, new in zip(prepared["partialPaymentFlag"], partial_flags)
        ]
    if "paymentDetailsRaw" not in prepared.columns:
        prepared["paymentDetailsRaw"] = payment_details
    else:
        prepared["paymentDetailsRaw"] = [
            existing if not is_missing(existing) else new
            for existing, new in zip(prepared["paymentDetailsRaw"], payment_details)
        ]
    return prepared


def fetch_live_frame(query: dict | None = None, limit: int | None = None) -> pd.DataFrame:
    database = get_live_database()
    invoice_collection = database[get_live_invoice_collection()]
    cursor = (
        invoice_collection.find(query or {}, _projection_from_live_map())
        .sort([("invoiceDate", -1), ("_id", -1)])
        .batch_size(LIVE_FETCH_BATCH_SIZE)
    )
    if limit is not None:
        cursor = cursor.limit(int(limit))

    rows = [_normalize_live_doc(doc) for doc in cursor]
    frame = pd.DataFrame(rows)
    frame = _enrich_with_related_collections(frame, database)
    frame = join_payment_transactions(frame, db=database)

    collections = sorted(database.list_collection_names())
    coverage = compute_live_coverage(frame)
    _store_live_diagnostics(coverage=coverage, collections=collections)
    logger.info(
        "Fetched live invoice frame rows=%d invoice_key_pct=%.2f customer_id_pct=%.2f admin_operational_pct=%.2f",
        len(frame),
        coverage.get("invoice_key_pct", 0.0),
        coverage.get("customer_id_pct", 0.0),
        coverage.get("admin_operational_pct", 0.0),
    )
    return frame
