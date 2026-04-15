from __future__ import annotations

from time import time

import pandas as pd

from ..features.customer_aggregates import CUSTOMER_AGGREGATE_DEFAULTS
from ..features.engineering import build_risk_main_feature_frame
from .parsing import parse_main_date, parse_payment_value
from .risk_canonical import canonicalize_risk_main_frame, fetch_risk_main_frame
from .risk_map import PRODUCTION_RISK_REQUEST_FIELD_MAP
from .utils import flatten_dict as flatten_payload


_CURRENT_ROW_COL = "_scoring_row_flag"
_CURRENT_ORDER_COL = "_scoring_row_order"
_NULL_TEXT_MARKERS = {"", "nan", "none", "<na>", "nat"}


def fetch_production_risk_main_dataset(query: dict | None = None, limit: int | None = None) -> pd.DataFrame:
    return fetch_risk_main_frame(query=query, limit=limit)


def frame_invoice_keys(df: pd.DataFrame) -> pd.Series:
    if "invoice_key" in df.columns:
        keys = df["invoice_key"]
    elif "invoiceNo" in df.columns:
        keys = df["invoiceNo"]
    elif "_id" in df.columns:
        keys = df["_id"]
    else:
        keys = pd.Series([f"row_{idx}" for idx in range(len(df))], index=df.index, dtype=object)
    return keys.fillna("").astype(str)


def build_risk_main_scoring_frame(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if target_df is None or target_df.empty:
        return pd.DataFrame()

    current = target_df.copy().reset_index(drop=True)
    current["invoiceNo"] = frame_invoice_keys(current)
    current[_CURRENT_ROW_COL] = 1
    current[_CURRENT_ORDER_COL] = range(len(current))
    current_keys = set(frame_invoice_keys(current))

    frames = []
    if history_df is not None and not history_df.empty:
        history = history_df.copy()
        history["invoiceNo"] = frame_invoice_keys(history)
        history = history[~frame_invoice_keys(history).isin(current_keys)].copy()
        if not history.empty:
            history[_CURRENT_ROW_COL] = 0
            history[_CURRENT_ORDER_COL] = -1
            frames.append(history)

    frames.append(current)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    canonical = canonicalize_risk_main_frame(combined)
    canonical[_CURRENT_ROW_COL] = combined[_CURRENT_ROW_COL].to_numpy()
    canonical[_CURRENT_ORDER_COL] = combined[_CURRENT_ORDER_COL].to_numpy()
    feature_frame = build_risk_main_feature_frame(canonical)
    current_features = feature_frame[feature_frame[_CURRENT_ROW_COL] == 1].copy()
    current_features = current_features.sort_values(_CURRENT_ORDER_COL).reset_index(drop=True)
    return current_features.drop(columns=[_CURRENT_ROW_COL, _CURRENT_ORDER_COL], errors="ignore")


CUSTOMER_ID_QUERY_FIELDS = (
    "customer.customerId",
    "customer.id",
    "customerId",
)


def _clean_customer_ids(customer_ids: list[str]) -> list[str]:
    output: set[str] = set()
    for value in customer_ids:
        cleaned = str(value).strip()
        if not cleaned or cleaned.lower() in _NULL_TEXT_MARKERS:
            continue
        output.add(cleaned)
    return sorted(output)


def _build_customer_history_query(customer_ids: list[str]) -> dict | None:
    normalized_ids = _clean_customer_ids(customer_ids)
    if not normalized_ids:
        return None

    return {
        "$or": [
            {field: {"$in": normalized_ids}}
            for field in CUSTOMER_ID_QUERY_FIELDS
        ]
    }


def fetch_production_risk_main_customer_history(customer_ids: list[str]) -> pd.DataFrame:
    query = _build_customer_history_query(customer_ids)
    if query is None:
        return pd.DataFrame()
    return fetch_risk_main_frame(query=query)


def fetch_production_risk_main_customer_aggregates(customer_ids: list[str]) -> pd.DataFrame:
    history = fetch_production_risk_main_customer_history(customer_ids)
    if history.empty:
        return pd.DataFrame()
    return build_risk_main_customer_aggregates(history, customer_ids)


def build_risk_main_customer_aggregates(full_df: pd.DataFrame, customer_ids: list[str]) -> pd.DataFrame:
    if full_df is None or full_df.empty or "customer.customerId" not in full_df.columns:
        return pd.DataFrame()

    normalized_ids = {str(value).strip() for value in customer_ids if str(value).strip()}
    if not normalized_ids:
        return pd.DataFrame()

    customer_key_series = (
        full_df["customer.customerId"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({marker: "" for marker in _NULL_TEXT_MARKERS})
    )
    history = full_df[customer_key_series.isin(normalized_ids)].copy()
    if history.empty:
        return pd.DataFrame()

    canonical = canonicalize_risk_main_frame(history)
    if canonical.empty or "customer_key" not in canonical.columns:
        return pd.DataFrame()
    canonical["customer_key"] = (
        canonical["customer_key"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({marker: "" for marker in _NULL_TEXT_MARKERS})
    )
    canonical = canonical[canonical["customer_key"] != ""].copy()
    if canonical.empty:
        return pd.DataFrame()

    grouped = (
        canonical.groupby("customer_key", as_index=False)
        .agg(
            customer_total_invoices=("invoice_key", "size"),
            customer_delayed_invoices=("target", "sum"),
            customer_avg_invoice=("invoice_amount", "mean"),
            customer_invoice_std=("invoice_amount", "std"),
            customer_avg_delay_days=("delay_days", "mean"),
            customer_max_delay_days=("delay_days", "max"),
        )
        .rename(columns={"customer_key": "customer.customerId"})
    )
    grouped["customer_delay_rate"] = (
        grouped["customer_delayed_invoices"]
        / grouped["customer_total_invoices"].replace({0: pd.NA})
    ).fillna(0.0)
    for column, default in CUSTOMER_AGGREGATE_DEFAULTS.items():
        if column in grouped.columns:
            grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(default)
        else:
            grouped[column] = default
    return grouped


def _payload_dict(payload) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(exclude_none=False)
    if hasattr(payload, "dict"):
        return payload.dict(exclude_none=False)
    return dict(payload or {})


def build_risk_main_manual_request_frame(segment: str, payload) -> pd.DataFrame:
    raw_payload = _payload_dict(payload)
    flat_payload = flatten_payload(raw_payload)
    customer_id = str(
        flat_payload.get("customer.customerId")
        or flat_payload.get("customerId")
        or ""
    ).strip()
    customer_name = flat_payload.get("customer.customerName") or flat_payload.get("customerName")
    manual_key = str(
        flat_payload.get("invoiceNo")
        or flat_payload.get("invoice_key")
        or flat_payload.get("_id")
        or f"manual_{customer_id or 'unknown'}_{int(time())}"
    )

    record: dict[str, object] = {
        "_id": manual_key,
        "invoiceNo": manual_key,
        "invoice_key": manual_key,
        "customer.customerId": customer_id or None,
        "customer.customerName": customer_name,
    }
    for external_key, normalized_field in PRODUCTION_RISK_REQUEST_FIELD_MAP.items():
        value = flat_payload.get(external_key)
        if value is not None:
            record[normalized_field] = value
    for key, value in flat_payload.items():
        if value is None:
            continue
        if key in PRODUCTION_RISK_REQUEST_FIELD_MAP:
            continue
        if "." in key:
            record[key] = value

    if not record.get("shipmentDetails.queryFor"):
        record["shipmentDetails.queryFor"] = str(segment or "all").strip().lower()
    record["salesPersonName"] = record.get("salesPersonName") or "Unknown"
    record["invoiceType"] = record.get("invoiceType") or "Invoice"
    record["paidStatus"] = record.get("paidStatus") or "Pending"
    if not record.get("selectedCustomerCurrency"):
        record["selectedCustomerCurrency"] = record.get("customer.custCurrency") or "Unknown"
    record["shipmentDetails.accountType"] = record.get("shipmentDetails.accountType") or "Unknown"
    record["shipmentDetails.incoTerms"] = record.get("shipmentDetails.incoTerms") or "Unknown"
    record["shipmentDetails.commodity"] = record.get("shipmentDetails.commodity") or "Unknown"
    record["companyCode"] = record.get("companyCode") or "Unknown"

    invoice_date = parse_main_date(record.get("invoiceDate"))
    if pd.isna(invoice_date):
        invoice_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    record["invoiceDate"] = invoice_date

    execution_date = parse_main_date(record.get("executionDate"))
    if pd.notna(execution_date):
        record["executionDate"] = execution_date

    terms_value = pd.to_numeric(record.get("paymentTerms"), errors="coerce")
    due_date = parse_main_date(record.get("invoiceDueDate"))
    if pd.isna(due_date):
        due_date = invoice_date + pd.to_timedelta(0 if pd.isna(terms_value) else float(terms_value), unit="D")
    record["invoiceDueDate"] = due_date

    payment_raw = (
        record.get("paymentDate_raw")
        or record.get("paymentDetailsRaw")
        or flat_payload.get("paymentDateRaw")
        or flat_payload.get("paymentDate_raw")
        or flat_payload.get("paymentDetailsRaw")
    )
    parsed_payment_date = parse_main_date(record.get("paymentDate"))
    payment_parse = parse_payment_value(
        payment_raw if payment_raw is not None else record.get("paymentDate"),
        invoice_date=invoice_date,
        due_date=due_date,
        paid_amount=pd.to_numeric(record.get("paidAmount"), errors="coerce"),
        invoice_amount=pd.to_numeric(record.get("taxableTotalAmountB"), errors="coerce"),
    )
    if pd.isna(parsed_payment_date) and pd.notna(payment_parse.payment_date):
        parsed_payment_date = payment_parse.payment_date
    record["paymentDate"] = parsed_payment_date
    if payment_raw is not None:
        record["paymentDate_raw"] = payment_raw
    if payment_parse.installment_count:
        record["paymentInstallmentCount"] = payment_parse.installment_count
    if payment_parse.partial_payment_flag:
        record["partialPaymentFlag"] = True

    invoice_amount = pd.to_numeric(record.get("taxableTotalAmountB"), errors="coerce")
    gross_amount = pd.to_numeric(record.get("totalAmountB"), errors="coerce")
    if pd.isna(invoice_amount):
        invoice_amount = pd.to_numeric(record.get("invoiceAmount"), errors="coerce")
    if pd.isna(gross_amount):
        gross_amount = invoice_amount
    record["taxableTotalAmountB"] = 0.0 if pd.isna(invoice_amount) else float(invoice_amount)
    record["totalAmountB"] = 0.0 if pd.isna(gross_amount) else float(gross_amount)

    return pd.DataFrame([record])



