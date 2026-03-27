from __future__ import annotations

from datetime import datetime, timezone
import os

import pandas as pd

from .risk_map import PRODUCTION_RISK_COLLECTION, PRODUCTION_RISK_DB_NAME, TARGET_DELAY_DAYS
from ..dbconnect import get_database
from .risk_map import NORMALIZED_TO_CANONICAL_FIELD_MAP
from .parsing import parse_main_date
from .utils import flatten_dict, safe_ratio

#only required column fetching from mongo to avoid bulk data transfer and then canonicalize it in pandas for further processing and model training
RISK_MAIN_FETCH_BATCH_SIZE = max(int(os.getenv("API_RISK_MAIN_FETCH_BATCH_SIZE", "1000")), 1)
RISK_MAIN_FETCH_PROJECTION = {
    "_id": 1,
    "invoiceNo": 1,
    "invoiceDate": 1,
    "invoiceDueDate": 1,
    "paymentDate": 1,
    "paidStatus": 1,
    "paymentTerms": 1,
    "selectedCustomerCurrency": 1,
    "taxableTotalAmountB": 1,
    "totalAmountB": 1,
    "customer.customerId": 1,
    "customer.customerName": 1,
    "salesPersonName": 1,
    "invoiceType": 1,
    "shipmentDetails.queryFor": 1,
    "shipmentDetails.accountType": 1,
    "shipmentDetails.incoTerms": 1,
    "shipmentDetails.commodity": 1,
    "executionDate": 1,
    "paymentInstallmentCount": 1,
    "partialPaymentFlag": 1,
    "paidAmount": 1,
    "tdsAmount": 1,
    "ytdExposure": 1,
    "receivables.notDueAmount": 1,
    "receivables.bucket0To15Amount": 1,
    "receivables.bucket16To30Amount": 1,
    "receivables.bucket31To45Amount": 1,
    "receivables.bucket46To60Amount": 1,
    "receivables.bucket60To90Amount": 1,
    "receivables.bucketAbove90Amount": 1,
    "shipmentDetails.grossWeight": 1,
    "shipmentDetails.chargeableWeight": 1,
    "shipmentDetails.volumeWeight": 1,
    "shipmentDetails.noOfContainers": 1,
    "shipmentDetails.originCity": 1,
    "shipmentDetails.originState": 1,
    "shipmentDetails.originCountry": 1,
    "shipmentDetails.destinationCity": 1,
    "shipmentDetails.destinationState": 1,
    "shipmentDetails.destinationCountry": 1,
    "customer.customerAccountType": 1,
    "customer.customerType": 1,
    "customer.category": 1,
    "customer.onboardDate": 1,
    "operational.jobNo": 1,
    "operational.bookingNo": 1,
    "operational.clearanceStatus": 1,
    "operational.gateInStatus": 1,
    "operational.lastTrackingStatus": 1,
    "operational.lastTrackingLocation": 1,
    "operational.shippingBillNo": 1,
    "operational.containerNo": 1,
    "paymentDate_raw": 1,
    "paymentDetailsRaw": 1,
    "legacy.invoice_ref_raw": 1,
    "companyCode": 1,
}
RISK_MAIN_RECOMMENDED_INDEXES = [
    {
        "name": "risk_main_segment_invoice_sort",
        "keys": [("shipmentDetails.queryFor", 1), ("invoiceDate", -1), ("_id", -1)],
    },
    {
        "name": "risk_main_customer_lookup",
        "keys": [("customer.customerId", 1)],
    },
    {
        "name": "risk_main_invoice_lookup",
        "keys": [("invoiceNo", 1)],
    },
]


def _series_from_frame(df: pd.DataFrame, column: str, default=None) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)

#Main mongo call/projection query to fetch the risk main data and convert it into a dataframe with batchsize to avoid bulk 
def fetch_risk_main_frame(query: dict | None = None, limit: int | None = None) -> pd.DataFrame:
    collection = get_database(PRODUCTION_RISK_DB_NAME)[PRODUCTION_RISK_COLLECTION]
    cursor = collection.find(query or {}, RISK_MAIN_FETCH_PROJECTION).batch_size(RISK_MAIN_FETCH_BATCH_SIZE)
    if limit is not None:
        cursor = cursor.limit(int(limit))
    rows = [flatten_dict(doc) for doc in cursor]
    return pd.DataFrame(rows)


def inspect_risk_main_indexes() -> dict:
    collection = get_database(PRODUCTION_RISK_DB_NAME)[PRODUCTION_RISK_COLLECTION]
    existing = list(collection.list_indexes())
    existing_specs = {
        tuple((str(key), int(direction)) for key, direction in index.get("key", {}).items())
        for index in existing
    }
    recommended = []
    for item in RISK_MAIN_RECOMMENDED_INDEXES:
        keys = tuple((str(key), int(direction)) for key, direction in item["keys"])
        recommended.append(
            {
                "name": item["name"],
                "keys": list(item["keys"]),
                "present": keys in existing_specs,
            }
        )
    return {
        "database": PRODUCTION_RISK_DB_NAME,
        "collection": PRODUCTION_RISK_COLLECTION,
        "existing": [
            {
                "name": str(index.get("name")),
                "keys": list(index.get("key", {}).items()),
            }
            for index in existing
        ],
        "recommended": recommended,
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
        "terms_days",
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

    due_fallback = canonical["invoice_date"] + pd.to_timedelta(canonical["terms_days"].fillna(0), unit="D")
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

