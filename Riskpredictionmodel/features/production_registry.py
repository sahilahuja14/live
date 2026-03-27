from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_CANONICAL_COLUMNS = [
    "invoice_key",
    "invoice_date",
    "due_date",
    "paid_status",
    "delay_days",
    "target",
    "source_system",
    "label_quality",
]

EXCLUDED_FEATURE_COLUMNS = {
    "_id",
    "invoice_key",
    "invoice_ref_raw",
    "payment_date",
    "payment_date_raw",
    "payment_details_raw",
    "delay_days",
    "target",
    "label_quality",
    "paid_status",
    "partial_payment_flag",
    "payment_installment_count",
    "customer_key",
    "customer_name",
    "job_no",
    "booking_no",
    "shipping_bill_no",
    "container_no",
    "invoice_date",
    "due_date",
    "execution_date",
    "customer_onboard_date",
}


@dataclass(frozen=True)
class ValidationResult:
    missing_columns: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_columns


def validate_production_columns(columns: list[str]) -> ValidationResult:
    missing = [column for column in REQUIRED_CANONICAL_COLUMNS if column not in columns]
    return ValidationResult(missing_columns=missing)


def select_model_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for column in df.columns:
        if column in EXCLUDED_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        features.append(column)
    return features
