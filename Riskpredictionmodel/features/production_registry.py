from __future__ import annotations

from dataclasses import dataclass, field

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
    missing_features: list[str] = field(default_factory=list)
    invalid_datetime_features: list[str] = field(default_factory=list)
    invalid_object_features: list[str] = field(default_factory=list)
    domain_gaps: dict[str, list[str]] = field(default_factory=dict)
    expected_feature_count: int = 0
    present_feature_count: int = 0

    @property
    def missing_count(self) -> int:
        return (
            len(self.missing_columns)
            + len(self.missing_features)
            + len(self.invalid_datetime_features)
            + len(self.invalid_object_features)
        )

    @property
    def is_valid(self) -> bool:
        return self.missing_count == 0


def validate_production_columns(columns: list[str]) -> ValidationResult:
    missing = [column for column in REQUIRED_CANONICAL_COLUMNS if column not in columns]
    return ValidationResult(missing_columns=missing)


def _feature_domain(name: str) -> str:
    lowered = str(name).lower()
    if lowered in {"invoice_key", "invoice_date", "due_date", "terms_days", "invoice_amount", "gross_amount"}:
        return "invoice"
    if lowered.startswith("customer") or lowered.startswith("cust_"):
        return "customer"
    if "payment" in lowered or lowered in {"paid_status", "paid_amount", "tds_amount", "delay_days"}:
        return "payment"
    if any(
        token in lowered
        for token in [
            "sales",
            "shipment",
            "origin",
            "destination",
            "route",
            "company",
            "job",
            "booking",
            "tracking",
            "gate",
            "clearance",
        ]
    ):
        return "admin_operational"
    return "feature_frame"


def _first_present_value(series: pd.Series):
    for value in series.tolist():
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        return value
    return None


def validate_feature_frame(df: pd.DataFrame, expected_features: list[str]) -> ValidationResult:
    production = validate_production_columns(list(df.columns))
    missing_features = [feature for feature in expected_features if feature not in df.columns]
    invalid_datetime_features = [
        feature
        for feature in expected_features
        if feature in df.columns and pd.api.types.is_datetime64_any_dtype(df[feature])
    ]

    invalid_object_features: list[str] = []
    for feature in expected_features:
        if feature not in df.columns:
            continue
        series = df[feature]
        sample = _first_present_value(series)
        if isinstance(sample, (dict, list, tuple, set)):
            invalid_object_features.append(feature)

    domain_gaps: dict[str, list[str]] = {}
    for name in sorted(
        set(production.missing_columns)
        | set(missing_features)
        | set(invalid_datetime_features)
        | set(invalid_object_features)
    ):
        domain_gaps.setdefault(_feature_domain(name), []).append(name)

    present_feature_count = sum(1 for feature in expected_features if feature in df.columns)
    return ValidationResult(
        missing_columns=production.missing_columns,
        missing_features=missing_features,
        invalid_datetime_features=invalid_datetime_features,
        invalid_object_features=invalid_object_features,
        domain_gaps=domain_gaps,
        expected_feature_count=len(expected_features),
        present_feature_count=present_feature_count,
    )


def select_model_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for column in df.columns:
        if column in EXCLUDED_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            continue
        features.append(column)
    return features
