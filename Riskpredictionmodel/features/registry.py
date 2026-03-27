from __future__ import annotations

from dataclasses import dataclass


CUSTOMER_ID_COL = "customer.customerId"
CUSTOMER_NAME_COL = "customer.customerName"
SEGMENT_COL = "shipmentDetails.queryFor"
ACCOUNT_TYPE_COL = "shipmentDetails.accountType"
INVOICE_DATE_COL = "invoiceDate"
DUE_DATE_COL = "invoiceDueDate"
PAYMENT_DATE_COL = "paymentDate"
PAID_STATUS_COL = "paidStatus"
AMOUNT_COL = "taxableTotalAmountB"
CURRENCY_COL = "selectedCustomerCurrency"


REQUIRED_BASE_COLUMNS = [
    CUSTOMER_ID_COL,
    AMOUNT_COL,
    SEGMENT_COL,
    DUE_DATE_COL,
    PAID_STATUS_COL,
]


LEAKAGE_COLUMNS = [
    "delay_days",
    "is_delayed",
    "is_credit_bad",
    "delay_severity_bucket",
    PAYMENT_DATE_COL,
]


RAW_FEATURES = [
    "account_type",
    "commodity",
    "incoTerms",
    "currency",
    "weight_discrepancy",
    "execution_gap_days",
    "customer_age_days",
    AMOUNT_COL,
    "totalAmountB",
    "amount_bucket",
    "customer_total_invoices",
    "customer_avg_invoice",
    "customer_invoice_std",
]


# Compatibility aliases while tree and WOE trainers still import separate names.
TREE_FEATURES = RAW_FEATURES
WOE_FEATURES = RAW_FEATURES


OUTPUT_REQUIRED_FEATURES = [
    CUSTOMER_ID_COL,
    CUSTOMER_NAME_COL,
    AMOUNT_COL,
    "currency",
    "commodity",
    "incoTerms",
    "weight_discrepancy",
    DUE_DATE_COL,
    PAID_STATUS_COL,
    SEGMENT_COL,
    ACCOUNT_TYPE_COL,
    "customer_total_invoices",
    "customer_avg_invoice",
    "customer_avg_delay_days",
]


NON_EXPLANATORY_FEATURES = {
    "source_system",
    "sales_owner",
    "document_type",
    "invoice_key",
    "label_quality",
}


FEATURE_TO_MONGO_FIELD_MAP = {
    "account_type": ACCOUNT_TYPE_COL,
    "commodity": "commodity",
    "incoTerms": "incoTerms",
    "currency": CURRENCY_COL,
    "weight_discrepancy": "weight_discrepancy",
    "invoice_amount": AMOUNT_COL,
    "gross_amount": "grossAmount",
    "terms_days": "paymentTerms",
    "shipment_mode": SEGMENT_COL,
    "customer_key": CUSTOMER_ID_COL,
    "customer_name": CUSTOMER_NAME_COL,
    AMOUNT_COL: AMOUNT_COL,
    "totalAmountB": "totalAmountB",
    "customer_total_invoices": "customer_total_invoices",
    "customer_delayed_invoices": "customer_delayed_invoices",
    "customer_delay_rate": "customer_delay_rate",
    "customer_avg_invoice": "customer_avg_invoice",
    "customer_invoice_std": "customer_invoice_std",
    "customer_avg_delay_days": "customer_avg_delay_days",
    "customer_max_delay_days": "customer_max_delay_days",
    "invoice_date": INVOICE_DATE_COL,
    "due_date": DUE_DATE_COL,
    "payment_date": PAYMENT_DATE_COL,
    "paid_status": PAID_STATUS_COL,
    "sales_owner": "salesPersonName",
    "document_type": "invoiceType",
    "amount_log": "taxableTotalAmountB_log",
    "prior_invoice_count": "customer_prior_invoice_count",
    "prior_paid_invoice_count": "customer_prior_paid_invoice_count",
    "prior_delay_rate": "customer_prior_delay_rate",
    "prior_severe_delay_rate": "customer_prior_severe_delay_rate",
    "prior_avg_delay_days": "customer_prior_avg_delay_days",
    "prior_max_delay_days": "customer_prior_max_delay_days",
    "prior_on_time_rate": "customer_prior_on_time_rate",
    "prior_late_invoice_count": "customer_prior_late_invoice_count",
    "prior_open_invoice_count": "customer_prior_open_invoice_count",
    "prior_avg_invoice_amount": "customer_prior_avg_invoice_amount",
    "invoice_amount_vs_prior_avg": "invoiceAmountVsCustomerAvg",
    "days_since_last_invoice": "customer_days_since_last_invoice",
    "recurring_delay_flag": "customer_recurring_delay_flag",
}


@dataclass(frozen=True)
class ValidationResult:
    missing_columns: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_columns


def validate_required_columns(columns: list[str]) -> ValidationResult:
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in columns]
    return ValidationResult(missing_columns=missing)
