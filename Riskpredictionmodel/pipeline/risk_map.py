from __future__ import annotations

import os
import re

from ..config import get_production_risk_collection, get_production_risk_db_name, init_env

init_env()

_PRODUCTION_MODEL_FAMILY_RAW = os.getenv("PRODUCTION_MODEL_FAMILY", "risk_main").strip().lower()
PRODUCTION_MODEL_FAMILY = "risk_main" if _PRODUCTION_MODEL_FAMILY_RAW != "risk_main" else _PRODUCTION_MODEL_FAMILY_RAW

PRODUCTION_RISK_DB_NAME = get_production_risk_db_name()
PRODUCTION_RISK_COLLECTION = get_production_risk_collection()
PRODUCTION_RISK_REGISTRY_PATH = os.getenv(
    "PRODUCTION_RISK_REGISTRY_PATH",
    os.path.join("models", "production", "risk_main_registry.json"),
)
PRODUCTION_RISK_ACTIVE_MODEL_TYPE = (
    os.getenv("PRODUCTION_RISK_ACTIVE_MODEL_TYPE", "risk_main_xgb").strip() or "risk_main_xgb"
)
PRODUCTION_RISK_ACTIVE_VERSION = (
    os.getenv("PRODUCTION_RISK_ACTIVE_VERSION", "risk_main_xgb_20260319_094014").strip()
    or "risk_main_xgb_20260319_094014"
)
TARGET_DELAY_DAYS = int(os.getenv("PRODUCTION_RISK_TARGET_DELAY_DAYS", "30"))


RAW_TO_NORMALIZED_FIELD_MAP = {
    "BillNo": "invoiceNo",
    "INVOICE": "legacy.invoice_ref_raw",
    "Date": "invoiceDate",
    "Due Date": "invoiceDueDate",
    "Payment Date": "paymentDate_raw",
    "Terms": "paymentTerms",
    " Status ": "paidStatus",
    "Customer Name": "customer.customerName",
    "Sales Person": "salesPersonName",
    "CT": "invoiceType",
    " Currency ": "selectedCustomerCurrency",
    " Bill Amount ": "taxableTotalAmountB",
    " Revenue ": "totalAmountB",
    " Paid ": "paidAmount",
    " TDS ": "tdsAmount",
    " YTD Exposure ": "ytdExposure",
    " Payment Details ": "paymentDetailsRaw",
    " Not Due ": "receivables.notDueAmount",
    " 0-15 ": "receivables.bucket0To15Amount",
    " 16-30 ": "receivables.bucket16To30Amount",
    " 31-45 ": "receivables.bucket31To45Amount",
    " 46-60 ": "receivables.bucket46To60Amount",
    " 60-90 ": "receivables.bucket60To90Amount",
    " Above 90 ": "receivables.bucketAbove90Amount",
    "Company": "companyCode",
}


NORMALIZED_TO_CANONICAL_FIELD_MAP = {
    "invoiceNo": "invoice_key",
    "invoiceDate": "invoice_date",
    "invoiceDueDate": "due_date",
    "paymentDate": "payment_date",
    "paidStatus": "paid_status",
    "paymentTerms": "terms_days",
    "selectedCustomerCurrency": "currency",
    "taxableTotalAmountB": "invoice_amount",
    "totalAmountB": "gross_amount",
    "customer.customerId": "customer_key",
    "customer.customerName": "customer_name",
    "salesPersonName": "sales_owner",
    "invoiceType": "document_type",
    "shipmentDetails.queryFor": "shipment_mode",
    "shipmentDetails.accountType": "account_type",
    "shipmentDetails.incoTerms": "inco_terms",
    "shipmentDetails.commodity": "commodity",
    "executionDate": "execution_date",
    "paymentInstallmentCount": "payment_installment_count",
    "partialPaymentFlag": "partial_payment_flag",
    "paidAmount": "paid_amount",
    "tdsAmount": "tds_amount",
    "ytdExposure": "ytd_exposure",
    "receivables.notDueAmount": "aging_not_due",
    "receivables.bucket0To15Amount": "aging_0_15",
    "receivables.bucket16To30Amount": "aging_16_30",
    "receivables.bucket31To45Amount": "aging_31_45",
    "receivables.bucket46To60Amount": "aging_46_60",
    "receivables.bucket60To90Amount": "aging_60_90",
    "receivables.bucketAbove90Amount": "aging_above_90",
    "shipmentDetails.grossWeight": "gross_weight",
    "shipmentDetails.chargeableWeight": "chargeable_weight",
    "shipmentDetails.volumeWeight": "volume_weight",
    "shipmentDetails.noOfContainers": "container_count",
    "shipmentDetails.originCity": "origin_city",
    "shipmentDetails.originState": "origin_state",
    "shipmentDetails.originCountry": "origin_country",
    "shipmentDetails.destinationCity": "destination_city",
    "shipmentDetails.destinationState": "destination_state",
    "shipmentDetails.destinationCountry": "destination_country",
    "customer.customerAccountType": "customer_account_type",
    "customer.customerType": "customer_type",
    "customer.category": "customer_category",
    "customer.onboardDate": "customer_onboard_date",
    "operational.jobNo": "job_no",
    "operational.bookingNo": "booking_no",
    "operational.clearanceStatus": "clearance_status",
    "operational.gateInStatus": "gatein_status",
    "operational.lastTrackingStatus": "tracking_status",
    "operational.lastTrackingLocation": "tracking_location",
    "operational.shippingBillNo": "shipping_bill_no",
    "operational.containerNo": "container_no",
}


PRODUCTION_RISK_REQUEST_FIELD_MAP = {
    "customerId": "customer.customerId",
    "customer.customerId": "customer.customerId",
    "customerName": "customer.customerName",
    "customer.customerName": "customer.customerName",
    "invoiceNo": "invoiceNo",
    "invoice_key": "invoiceNo",
    "_id": "_id",
    "invoiceDate": "invoiceDate",
    "executionDate": "executionDate",
    "dueDate": "invoiceDueDate",
    "invoiceDueDate": "invoiceDueDate",
    "paymentDate": "paymentDate",
    "paymentDateRaw": "paymentDate_raw",
    "paymentDate_raw": "paymentDate_raw",
    "paymentDetailsRaw": "paymentDetailsRaw",
    "paidStatus": "paidStatus",
    "salesOwner": "salesPersonName",
    "salesPersonName": "salesPersonName",
    "currency": "selectedCustomerCurrency",
    "selectedCustomerCurrency": "selectedCustomerCurrency",
    "termsDays": "paymentTerms",
    "paymentTerms": "paymentTerms",
    "taxableTotalAmountB": "taxableTotalAmountB",
    "invoiceAmount": "taxableTotalAmountB",
    "totalAmountB": "totalAmountB",
    "grossAmount": "totalAmountB",
    "documentType": "invoiceType",
    "invoiceType": "invoiceType",
    "shipmentMode": "shipmentDetails.queryFor",
    "shipmentDetails.queryFor": "shipmentDetails.queryFor",
    "accountType": "shipmentDetails.accountType",
    "shipmentDetails.accountType": "shipmentDetails.accountType",
    "incoTerms": "shipmentDetails.incoTerms",
    "shipmentDetails.incoTerms": "shipmentDetails.incoTerms",
    "commodity": "shipmentDetails.commodity",
    "shipmentDetails.commodity": "shipmentDetails.commodity",
    "grossWeight": "shipmentDetails.grossWeight",
    "shipmentDetails.grossWeight": "shipmentDetails.grossWeight",
    "chargeableWeight": "shipmentDetails.chargeableWeight",
    "shipmentDetails.chargeableWeight": "shipmentDetails.chargeableWeight",
    "volumeWeight": "shipmentDetails.volumeWeight",
    "shipmentDetails.volumeWeight": "shipmentDetails.volumeWeight",
    "containerCount": "shipmentDetails.noOfContainers",
    "shipmentDetails.noOfContainers": "shipmentDetails.noOfContainers",
    "originCity": "shipmentDetails.originCity",
    "shipmentDetails.originCity": "shipmentDetails.originCity",
    "originState": "shipmentDetails.originState",
    "shipmentDetails.originState": "shipmentDetails.originState",
    "originCountry": "shipmentDetails.originCountry",
    "shipmentDetails.originCountry": "shipmentDetails.originCountry",
    "destinationCity": "shipmentDetails.destinationCity",
    "shipmentDetails.destinationCity": "shipmentDetails.destinationCity",
    "destinationState": "shipmentDetails.destinationState",
    "shipmentDetails.destinationState": "shipmentDetails.destinationState",
    "destinationCountry": "shipmentDetails.destinationCountry",
    "shipmentDetails.destinationCountry": "shipmentDetails.destinationCountry",
    "customerAccountType": "customer.customerAccountType",
    "customer.customerAccountType": "customer.customerAccountType",
    "customerType": "customer.customerType",
    "customer.customerType": "customer.customerType",
    "customerCategory": "customer.category",
    "customer.category": "customer.category",
    "customerOnboardDate": "customer.onboardDate",
    "customer.onboardDate": "customer.onboardDate",
    "customerCurrency": "customer.custCurrency",
    "customer.custCurrency": "customer.custCurrency",
    "paidAmount": "paidAmount",
    "tdsAmount": "tdsAmount",
    "ytdExposure": "ytdExposure",
    "notDueAmount": "receivables.notDueAmount",
    "receivables.notDueAmount": "receivables.notDueAmount",
    "bucket0To15Amount": "receivables.bucket0To15Amount",
    "receivables.bucket0To15Amount": "receivables.bucket0To15Amount",
    "bucket16To30Amount": "receivables.bucket16To30Amount",
    "receivables.bucket16To30Amount": "receivables.bucket16To30Amount",
    "bucket31To45Amount": "receivables.bucket31To45Amount",
    "receivables.bucket31To45Amount": "receivables.bucket31To45Amount",
    "bucket46To60Amount": "receivables.bucket46To60Amount",
    "receivables.bucket46To60Amount": "receivables.bucket46To60Amount",
    "bucket60To90Amount": "receivables.bucket60To90Amount",
    "receivables.bucket60To90Amount": "receivables.bucket60To90Amount",
    "bucketAbove90Amount": "receivables.bucketAbove90Amount",
    "receivables.bucketAbove90Amount": "receivables.bucketAbove90Amount",
    "jobNo": "operational.jobNo",
    "operational.jobNo": "operational.jobNo",
    "bookingNo": "operational.bookingNo",
    "operational.bookingNo": "operational.bookingNo",
    "clearanceStatus": "operational.clearanceStatus",
    "operational.clearanceStatus": "operational.clearanceStatus",
    "gateInStatus": "operational.gateInStatus",
    "operational.gateInStatus": "operational.gateInStatus",
    "trackingStatus": "operational.lastTrackingStatus",
    "operational.lastTrackingStatus": "operational.lastTrackingStatus",
    "trackingLocation": "operational.lastTrackingLocation",
    "operational.lastTrackingLocation": "operational.lastTrackingLocation",
    "shippingBillNo": "operational.shippingBillNo",
    "operational.shippingBillNo": "operational.shippingBillNo",
    "containerNo": "operational.containerNo",
    "operational.containerNo": "operational.containerNo",
    "company": "companyCode",
    "companyCode": "companyCode",
}


PRODUCTION_RISK_FEATURE_ALIASES = {
    "invoice_amount": "Invoice Amount",
    "gross_amount": "Gross Amount",
    "terms_days": "Payment Terms",
    "shipment_mode": "Shipment Mode",
    "account_type": "Account Type",
    "inco_terms": "Inco Terms",
    "commodity": "Commodity",
    "customer_key": "Customer Id",
    "customer_name": "Customer Name",
    "sales_owner": "Sales Owner",
    "document_type": "Document Type",
    "currency": "Currency",
    "paid_amount": "Paid Amount",
    "tds_amount": "TDS Amount",
    "ytd_exposure": "YTD Exposure",
    "aging_total": "Total Aging",
    "aging_total_to_invoice_ratio": "Aging To Invoice Ratio",
    "receivable_pressure_index": "Receivable Pressure",
    "working_capital_stress": "Working Capital Stress",
    "realization_efficiency": "Realization Efficiency",
    "prior_delay_rate": "Customer Prior Delay Rate",
    "prior_avg_delay_days": "Customer Prior Avg Delay",
    "customer_hist_delay_rate": "Customer History Delay Rate",
    "customer_hist_recent3_delay_rate": "Customer Recent 3 Delay Rate",
    "customer_hist_recent5_delay_rate": "Customer Recent 5 Delay Rate",
    "cust_company_hist_delay_rate": "Customer Company Delay Rate",
    "cust_company_hist_recent3_delay_rate": "Customer Company Recent 3 Delay Rate",
    "cust_sales_hist_recent3_delay_rate": "Customer Sales Recent 3 Delay Rate",
    "cust_currency_hist_recent3_delay_rate": "Customer Currency Recent 3 Delay Rate",
    "cust_terms_hist_recent3_delay_rate": "Customer Terms Recent 3 Delay Rate",
    "sales_company_hist_recent3_delay_rate": "Sales Company Recent 3 Delay Rate",
    "route_company_hist_recent3_delay_rate": "Route Company Recent 3 Delay Rate",
    "severity_x_recurring_flag": "Severity x Recurring Delay",
}

PRODUCTION_RISK_NON_EXPLANATORY_FEATURES = {
    "source_system",
    "invoice_key",
    "payment_date",
    "payment_date_raw",
    "payment_details_raw",
    "label_quality",
    "customer_sales_key",
    "customer_company_key",
    "sales_company_key",
    "route_company_key",
    "customer_currency_key",
    "customer_terms_key",
}



_PRETTY_TOKENS = {
    "hist": "History",
    "avg": "Avg",
    "rate": "Rate",
    "recent3": "Recent 3",
    "recent5": "Recent 5",
    "cust": "Customer",
    "terms": "Terms",
    "currency": "Currency",
    "sales": "Sales",
    "company": "Company",
    "route": "Route",
    "delay": "Delay",
    "invoice": "Invoice",
    "amount": "Amount",
    "aging": "Aging",
    "exposure": "Exposure",
}


def display_feature_name(name: str | None) -> str | None:
    if name is None:
        return None
    cleaned = str(name).strip()
    if not cleaned:
        return None
    alias = PRODUCTION_RISK_FEATURE_ALIASES.get(cleaned)
    if alias:
        return alias
    if "__x__" in cleaned:
        return " x ".join(filter(None, (display_feature_name(part) for part in cleaned.split("__x__"))))
    text = cleaned.replace(".", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split(" "):
        words.append(_PRETTY_TOKENS.get(word.lower(), word.title()))
    return " ".join(words)
