from __future__ import annotations

from typing import Any

from ..features.registry import ACCOUNT_TYPE_COL, AMOUNT_COL, CURRENCY_COL, SEGMENT_COL


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _feature_names(record: dict) -> list[str]:
    features = record.get("top_features") or []
    names: list[str] = []
    for item in features:
        if not isinstance(item, dict):
            continue
        name = str(item.get("base_feature") or item.get("feature") or "").strip()
        if name:
            names.append(name)
    return names


def build_credit_suggestions(record: dict, max_items: int = 3) -> list[str]:
    pd_value = _safe_float(record.get("pd"))
    avg_delay = _safe_float(record.get("customer_avg_delay_days"))
    total_invoices = _safe_float(record.get("customer_total_invoices"))
    approval = str(record.get("approval") or "").strip().lower()
    risk_band = str(record.get("risk_band") or "").strip().lower()
    feature_names = _feature_names(record)

    suggestions: list[str] = []

    if approval == "reject" or risk_band == "high risk" or pd_value >= 0.65:
        suggestions.append("Move to advance payment or LC-backed approval before releasing credit.")
    elif risk_band == "medium risk" or pd_value >= 0.15:
        suggestions.append("Approve with tighter payment terms and weekly collections follow-up.")
    else:
        suggestions.append("Approve on standard terms with routine monitoring.")

    if total_invoices >= 3 and avg_delay >= 15:
        suggestions.append("Review customer payment behavior and consider reducing the working credit limit.")

    if any(name in {"invoice_amount", "gross_amount", AMOUNT_COL, "grossAmount", "amount_log", "taxableTotalAmountB_log"} for name in feature_names):
        suggestions.append("Route high-value exposure through finance approval before final credit release.")

    if any(name in {"weight_discrepancy", "execution_gap_days"} for name in feature_names):
        suggestions.append("Validate shipment and billing discrepancies before operational release.")

    if any(name in {"currency", CURRENCY_COL, "inco_terms", "incoTerms", "shipment_mode", SEGMENT_COL, "commodity"} for name in feature_names):
        suggestions.append("Review lane, term, and commodity risk before confirming the commercial commitment.")

    if any(name in {"account_type", ACCOUNT_TYPE_COL, "paidStatus", "paid_status"} for name in feature_names):
        suggestions.append("Reconfirm payment-status evidence and account terms before marking the invoice safe.")

    deduped: list[str] = []
    for item in suggestions:
        if item not in deduped:
            deduped.append(item)
        if len(deduped) >= max_items:
            break
    return deduped
