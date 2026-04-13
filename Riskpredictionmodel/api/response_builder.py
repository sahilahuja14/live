from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from bson import ObjectId

from ..scoring.decisioning import build_credit_suggestions
from ..scoring.utils import _approval, _risk_band, _scale_score
from ..features.registry import ACCOUNT_TYPE_COL, AMOUNT_COL, CUSTOMER_ID_COL, CUSTOMER_NAME_COL, DUE_DATE_COL, PAYMENT_DATE_COL, SEGMENT_COL
from ..pipeline.risk_map import PRODUCTION_RISK_NON_EXPLANATORY_FEATURES, display_feature_name


def _ts_to_iso(ts: float) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _json_safe_value(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _normalize_response_records(df: pd.DataFrame):
    records = df.to_dict(orient="records")
    return [_json_safe_value(row) for row in records]


def _first_present_value(series: pd.Series):
    for value in series.tolist():
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _display_amount_series(df: pd.DataFrame) -> pd.Series:
    taxable_amount = pd.to_numeric(df.get(AMOUNT_COL), errors="coerce").fillna(0.0)
    gross_amount = pd.to_numeric(
        df.get("totalAmountB", df.get("grossAmount", pd.Series(dtype=object))),
        errors="coerce",
    )
    if gross_amount.dropna().empty:
        return taxable_amount
    return gross_amount.fillna(taxable_amount)


def _top_customer_page_records(customer_frame: pd.DataFrame, top_n: int = 5) -> list[dict]:
    if customer_frame.empty:
        return []

    ranked = customer_frame.copy()
    ranked["__sort_pd"] = pd.to_numeric(ranked.get("pd"), errors="coerce").fillna(-1.0)
    ranked["__sort_delay"] = pd.to_numeric(ranked.get("average_delay_days"), errors="coerce").fillna(-1.0)
    ranked["__sort_score"] = pd.to_numeric(ranked.get("score"), errors="coerce").fillna(999999.0)
    ranked = ranked.sort_values(
        ["__sort_pd", "__sort_delay", "__sort_score"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(
        columns=["__sort_pd", "__sort_delay", "__sort_score"],
        errors="ignore",
    )
    return _normalize_response_records(ranked.head(max(int(top_n), 1)))


def _weighted_customer_pd(
    pd_series: pd.Series,
    amount_series: pd.Series,
    open_mask: pd.Series,
) -> tuple[float, dict]:
    total_invoices = int(len(pd_series))
    normalized_open_mask = open_mask.fillna(False).astype(bool)
    total_open_invoices = int(normalized_open_mask.sum())
    use_open_invoices = bool(normalized_open_mask.any())
    population = "open_invoices" if use_open_invoices else "all_invoices_no_open"
    base_mask = open_mask if use_open_invoices else pd.Series(True, index=pd_series.index)
    selected_pd = pd_series[base_mask].fillna(0.0)
    selected_amount = amount_series[base_mask].fillna(0.0).clip(lower=0.0)

    if selected_pd.empty:
        return 0.0, {
            "path": "empty_input",
            "population": population,
            "weight_method": "equal_weight",
            "invoices_used": 0,
            "total_open_invoices": total_open_invoices,
            "total_invoices": total_invoices,
            "total_weight": 0.0,
        }

    total_weight = float(selected_amount.sum())
    if total_weight > 0:
        weighted = float(
            np.average(
                selected_pd.to_numpy(dtype=float),
                weights=selected_amount.to_numpy(dtype=float),
            )
        )
        weight_method = "amount_weighted"
    else:
        weighted = float(selected_pd.mean())
        weight_method = "equal_weight_fallback"

    trace = {
        "path": f"{population}:{weight_method}",
        "population": population,
        "weight_method": weight_method,
        "invoices_used": int(len(selected_pd)),
        "total_open_invoices": total_open_invoices,
        "total_invoices": total_invoices,
        "total_weight": round(total_weight, 2),
    }
    return float(np.clip(weighted, 1e-6, 1 - 1e-6)), trace


def _feature_quality_payload(feature_quality: dict | None, row_count: int) -> dict:
    quality = dict(feature_quality or {})
    scored_rows = int(quality.get("scored_invoice_rows", row_count) or row_count)
    dropped_rows = int(quality.get("dropped_invoice_rows", 0) or 0)
    missing_feature_count = int(quality.get("missing_feature_count", 0) or 0)
    invalid_object_count = int(quality.get("invalid_object_feature_count", 0) or 0)
    invalid_datetime_count = int(quality.get("invalid_datetime_feature_count", 0) or 0)
    feature_validation_passed = bool(
        quality.get(
            "feature_validation_passed",
            missing_feature_count == 0 and invalid_object_count == 0 and invalid_datetime_count == 0,
        )
    )
    return {
        "feature_validation_passed": feature_validation_passed,
        "scored_invoice_rows": scored_rows,
        "dropped_invoice_rows": dropped_rows,
        "missing_feature_count": missing_feature_count,
        "invalid_object_feature_count": invalid_object_count,
        "invalid_datetime_feature_count": invalid_datetime_count,
        "scoring_context": quality.get("scoring_context"),
    }


def _filter_customer_rows(df: pd.DataFrame, customer_id: str | None) -> pd.DataFrame:
    if not customer_id or CUSTOMER_ID_COL not in df.columns:
        return df
    customer_str = str(customer_id).strip()
    return df[df[CUSTOMER_ID_COL].fillna("").astype(str) == customer_str].copy()


def _normalize_top_features(top_features) -> list[dict]:
    ranked: dict[str, dict] = {}
    for item in top_features or []:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("base_feature") or item.get("feature") or "").strip()
        if raw_name in PRODUCTION_RISK_NON_EXPLANATORY_FEATURES:
            continue
        contribution = _json_safe_value(item.get("contribution"))
        try:
            contribution_value = float(contribution or 0.0)
        except Exception:
            contribution_value = 0.0
        if abs(contribution_value) <= 0:
            continue
        display_name = display_feature_name(raw_name) or raw_name
        candidate = {
            "feature": display_name,
            "base_feature": display_name,
            "model_feature": raw_name or None,
            "contribution": contribution_value,
            "direction": _json_safe_value(item.get("direction")),
        }
        current = ranked.get(display_name)
        if current is None or abs(candidate["contribution"]) > abs(current["contribution"]):
            ranked[display_name] = candidate

    normalized = sorted(
        ranked.values(),
        key=lambda row: abs(float(row.get("contribution", 0.0) or 0.0)),
        reverse=True,
    )[:5]
    return [
        {
            "feature": item["feature"],
            "base_feature": item["base_feature"],
            "model_feature": item["model_feature"],
            "contribution": round(float(item["contribution"]), 6),
            "direction": item["direction"],
        }
        for item in normalized
    ]


def _derive_contract_fields(record: dict) -> dict:
    segment = record.get(SEGMENT_COL) or record.get("shipment_mode") or record.get("segment")
    invoice_date = record.get("invoiceDate") or record.get("invoice_date")
    due_date = record.get(DUE_DATE_COL) or record.get("due_date") or record.get("invoiceDueDate")
    payment_date = record.get(PAYMENT_DATE_COL) or record.get("payment_date")
    paid_status = record.get("paidStatus") or record.get("paid_status")
    amount = record.get(AMOUNT_COL) if record.get(AMOUNT_COL) is not None else record.get("invoice_amount")
    gross_amount = (
        record.get("totalAmountB")
        if record.get("totalAmountB") is not None
        else record.get("grossAmount")
        if record.get("grossAmount") is not None
        else record.get("gross_amount")
    )
    customer_name = record.get(CUSTOMER_NAME_COL) or record.get("customer_name")
    customer_id = record.get(CUSTOMER_ID_COL) or record.get("customer_key") or record.get("customerId")
    raw_top_features = record.get("top_features") or []
    contract = {
        "_id": record.get("_id") or record.get("invoice_key") or record.get("invoiceNo"),
        "invoice_key": record.get("invoice_key") or record.get("invoiceNo") or record.get("_id"),
        CUSTOMER_ID_COL: customer_id,
        CUSTOMER_NAME_COL: customer_name,
        SEGMENT_COL: segment,
        ACCOUNT_TYPE_COL: record.get(ACCOUNT_TYPE_COL) or record.get("account_type") or record.get("shipmentDetails.accountType"),
        "invoiceDate": invoice_date,
        DUE_DATE_COL: due_date,
        PAYMENT_DATE_COL: payment_date,
        "paidStatus": paid_status,
        "commodity": record.get("commodity") or record.get("shipmentDetails.commodity"),
        "currency": record.get("currency") or record.get("selectedCustomerCurrency"),
        AMOUNT_COL: amount,
        "totalAmountB": gross_amount,
        "grossAmount": gross_amount,
        "incoTerms": record.get("incoTerms") or record.get("shipmentDetails.incoTerms") or record.get("inco_terms"),
        "salesOwner": record.get("salesOwner") or record.get("sales_owner") or record.get("salesPersonName"),
        "documentType": record.get("documentType") or record.get("document_type") or record.get("invoiceType"),
        "paymentTerms": record.get("paymentTerms") if record.get("paymentTerms") is not None else record.get("terms_days"),
        "weight_discrepancy": record.get("weight_discrepancy"),
        "delay_days": record.get("delay_days"),
        "is_delayed": record.get("is_delayed", record.get("target")),
        "customer_total_invoices": record.get("customer_total_invoices"),
        "customer_avg_delay_days": record.get("customer_avg_delay_days"),
        "customer_avg_invoice": record.get("customer_avg_invoice"),
        "pd": record.get("pd"),
        "score": record.get("score"),
        "risk_band": record.get("risk_band"),
        "approval": record.get("approval"),
        "top_features": raw_top_features,
        "model_family": record.get("model_family"),
        "model_type": record.get("model_type"),
        "model_version": record.get("model_version"),
        "approval_threshold_policy": record.get("approval_threshold_policy"),
        "approval_threshold": record.get("approval_threshold"),
        "scoring_timestamp": record.get("scoring_timestamp"),
    }
    suggestions = build_credit_suggestions(contract)
    contract["top_features"] = _normalize_top_features(raw_top_features)
    contract["suggested_actions"] = suggestions
    contract["primary_action"] = suggestions[0] if suggestions else None
    return contract


def _shape_response_frame(df: pd.DataFrame, response_mode: str = "lean") -> pd.DataFrame:
    mode = (response_mode or "lean").strip().lower()
    if mode == "full":
        shaped = df.copy()
        records = shaped.to_dict(orient="records")
        enriched = []
        for record in records:
            suggestions = build_credit_suggestions(record)
            record["top_features"] = _normalize_top_features(record.get("top_features"))
            record["suggested_actions"] = suggestions
            record["primary_action"] = suggestions[0] if suggestions else None
            enriched.append(record)
        return pd.DataFrame(enriched)

    contracts = [_derive_contract_fields(record) for record in df.to_dict(orient="records")]
    return pd.DataFrame(contracts)


def _build_top_pd_customers(df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    if df.empty:
        return []

    customer_ids = df.get(CUSTOMER_ID_COL, pd.Series(dtype=object)).fillna("").astype(str).str.strip()
    working = df.copy()
    working[CUSTOMER_ID_COL] = customer_ids
    working = working[working[CUSTOMER_ID_COL] != ""].copy()
    if working.empty:
        return []

    working["__pd_rank"] = pd.to_numeric(working.get("pd", pd.Series(dtype=object)), errors="coerce").fillna(-1.0)
    working["__score_rank"] = pd.to_numeric(working.get("score", pd.Series(dtype=object)), errors="coerce").fillna(999999.0)
    working["__delay_rank"] = pd.to_numeric(working.get("delay_days", pd.Series(dtype=object)), errors="coerce").fillna(-1.0)

    ranked = (
        working.sort_values(
            ["__pd_rank", "__delay_rank", "__score_rank"],
            ascending=[False, False, True],
            na_position="last",
        )
        .drop_duplicates(subset=[CUSTOMER_ID_COL], keep="first")
        .head(max(int(top_n), 1))
        .copy()
    )

    ranked = ranked.drop(columns=["__pd_rank", "__score_rank", "__delay_rank"], errors="ignore")
    return _normalize_response_records(ranked)


def _build_scored_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "actual_delay_rate": None,
            "average_pd": None,
            "average_score": None,
            "approval_mix": {},
            "risk_band_mix": {},
            "top_pd_customers": [],
        }

    pd_series = pd.to_numeric(df.get("pd", pd.Series(dtype=object)), errors="coerce").fillna(0.0)
    target_source = df.get("is_delayed", df.get("actual_delay_rate", pd.Series(dtype=object)))
    target_series = pd.to_numeric(target_source, errors="coerce").fillna(0.0)
    score_series = pd.to_numeric(df.get("score", pd.Series(dtype=object)), errors="coerce")
    actual_delay_rate = None if target_series.empty else round(float(target_series.mean()), 6)
    average_pd = None if pd_series.empty else round(float(pd_series.mean()), 6)
    average_score = None if score_series.isna().all() else round(float(score_series.fillna(0.0).mean()), 2)
    return {
        "rows": int(len(df)),
        "actual_delay_rate": actual_delay_rate,
        "average_pd": average_pd,
        "average_score": average_score,
        "approval_mix": {
            str(key): int(value)
            for key, value in df.get("approval", pd.Series(dtype=object)).fillna("Unknown").astype(str).value_counts().to_dict().items()
        },
        "risk_band_mix": {
            str(key): int(value)
            for key, value in df.get("risk_band", pd.Series(dtype=object)).fillna("Unknown").astype(str).value_counts().to_dict().items()
        },
        "top_pd_customers": _build_top_pd_customers(df, top_n=5),
    }


def _build_customer_page_summary(customer_frame: pd.DataFrame) -> dict:
    if customer_frame.empty:
        empty_summary = {
            "customers": 0,
            "avg_customer_pd": None,
            "avg_customer_score": None,
            "avg_delay_days": None,
            "avg_actual_delay_rate": None,
            "approval_mix": {},
            "risk_band_mix": {},
            "top_customers_by_pd": [],
        }
        empty_summary.update(
            {
                "rows": empty_summary["customers"],
                "average_pd": empty_summary["avg_customer_pd"],
                "average_score": empty_summary["avg_customer_score"],
                "actual_delay_rate": empty_summary["avg_actual_delay_rate"],
                "top_pd_customers": empty_summary["top_customers_by_pd"],
            }
        )
        return empty_summary

    pd_series = pd.to_numeric(customer_frame.get("pd", pd.Series(dtype=object)), errors="coerce")
    score_series = pd.to_numeric(customer_frame.get("score", pd.Series(dtype=object)), errors="coerce")
    delay_series = pd.to_numeric(customer_frame.get("average_delay_days", pd.Series(dtype=object)), errors="coerce")
    delay_rate_series = pd.to_numeric(customer_frame.get("actual_delay_rate", pd.Series(dtype=object)), errors="coerce")
    summary = {
        "customers": int(len(customer_frame)),
        "avg_customer_pd": None if pd_series.dropna().empty else round(float(pd_series.fillna(0.0).mean()), 6),
        "avg_customer_score": None if score_series.dropna().empty else round(float(score_series.fillna(0.0).mean()), 2),
        "avg_delay_days": None if delay_series.dropna().empty else round(float(delay_series.fillna(0.0).mean()), 2),
        "avg_actual_delay_rate": None if delay_rate_series.dropna().empty else round(float(delay_rate_series.fillna(0.0).mean()), 6),
        "approval_mix": {
            str(key): int(value)
            for key, value in customer_frame.get("approval", pd.Series(dtype=object)).fillna("Unknown").astype(str).value_counts().to_dict().items()
        },
        "risk_band_mix": {
            str(key): int(value)
            for key, value in customer_frame.get("risk_band", pd.Series(dtype=object)).fillna("Unknown").astype(str).value_counts().to_dict().items()
        },
        "top_customers_by_pd": _top_customer_page_records(customer_frame, top_n=5),
    }
    summary.update(
        {
            "rows": summary["customers"],
            "average_pd": summary["avg_customer_pd"],
            "average_score": summary["avg_customer_score"],
            "actual_delay_rate": summary["avg_actual_delay_rate"],
            "top_pd_customers": summary["top_customers_by_pd"],
        }
    )
    return summary


def _aggregate_customer_top_features(records_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    stats: dict[str, dict] = {}
    for features in records_df.get("top_features", pd.Series(dtype=object)):
        if not isinstance(features, list):
            continue
        for item in features:
            if not isinstance(item, dict):
                continue
            feature_name = str(item.get("feature") or item.get("base_feature") or "").strip()
            if not feature_name:
                continue
            model_feature = str(item.get("model_feature") or item.get("base_feature") or feature_name).strip()
            contribution = float(pd.to_numeric(item.get("contribution"), errors="coerce") or 0.0)
            current = stats.setdefault(
                feature_name,
                {
                    "feature": feature_name,
                    "base_feature": feature_name,
                    "model_feature": model_feature or None,
                    "contribution_sum": 0.0,
                    "count": 0,
                },
            )
            current["contribution_sum"] += contribution
            current["count"] += 1

    ranked = sorted(
        stats.values(),
        key=lambda item: abs(item["contribution_sum"]) / max(item["count"], 1),
        reverse=True,
    )[:top_n]

    output: list[dict] = []
    for item in ranked:
        avg_contribution = item["contribution_sum"] / max(item["count"], 1)
        direction = "increase_pd" if avg_contribution > 0 else "decrease_pd" if avg_contribution < 0 else "neutral"
        output.append(
            {
                "feature": item["feature"],
                "base_feature": item["base_feature"],
                "model_feature": item["model_feature"],
                "contribution": round(float(avg_contribution), 6),
                "direction": direction,
                "invoice_count": int(item["count"]),
            }
        )
    return output


def _build_customer_portfolio_record(
    records_df: pd.DataFrame,
    *,
    segment: str | None = None,
    customer_id: str | None = None,
    model_type: str = "risk_main_xgb",
    feature_quality: dict | None = None,
    segment_invoice_rows: int | None = None,
    approval_threshold_override: float | None = None,
) -> dict:
    if records_df.empty:
        raise ValueError("Customer portfolio record requires at least one invoice row.")

    df = records_df.copy().reset_index(drop=True)
    pd_series = pd.to_numeric(df.get("pd"), errors="coerce").fillna(0.0)
    delay_series = pd.to_numeric(df.get("delay_days"), errors="coerce")
    target_series = pd.to_numeric(df.get("is_delayed"), errors="coerce").fillna(0.0)
    taxable_amount_series = pd.to_numeric(df.get(AMOUNT_COL), errors="coerce").fillna(0.0)
    display_amount_series = _display_amount_series(df)
    paid_status = df.get("paidStatus", pd.Series(dtype=object)).fillna("").astype(str).str.strip().str.lower()
    open_mask = paid_status != "paid"

    resolved_customer_id = str(
        customer_id
        or _first_present_value(df.get(CUSTOMER_ID_COL, pd.Series(dtype=object)))
        or ""
    ).strip()
    resolved_customer_name = _first_present_value(df.get(CUSTOMER_NAME_COL, pd.Series(dtype=object)))
    resolved_segment = str(
        segment
        or _first_present_value(df.get(SEGMENT_COL, pd.Series(dtype=object)))
        or "all"
    ).strip().lower()
    approval_threshold_value = pd.to_numeric(df.get("approval_threshold"), errors="coerce").dropna()
    if approval_threshold_override is not None:
        approval_threshold = float(approval_threshold_override)
    elif not approval_threshold_value.empty:
        approval_threshold = float(approval_threshold_value.iloc[0])
    else:
        approval_threshold = 0.30

    weight_series = taxable_amount_series
    if float(weight_series.clip(lower=0.0).sum()) <= 0:
        weight_series = display_amount_series

    customer_pd, pd_trace = _weighted_customer_pd(pd_series, weight_series, open_mask)
    customer_score = int(_scale_score([customer_pd])[0])
    customer_risk_band = str(_risk_band([customer_pd])[0])
    customer_approval = str(_approval([customer_pd], threshold=approval_threshold)[0])

    quality = _feature_quality_payload(feature_quality, row_count=len(df))
    first_total_invoices = pd.to_numeric(df.get("customer_total_invoices", pd.Series(dtype=object)), errors="coerce").dropna()
    first_avg_delay = pd.to_numeric(df.get("customer_avg_delay_days", pd.Series(dtype=object)), errors="coerce").dropna()
    first_avg_invoice = pd.to_numeric(df.get("customer_avg_invoice", pd.Series(dtype=object)), errors="coerce").dropna()
    first_delay_rate = pd.to_numeric(df.get("customer_delay_rate", pd.Series(dtype=object)), errors="coerce").dropna()

    customer_record = {
        "segment": resolved_segment,
        "customerId": resolved_customer_id,
        CUSTOMER_ID_COL: resolved_customer_id,
        "customerName": resolved_customer_name,
        CUSTOMER_NAME_COL: resolved_customer_name,
        SEGMENT_COL: resolved_segment,
        "invoice_rows_scored": int(len(df)),
        "segment_invoice_rows": int(segment_invoice_rows) if segment_invoice_rows is not None else int(len(df)),
        "paid_invoices": int((paid_status == "paid").sum()),
        "open_invoices": int((paid_status != "paid").sum()),
        AMOUNT_COL: round(float(taxable_amount_series.sum()), 2),
        "totalAmountB": round(float(display_amount_series.sum()), 2),
        "grossAmount": round(float(display_amount_series.sum()), 2),
        "average_invoice_amount": round(float(display_amount_series.mean()), 2),
        "average_taxable_amount": round(float(taxable_amount_series.mean()), 2),
        "actual_delay_rate": round(float(target_series.mean()), 6),
        "average_delay_days": round(float(delay_series.fillna(0.0).mean()), 2),
        "max_delay_days": None if delay_series.dropna().empty else int(delay_series.max()),
        "average_pd": round(float(pd_series.mean()), 6),
        "max_pd": round(float(pd_series.max()), 6),
        "max_invoice_pd": round(float(pd_series.max()), 6),
        "pd": round(float(customer_pd), 6),
        "pd_computation_trace": pd_trace,
        "score": customer_score,
        "risk_band": customer_risk_band,
        "approval": customer_approval,
        "customer_total_invoices": int(first_total_invoices.iloc[0]) if not first_total_invoices.empty else int(len(df)),
        "customer_avg_delay_days": round(float(first_avg_delay.iloc[0]), 2) if not first_avg_delay.empty else round(float(delay_series.fillna(0.0).mean()), 2),
        "customer_avg_invoice": round(float(first_avg_invoice.iloc[0]), 2) if not first_avg_invoice.empty else round(float(display_amount_series.mean()), 2),
        "customer_delay_rate": round(float(first_delay_rate.iloc[0]), 6) if not first_delay_rate.empty else round(float(target_series.mean()), 6),
        "top_features": _aggregate_customer_top_features(df, top_n=5),
        "model_family": _first_present_value(df.get("model_family", pd.Series(dtype=object))),
        "model_type": _first_present_value(df.get("model_type", pd.Series(dtype=object))) or model_type,
        "model_version": _first_present_value(df.get("model_version", pd.Series(dtype=object))) or "unknown",
        "approval_threshold_policy": _first_present_value(df.get("approval_threshold_policy", pd.Series(dtype=object))),
        "approval_threshold": approval_threshold,
        "scoring_timestamp": _first_present_value(df.get("scoring_timestamp", pd.Series(dtype=object))),
    }
    customer_record.update(quality)
    suggestions = build_credit_suggestions(customer_record)
    customer_record["suggested_actions"] = suggestions
    customer_record["primary_action"] = suggestions[0] if suggestions else None
    return customer_record


def build_customer_portfolio_frame(
    records: pd.DataFrame | list[dict],
    *,
    segment: str | None = None,
    feature_quality_by_customer: dict[str, dict] | None = None,
    segment_invoice_counts: dict[str, int] | None = None,
    approval_threshold_override: float | None = None,
) -> pd.DataFrame:
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame()
    if CUSTOMER_ID_COL not in frame.columns:
        return pd.DataFrame()

    working = frame.copy()
    working[CUSTOMER_ID_COL] = working[CUSTOMER_ID_COL].fillna("").astype(str).str.strip()
    working = working[working[CUSTOMER_ID_COL] != ""].copy()
    if working.empty:
        return pd.DataFrame()

    output: list[dict] = []
    for customer_key, group in working.groupby(CUSTOMER_ID_COL, sort=False):
        quality = None if feature_quality_by_customer is None else feature_quality_by_customer.get(customer_key)
        output.append(
            _build_customer_portfolio_record(
                group.copy(),
                segment=segment,
                customer_id=customer_key,
                feature_quality=quality,
                segment_invoice_rows=None if segment_invoice_counts is None else segment_invoice_counts.get(customer_key),
                approval_threshold_override=approval_threshold_override,
            )
        )

    customer_frame = pd.DataFrame(output)
    if customer_frame.empty:
        return customer_frame

    customer_frame["__sort_pd"] = pd.to_numeric(customer_frame.get("pd"), errors="coerce").fillna(0.0)
    customer_frame["__sort_delay"] = pd.to_numeric(customer_frame.get("average_delay_days"), errors="coerce").fillna(0.0)
    customer_frame["__sort_score"] = pd.to_numeric(customer_frame.get("score"), errors="coerce").fillna(999999.0)
    customer_frame = customer_frame.sort_values(
        ["__sort_pd", "__sort_delay", "__sort_score"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(columns=["__sort_pd", "__sort_delay", "__sort_score"], errors="ignore")
    return customer_frame.reset_index(drop=True)


def _build_customer_summary_payload(
    records: list[dict],
    segment: str,
    customer_id: str,
    history_preview_limit: int,
    model_type: str = "risk_main_xgb",
    feature_quality: dict | None = None,
    include_history_preview: bool = True,
    include_invoice_top_features: bool = True,
    segment_invoice_rows: int | None = None,
    approval_threshold_override: float | None = None,
) -> dict:
    if not records:
        return {
            "segment": segment.lower(),
            "customerId": customer_id,
            "model_type": model_type,
            "invoice_rows_scored": 0,
            "history_preview_limit": int(history_preview_limit),
            "customer_summary": None,
            "history_preview": [],
            "feature_quality": _feature_quality_payload(feature_quality, row_count=0),
        }

    df = pd.DataFrame(records)
    customer_summary = _build_customer_portfolio_record(
        df,
        segment=segment,
        customer_id=customer_id,
        model_type=model_type,
        feature_quality=feature_quality,
        segment_invoice_rows=segment_invoice_rows,
        approval_threshold_override=approval_threshold_override,
    )
    preview_rows = []
    if include_history_preview:
        preview_limit = max(int(history_preview_limit), 1)
        preview_rows = _normalize_response_records(df.head(preview_limit))
        if not include_invoice_top_features:
            for row in preview_rows:
                if isinstance(row, dict):
                    row.pop("top_features", None)

    return {
        "segment": segment.lower(),
        "customerId": customer_id,
        "model_type": customer_summary["model_type"],
        "invoice_rows_scored": customer_summary["invoice_rows_scored"],
        "history_preview_limit": int(history_preview_limit),
        "customer_summary": _json_safe_value(customer_summary),
        "history_preview": preview_rows,
        "feature_quality": _feature_quality_payload(feature_quality, row_count=len(df)),
    }


def build_customer_history_payload(
    *,
    segment: str,
    customer_id: str,
    customer_summary: dict,
    records: list[dict],
    total_available: int,
    returned: int,
    next_cursor: str | None,
    snapshot_meta: dict,
    feature_snapshot: list[dict] | None = None,
    canonical_snapshot: list[dict] | None = None,
) -> dict:
    return {
        "segment": segment.lower(),
        "customerId": customer_id,
        "customer_summary": _json_safe_value(customer_summary),
        "feature_quality": _json_safe_value(
            {
                "feature_validation_passed": customer_summary.get("feature_validation_passed"),
                "scored_invoice_rows": customer_summary.get("scored_invoice_rows"),
                "dropped_invoice_rows": customer_summary.get("dropped_invoice_rows"),
                "missing_feature_count": customer_summary.get("missing_feature_count"),
                "invalid_object_feature_count": customer_summary.get("invalid_object_feature_count"),
                "invalid_datetime_feature_count": customer_summary.get("invalid_datetime_feature_count"),
                "scoring_context": customer_summary.get("scoring_context"),
            }
        ),
        "total_available": int(total_available),
        "count": int(returned),
        "records": _json_safe_value(records),
        "feature_snapshot": _json_safe_value(feature_snapshot or []),
        "canonical_snapshot": _json_safe_value(canonical_snapshot or []),
        "pagination": {
            "returned": int(returned),
            "has_more": bool(next_cursor),
            "next_cursor": next_cursor,
        },
        "source": _json_safe_value(snapshot_meta),
    }


def response_from_raw(raw_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    raw = raw_df.copy().reset_index(drop=True)
    scored = scored_df.reset_index(drop=True)

    overlapping = [col for col in scored.columns if col in raw.columns]
    if overlapping:
        raw = raw.drop(columns=overlapping, errors="ignore")

    merged = pd.concat([raw, scored], axis=1)
    ordered_cols = list(raw_df.columns) + [col for col in scored.columns if col not in raw_df.columns]
    return merged.reindex(columns=ordered_cols).copy()
