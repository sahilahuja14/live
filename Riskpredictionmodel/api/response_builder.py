from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from bson import ObjectId

from ..scoring.decisioning import build_credit_suggestions
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
        "grossAmount": record.get("grossAmount") if record.get("grossAmount") is not None else record.get("gross_amount") or record.get("totalAmountB"),
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


def _build_scored_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "actual_delay_rate": None,
            "average_pd": None,
            "average_score": None,
            "approval_mix": {},
            "risk_band_mix": {},
        }

    pd_series = pd.to_numeric(df.get("pd"), errors="coerce").fillna(0.0)
    target_series = pd.to_numeric(df.get("is_delayed"), errors="coerce").fillna(0.0)
    score_series = pd.to_numeric(df.get("score"), errors="coerce")
    return {
        "rows": int(len(df)),
        "actual_delay_rate": round(float(target_series.mean()), 6),
        "average_pd": round(float(pd_series.mean()), 6),
        "average_score": None if score_series.isna().all() else round(float(score_series.fillna(0.0).mean()), 2),
        "approval_mix": {
            str(key): int(value)
            for key, value in df.get("approval", pd.Series(dtype=object)).fillna("Unknown").astype(str).value_counts().to_dict().items()
        },
        "risk_band_mix": {
            str(key): int(value)
            for key, value in df.get("risk_band", pd.Series(dtype=object)).fillna("Unknown").astype(str).value_counts().to_dict().items()
        },
    }


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
            }
        )
    return output


def _build_customer_summary_payload(
    records: list[dict],
    segment: str,
    customer_id: str,
    limit: int,
    model_type: str = "risk_main_xgb",
) -> dict:
    if not records:
        return {
            "segment": segment.lower(),
            "customerId": customer_id,
            "model_type": model_type,
            "invoice_rows_scored": 0,
            "limit_applied": limit,
            "customer_summary": None,
        }

    df = pd.DataFrame(records)
    pd_series = pd.to_numeric(df.get("pd"), errors="coerce").fillna(0.0)
    score_series = pd.to_numeric(df.get("score"), errors="coerce")
    delay_series = pd.to_numeric(df.get("delay_days"), errors="coerce")
    target_series = pd.to_numeric(df.get("is_delayed"), errors="coerce").fillna(0.0)
    amount_series = pd.to_numeric(df.get(AMOUNT_COL), errors="coerce").fillna(0.0)
    paid_status = df.get("paidStatus", pd.Series(dtype=object)).fillna("").astype(str).str.lower()
    risk_rank = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}
    risk_values = df.get("risk_band", pd.Series(dtype=object)).fillna("Low Risk").astype(str)
    worst_risk = max(risk_values.tolist(), key=lambda item: risk_rank.get(item, 0))
    approval_values = df.get("approval", pd.Series(dtype=object)).fillna("Approve").astype(str)

    customer_summary = {
        "segment": segment.lower(),
        "customerId": customer_id,
        CUSTOMER_NAME_COL: next(
            (
                str(value)
                for value in df.get(CUSTOMER_NAME_COL, pd.Series(dtype=object)).tolist()
                if str(value).strip() and str(value).strip().lower() != "nan"
            ),
            None,
        ),
        SEGMENT_COL: next(
            (
                str(value)
                for value in df.get(SEGMENT_COL, pd.Series(dtype=object)).tolist()
                if str(value).strip()
            ),
            segment.lower(),
        ),
        "invoice_rows_scored": int(len(df)),
        "limit_applied": int(limit),
        "paid_invoices": int((paid_status == "paid").sum()),
        "open_invoices": int((paid_status != "paid").sum()),
        AMOUNT_COL: round(float(amount_series.sum()), 2),
        "average_invoice_amount": round(float(amount_series.mean()), 2),
        "actual_delay_rate": round(float(target_series.mean()), 6),
        "average_delay_days": round(float(delay_series.fillna(0.0).mean()), 2),
        "max_delay_days": None if delay_series.dropna().empty else int(delay_series.max()),
        "average_pd": round(float(pd_series.mean()), 6),
        "max_pd": round(float(pd_series.max()), 6),
        "pd": round(float(pd_series.max()), 6),
        "score": None if score_series.isna().all() else int(score_series.min()),
        "risk_band": worst_risk,
        "approval": "Reject" if (approval_values.str.lower() == "reject").any() else "Approve",
        "top_features": _aggregate_customer_top_features(df, top_n=5),
        "model_family": next((str(value) for value in df.get("model_family", pd.Series(dtype=object)).dropna().astype(str).tolist()), None),
        "model_type": next((str(value) for value in df.get("model_type", pd.Series(dtype=object)).dropna().astype(str).tolist()), model_type),
        "model_version": next((str(value) for value in df.get("model_version", pd.Series(dtype=object)).dropna().astype(str).tolist()), "unknown"),
        "approval_threshold_policy": next((str(value) for value in df.get("approval_threshold_policy", pd.Series(dtype=object)).dropna().astype(str).tolist()), None),
        "approval_threshold": None
        if pd.to_numeric(df.get("approval_threshold"), errors="coerce").dropna().empty
        else float(pd.to_numeric(df.get("approval_threshold"), errors="coerce").dropna().iloc[0]),
        "scoring_timestamp": next((str(value) for value in df.get("scoring_timestamp", pd.Series(dtype=object)).dropna().astype(str).tolist()), None),
    }
    suggestions = build_credit_suggestions(customer_summary)
    customer_summary["suggested_actions"] = suggestions
    customer_summary["primary_action"] = suggestions[0] if suggestions else None

    return {
        "segment": segment.lower(),
        "customerId": customer_id,
        "model_type": customer_summary["model_type"],
        "invoice_rows_scored": customer_summary["invoice_rows_scored"],
        "limit_applied": customer_summary["limit_applied"],
        "customer_summary": _json_safe_value(customer_summary),
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
