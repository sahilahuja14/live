from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Riskpredictionmodel.config import (
    get_database_name,
    get_live_db_name,
    get_live_mongo_uri,
    get_mongo_uri,
    get_source_mode,
    init_env,
)
from Riskpredictionmodel.features.production_registry import validate_feature_frame
from Riskpredictionmodel.pipeline.live_adapter import compute_live_coverage, get_live_diagnostics
from Riskpredictionmodel.pipeline.risk_canonical import canonicalize_risk_main_frame, fetch_risk_main_frame
from Riskpredictionmodel.pipeline.risk_main import build_risk_main_scoring_frame
from Riskpredictionmodel.scoring.model import load_production_artifacts, score_production_frame


CANONICAL_DOMAIN_GROUPS = {
    "invoice": [
        "invoice_key",
        "invoice_date",
        "due_date",
        "terms_days",
        "invoice_amount",
        "gross_amount",
        "document_type",
        "currency",
        "company",
    ],
    "payment_finance": [
        "paid_status",
        "payment_date",
        "payment_date_raw",
        "payment_details_raw",
        "paid_amount",
        "tds_amount",
        "ytd_exposure",
        "payment_installment_count",
        "partial_payment_flag",
    ],
    "aging_receivables": [
        "aging_not_due",
        "aging_0_15",
        "aging_16_30",
        "aging_31_45",
        "aging_46_60",
        "aging_60_90",
        "aging_above_90",
        "aging_total",
        "aging_total_to_invoice_ratio",
    ],
    "customer": [
        "customer_key",
        "customer_name",
        "customer_account_type",
        "customer_type",
        "customer_category",
        "customer_onboard_date",
        "customer_age_days",
    ],
    "shipment_route": [
        "shipment_mode",
        "account_type",
        "inco_terms",
        "commodity",
        "gross_weight",
        "chargeable_weight",
        "volume_weight",
        "container_count",
        "origin_city",
        "origin_state",
        "origin_country",
        "destination_city",
        "destination_state",
        "destination_country",
    ],
    "operational": [
        "sales_owner",
        "job_no",
        "booking_no",
        "clearance_status",
        "gatein_status",
        "tracking_status",
        "tracking_location",
        "shipping_bill_no",
        "container_no",
    ],
}


def _print_source_config() -> str:
    source_mode = get_source_mode()
    print("source_mode:", source_mode)
    print("risk_main_db:", get_database_name())
    print("risk_main_uri:", get_mongo_uri())
    print("live_db:", get_live_db_name())
    print("live_uri:", get_live_mongo_uri())
    return source_mode


def _coverage_pct(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return round(float(series.notna().mean() * 100.0), 2)


def _feature_domain(name: str) -> str:
    lowered = str(name).lower()
    if lowered in {"invoice_amount", "gross_amount", "terms_days", "days_to_due"}:
        return "invoice"
    if lowered.startswith("customer") or lowered.startswith("cust_"):
        return "customer"
    if "payment" in lowered or lowered in {"paid_amount", "tds_amount", "ytd_exposure"}:
        return "payment"
    if any(
        token in lowered
        for token in [
            "aging",
            "receivable",
            "exposure",
            "capital",
            "realization",
            "unpaid",
        ]
    ):
        return "finance_aging"
    if any(
        token in lowered
        for token in [
            "shipment",
            "origin",
            "destination",
            "route",
            "commodity",
            "weight",
            "container",
            "inco",
            "account_type",
        ]
    ):
        return "shipment_route"
    if any(
        token in lowered
        for token in ["sales", "company", "job", "booking", "tracking", "gate", "clearance"]
    ):
        return "operational"
    return "feature_frame"


def _canonical_domain_summary(canonical: pd.DataFrame) -> list[dict]:
    summary: list[dict] = []
    for domain, columns in CANONICAL_DOMAIN_GROUPS.items():
        present = [column for column in columns if column in canonical.columns]
        if not present:
            summary.append(
                {
                    "domain": domain,
                    "present_columns": 0,
                    "avg_coverage_pct": 0.0,
                    "zero_coverage_columns": [],
                }
            )
            continue
        coverage = {column: _coverage_pct(canonical[column]) for column in present}
        zero_coverage = [column for column, pct in coverage.items() if pct <= 0.0]
        summary.append(
            {
                "domain": domain,
                "present_columns": len(present),
                "avg_coverage_pct": round(sum(coverage.values()) / len(coverage), 2),
                "zero_coverage_columns": zero_coverage,
            }
        )
    return summary


def _feature_quality(scoring_frame: pd.DataFrame, expected_features: list[str]) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    domain_rollup: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"feature_count": 0, "avg_non_null_pct": 0.0, "features_with_data": 0, "zero_coverage_features": 0}
    )
    for feature in expected_features:
        series = scoring_frame[feature] if feature in scoring_frame.columns else pd.Series(dtype=float)
        non_null_pct = _coverage_pct(series) if feature in scoring_frame.columns else 0.0
        unique_non_null = int(series.dropna().nunique()) if feature in scoring_frame.columns else 0
        domain = _feature_domain(feature)
        row = {
            "feature": feature,
            "domain": domain,
            "present_in_frame": feature in scoring_frame.columns,
            "non_null_pct": non_null_pct,
            "unique_non_null": unique_non_null,
        }
        rows.append(row)

        bucket = domain_rollup[domain]
        bucket["feature_count"] += 1
        bucket["avg_non_null_pct"] += non_null_pct
        if non_null_pct > 0:
            bucket["features_with_data"] += 1
        else:
            bucket["zero_coverage_features"] += 1

    summary: list[dict] = []
    for domain, values in sorted(domain_rollup.items()):
        count = int(values["feature_count"])
        summary.append(
            {
                "domain": domain,
                "feature_count": count,
                "features_with_data": int(values["features_with_data"]),
                "zero_coverage_features": int(values["zero_coverage_features"]),
                "avg_non_null_pct": round(float(values["avg_non_null_pct"]) / max(count, 1), 2),
            }
        )
    return rows, summary


def _compare_with_risk_main(query: dict | None, limit: int) -> tuple[list[dict], list[dict]]:
    original_mode = os.environ.get("PRODUCTION_RISK_SOURCE_MODE")
    os.environ["PRODUCTION_RISK_SOURCE_MODE"] = "risk_main"
    try:
        risk_raw = fetch_risk_main_frame(query=query or None, limit=limit)
        risk_canonical = canonicalize_risk_main_frame(risk_raw)
    finally:
        if original_mode is None:
            os.environ.pop("PRODUCTION_RISK_SOURCE_MODE", None)
        else:
            os.environ["PRODUCTION_RISK_SOURCE_MODE"] = original_mode

    live_raw = fetch_risk_main_frame(query=query or None, limit=limit)
    live_canonical = canonicalize_risk_main_frame(live_raw)

    risk_cov = {column: float(risk_canonical[column].notna().mean()) for column in risk_canonical.columns}
    live_cov = {column: float(live_canonical[column].notna().mean()) for column in live_canonical.columns}
    all_cols = sorted(set(risk_cov) | set(live_cov))

    missing_in_live: list[dict] = []
    sparse_in_live: list[dict] = []
    for column in all_cols:
        risk_pct = round(risk_cov.get(column, 0.0) * 100.0, 2)
        live_pct = round(live_cov.get(column, 0.0) * 100.0, 2)
        payload = {"column": column, "risk_main_pct": risk_pct, "live_pct": live_pct}
        if risk_pct >= 80.0 and live_pct <= 5.0:
            missing_in_live.append(payload)
        elif risk_pct >= 80.0 and live_pct < risk_pct - 40.0:
            sparse_in_live.append(payload)
    return missing_in_live, sparse_in_live


def _print_frame_summary(canonical, scoring_frame, validation) -> None:
    print("canonical rows:", len(canonical))
    print("feature rows:", len(scoring_frame))
    print("missing feature count:", validation.missing_count)
    print("domain gaps:", validation.domain_gaps)

    print(canonical.head(10))
    print(canonical.isna().mean().sort_values(ascending=False).head(30))
    print(canonical["delay_days"].describe())
    print(canonical["target"].value_counts(dropna=False, normalize=True))
    print(canonical["label_quality"].value_counts(dropna=False, normalize=True))

    numeric = canonical.select_dtypes(include="number")
    print(numeric.skew(numeric_only=True).sort_values(key=np.abs, ascending=False).head(25))

    suspect = canonical[
        (canonical["invoice_amount"] <= 0)
        | (canonical["aging_total_to_invoice_ratio"].abs() > 5)
        | (canonical["gross_to_invoice_ratio"].abs() > 5)
        | (canonical["paid_to_invoice_ratio"].abs() > 5)
        | (canonical["tds_to_invoice_ratio"].abs() > 5)
    ]
    print(
        suspect[
            [
                "invoice_key",
                "invoice_amount",
                "gross_amount",
                "paid_amount",
                "tds_amount",
                "aging_total",
                "aging_total_to_invoice_ratio",
                "gross_to_invoice_ratio",
                "paid_to_invoice_ratio",
                "tds_to_invoice_ratio",
                "paid_status",
                "label_quality",
            ]
        ].head(50)
    )

    print(canonical.groupby("label_quality")["delay_days"].describe())
    print(canonical.groupby("label_quality")["target"].mean())

    print(
        canonical[
            ["invoice_key", "payment_date", "payment_date_raw", "paid_status", "label_quality"]
        ].head(30)
    )

    core = [
        "invoice_key",
        "customer_key",
        "invoice_date",
        "due_date",
        "invoice_amount",
        "paid_status",
        "shipment_mode",
    ]
    print(canonical[core].isna().mean().sort_values(ascending=False))

    print("canonical domain quality:")
    for item in _canonical_domain_summary(canonical):
        print(item)


def _print_scored_summary(scored) -> None:
    print(scored[["pd", "score", "risk_band"]].describe(include="all"))
    print(scored["risk_band"].value_counts(dropna=False, normalize=True))


def _print_feature_quality(scoring_frame: pd.DataFrame, expected_features: list[str]) -> tuple[list[dict], list[dict]]:
    feature_rows, domain_summary = _feature_quality(scoring_frame, expected_features)
    total = len(feature_rows)
    with_data = sum(1 for row in feature_rows if row["non_null_pct"] > 0.0)
    dense = sum(1 for row in feature_rows if row["non_null_pct"] >= 80.0)
    sparse = [row for row in feature_rows if row["non_null_pct"] < 20.0]
    zero_cov = [row for row in feature_rows if row["non_null_pct"] <= 0.0]
    constant = [row for row in feature_rows if row["unique_non_null"] <= 1]
    low_info = [row for row in feature_rows if row["unique_non_null"] <= 3]

    print("feature quality summary:")
    print(
        {
            "expected_features": total,
            "features_with_data": with_data,
            "dense_features_ge_80pct": dense,
            "sparse_features_lt_20pct": len(sparse),
            "zero_coverage_features": len(zero_cov),
            "constant_or_single_value_features": len(constant),
            "low_information_features_le_3_unique": len(low_info),
        }
    )
    print("feature domain quality:")
    for item in domain_summary:
        print(item)
    print("lowest coverage model features:")
    for item in sorted(feature_rows, key=lambda row: (row["non_null_pct"], row["feature"]))[:40]:
        print(item)
    print("constant / low-information model features:")
    for item in sorted(feature_rows, key=lambda row: (row["unique_non_null"], row["feature"]))[:40]:
        print(item)
    return feature_rows, domain_summary


def _build_query(args) -> dict:
    query: dict = {}
    if args.paid_status:
        query["paidStatus"] = args.paid_status

    invoice_date_query: dict = {}
    if args.invoice_date_from:
        invoice_date_query["$gte"] = datetime.fromisoformat(args.invoice_date_from)
    if args.invoice_date_to:
        invoice_date_query["$lte"] = datetime.fromisoformat(args.invoice_date_to)
    if invoice_date_query:
        query["invoiceDate"] = invoice_date_query
    return query


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the final canonical and feature frames before live scoring.")
    parser.add_argument("--limit", type=int, default=200, help="Number of rows to inspect.")
    parser.add_argument("--paid-status", help="Optional paidStatus filter, e.g. Pending or Paid.")
    parser.add_argument("--invoice-date-from", help="Optional lower invoiceDate bound in YYYY-MM-DD format.")
    parser.add_argument("--invoice-date-to", help="Optional upper invoiceDate bound in YYYY-MM-DD format.")
    parser.add_argument(
        "--allow-risk-main",
        action="store_true",
        help="Allow execution even when source mode is still risk_main.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Write canonical, feature, and scored sample CSVs to the project root.",
    )
    parser.add_argument(
        "--compare-risk-main",
        action="store_true",
        help="Compare live canonical coverage against the risk_main path for the same query and limit.",
    )
    args = parser.parse_args()

    init_env()
    source_mode = _print_source_config()
    if source_mode != "live_collections" and not args.allow_risk_main:
        print("Refusing to continue because source_mode is not live_collections.")
        print("Set PRODUCTION_RISK_SOURCE_MODE=live_collections in the root .env or rerun with --allow-risk-main.")
        return 1

    query = _build_query(args)
    print("query:", query or "{}")
    raw_frame = fetch_risk_main_frame(query=query or None, limit=max(int(args.limit), 1))
    canonical = canonicalize_risk_main_frame(raw_frame)
    scoring_frame = build_risk_main_scoring_frame(raw_frame, history_df=raw_frame)

    artifacts = load_production_artifacts()
    validation = validate_feature_frame(scoring_frame, artifacts["features"])
    scored = score_production_frame(scoring_frame.copy())

    print("raw rows:", len(raw_frame))
    if source_mode == "live_collections":
        print("live diagnostics:", get_live_diagnostics())
        print("live coverage:", compute_live_coverage(raw_frame))

    _print_frame_summary(canonical, scoring_frame, validation)
    feature_rows, feature_domain_summary = _print_feature_quality(scoring_frame, artifacts["features"])
    _print_scored_summary(scored)

    missing_in_live: list[dict] = []
    sparse_in_live: list[dict] = []
    if args.compare_risk_main and source_mode == "live_collections":
        missing_in_live, sparse_in_live = _compare_with_risk_main(query, max(int(args.limit), 1))
        print("risk_main vs live missing-in-live columns:")
        for item in missing_in_live:
            print(item)
        print("risk_main vs live sparse-in-live columns:")
        for item in sparse_in_live:
            print(item)

    if args.export_csv:
        canonical_path = PROJECT_ROOT / "live_canonical_sample.csv"
        feature_path = PROJECT_ROOT / "live_feature_frame_sample.csv"
        scored_path = PROJECT_ROOT / "live_scored_sample.csv"
        feature_quality_path = PROJECT_ROOT / "live_feature_quality.csv"
        domain_quality_path = PROJECT_ROOT / "live_feature_domain_quality.json"
        gap_report_path = PROJECT_ROOT / "live_vs_risk_main_gap_report.json"
        canonical.to_csv(canonical_path, index=False)
        scoring_frame.to_csv(feature_path, index=False)
        scored.to_csv(scored_path, index=False)
        pd.DataFrame(feature_rows).to_csv(feature_quality_path, index=False)
        domain_quality_payload = {
            "canonical_domain_quality": _canonical_domain_summary(canonical),
            "feature_domain_quality": feature_domain_summary,
        }
        domain_quality_path.write_text(json.dumps(domain_quality_payload, indent=2), encoding="utf-8")
        gap_report_path.write_text(
            json.dumps(
                {
                    "query": query,
                    "missing_in_live": missing_in_live,
                    "sparse_in_live": sparse_in_live,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("wrote:", canonical_path)
        print("wrote:", feature_path)
        print("wrote:", scored_path)
        print("wrote:", feature_quality_path)
        print("wrote:", domain_quality_path)
        print("wrote:", gap_report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
