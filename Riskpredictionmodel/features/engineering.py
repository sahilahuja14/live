from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

from .point_in_time import add_point_in_time_customer_features
from ..pipeline.utils import safe_ratio


def _safe_text_col(series: pd.Series, default: str = "Unknown", case: str = "none") -> pd.Series:
    output = series.fillna(default).astype(str).str.strip()
    output = output.replace({"": default})
    if case == "lower":
        output = output.str.lower()
    elif case == "upper":
        output = output.str.upper()
    elif case == "title":
        output = output.str.title()
    return output


def _safe_num_col(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _add_calendar_features(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    values = pd.to_datetime(df[column], errors="coerce")
    iso = values.dt.isocalendar()
    month = values.dt.month.fillna(0)
    weekday = values.dt.weekday.fillna(0)
    payload = {
        f"{prefix}_year": values.dt.year.fillna(0).astype(int),
        f"{prefix}_quarter": values.dt.quarter.fillna(0).astype(int),
        f"{prefix}_month": month.astype(int),
        f"{prefix}_week": iso.week.fillna(0).astype(int),
        f"{prefix}_day": values.dt.day.fillna(0).astype(int),
        f"{prefix}_weekday": weekday.astype(int),
        f"{prefix}_day_of_year": values.dt.dayofyear.fillna(0).astype(int),
        f"{prefix}_is_month_start": values.dt.is_month_start.fillna(False).astype(int),
        f"{prefix}_is_month_end": values.dt.is_month_end.fillna(False).astype(int),
        f"{prefix}_is_quarter_start": values.dt.is_quarter_start.fillna(False).astype(int),
        f"{prefix}_is_quarter_end": values.dt.is_quarter_end.fillna(False).astype(int),
        f"{prefix}_is_weekend": values.dt.weekday.fillna(0).isin([5, 6]).astype(int),
        f"{prefix}_fortnight": pd.Series(np.where(values.dt.day.fillna(99) <= 15, 1, 2), index=df.index),
        f"{prefix}_month_sin": np.sin(2 * np.pi * month / 12.0),
        f"{prefix}_month_cos": np.cos(2 * np.pi * month / 12.0),
        f"{prefix}_weekday_sin": np.sin(2 * np.pi * weekday / 7.0),
        f"{prefix}_weekday_cos": np.cos(2 * np.pi * weekday / 7.0),
    }
    return pd.concat([df, pd.DataFrame(payload, index=df.index)], axis=1).copy()


def _add_frequency_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        counts = out[column].fillna("Unknown").astype(str).value_counts(dropna=False)
        normalized = out[column].fillna("Unknown").astype(str)
        out[f"{column}_freq"] = normalized.map(counts).fillna(0).astype(float)
        out[f"{column}_share"] = normalized.map(counts / max(len(out), 1)).fillna(0.0).astype(float)
    return out


def _add_additional_customer_history_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "prior_invoice_amount_sum": 0.0,
        "prior_invoice_amount_std": 0.0,
        "prior_terms_avg": 0.0,
        "prior_terms_std": 0.0,
        "prior_paid_ratio_avg": 0.0,
        "prior_unpaid_ratio_avg": 0.0,
        "prior_exposure_avg": 0.0,
        "prior_exposure_max": 0.0,
        "prior_aging_total_avg": 0.0,
        "prior_aging_total_max": 0.0,
        "prior_partial_payment_rate": 0.0,
        "prior_installment_count_avg": 0.0,
        "prior_mode_diversity": 0.0,
        "prior_inco_diversity": 0.0,
        "prior_sales_owner_diversity": 0.0,
        "prior_company_diversity": 0.0,
        "prior_route_diversity": 0.0,
        "prior_top_mode_share": 0.0,
        "prior_top_inco_share": 0.0,
        "prior_avg_weight_discrepancy": 0.0,
        "prior_max_weight_discrepancy": 0.0,
    }
    out = out.assign(**defaults)

    helper = out.sort_values(["customer_key", "invoice_date", "invoice_key"]).reset_index()
    for _, group in helper.groupby("customer_key", dropna=False, sort=False):
        amounts: list[float] = []
        terms: list[float] = []
        paid_ratios: list[float] = []
        unpaid_ratios: list[float] = []
        exposures: list[float] = []
        aging_totals: list[float] = []
        weight_discrepancies: list[float] = []
        partial_flags: list[int] = []
        installment_counts: list[float] = []
        modes = Counter()
        incos = Counter()
        sales_owners = Counter()
        companies = Counter()
        routes = Counter()

        for _, row in group.iterrows():
            idx = row["index"]
            seen_count = len(amounts)
            if seen_count:
                out.at[idx, "prior_invoice_amount_sum"] = float(np.sum(amounts))
                out.at[idx, "prior_invoice_amount_std"] = float(np.std(amounts))
                out.at[idx, "prior_terms_avg"] = float(np.mean(terms))
                out.at[idx, "prior_terms_std"] = float(np.std(terms))
                out.at[idx, "prior_paid_ratio_avg"] = float(np.mean(paid_ratios))
                out.at[idx, "prior_unpaid_ratio_avg"] = float(np.mean(unpaid_ratios))
                out.at[idx, "prior_exposure_avg"] = float(np.mean(exposures))
                out.at[idx, "prior_exposure_max"] = float(np.max(exposures))
                out.at[idx, "prior_aging_total_avg"] = float(np.mean(aging_totals))
                out.at[idx, "prior_aging_total_max"] = float(np.max(aging_totals))
                out.at[idx, "prior_partial_payment_rate"] = float(np.mean(partial_flags))
                out.at[idx, "prior_installment_count_avg"] = float(np.mean(installment_counts))
                out.at[idx, "prior_mode_diversity"] = float(len(modes))
                out.at[idx, "prior_inco_diversity"] = float(len(incos))
                out.at[idx, "prior_sales_owner_diversity"] = float(len(sales_owners))
                out.at[idx, "prior_company_diversity"] = float(len(companies))
                out.at[idx, "prior_route_diversity"] = float(len(routes))
                out.at[idx, "prior_top_mode_share"] = float(max(modes.values()) / seen_count)
                out.at[idx, "prior_top_inco_share"] = float(max(incos.values()) / seen_count)
                out.at[idx, "prior_avg_weight_discrepancy"] = float(np.mean(weight_discrepancies))
                out.at[idx, "prior_max_weight_discrepancy"] = float(np.max(weight_discrepancies))

            amounts.append(float(row["invoice_amount"]))
            terms.append(float(row["terms_days"]))
            paid_ratios.append(float(row["paid_to_invoice_ratio"]))
            unpaid_ratios.append(float(row["unpaid_to_invoice_ratio"]))
            exposures.append(float(row["exposure_to_invoice_ratio"]))
            aging_totals.append(float(row["aging_total_to_invoice_ratio"]))
            weight_discrepancies.append(float(row["weight_discrepancy"]))
            partial_flags.append(int(bool(row["partial_payment_flag"])))
            installment_counts.append(float(row["payment_installment_count"]))
            modes[str(row["shipment_mode"])] += 1
            incos[str(row["inco_terms"])] += 1
            sales_owners[str(row["sales_owner"])] += 1
            companies[str(row["company"])] += 1
            route_key = f"{row['origin_country']}::{row['destination_country']}"
            routes[route_key] += 1

    return out


def _terms_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(-1.0)
    buckets = pd.cut(
        numeric,
        bins=[-2, 0, 15, 30, 45, 60, 90, 3650],
        labels=["missing", "0_15", "16_30", "31_45", "46_60", "61_90", "91_plus"],
    )
    return buckets.astype(str).replace({"nan": "missing"})


def _amount_bucket(series: pd.Series, q: int = 8) -> pd.Series:
    ranked = pd.to_numeric(series, errors="coerce").fillna(0.0).rank(method="first")
    try:
        buckets = pd.qcut(ranked, q=q, duplicates="drop")
        return buckets.astype(str).replace({"nan": "all"})
    except ValueError:
        return pd.Series("all", index=series.index, dtype="object")


def _compose_key(*series_list: pd.Series) -> pd.Series:
    parts = [_safe_text_col(series.astype("object"), default="Unknown") for series in series_list]
    output = parts[0]
    for series in parts[1:]:
        output = output + "|" + series
    return output


def _add_entity_history_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "route_key" not in out.columns:
        out["route_key"] = out["origin_country"].fillna("Unknown") + "->" + out["destination_country"].fillna("Unknown")

    out["terms_bucket"] = _terms_bucket(out.get("terms_days", pd.Series(index=out.index, dtype=float)))
    out["amount_bucket"] = _amount_bucket(out.get("invoice_amount", pd.Series(index=out.index, dtype=float)))
    out["customer_sales_key"] = _compose_key(out["customer_key"], out["sales_owner"])
    out["customer_company_key"] = _compose_key(out["customer_key"], out["company"])
    out["sales_company_key"] = _compose_key(out["sales_owner"], out["company"])
    out["route_company_key"] = _compose_key(out["route_key"], out["company"])
    out["customer_currency_key"] = _compose_key(out["customer_key"], out["currency"])
    out["customer_terms_key"] = _compose_key(out["customer_key"], out["terms_bucket"])

    specs = [
        ("customer_key", "customer_hist"),
        ("sales_owner", "sales_hist"),
        ("company", "company_hist"),
        ("route_key", "route_hist"),
        ("shipment_mode", "mode_hist"),
        ("currency", "currency_hist"),
        ("document_type", "doc_hist"),
        ("terms_bucket", "terms_hist"),
        ("amount_bucket", "amount_hist"),
        ("customer_sales_key", "cust_sales_hist"),
        ("customer_company_key", "cust_company_hist"),
        ("sales_company_key", "sales_company_hist"),
        ("route_company_key", "route_company_hist"),
        ("customer_currency_key", "cust_currency_hist"),
        ("customer_terms_key", "cust_terms_hist"),
    ]
    positions = {idx: pos for pos, idx in enumerate(out.index)}
    payload: dict[str, np.ndarray] = {}
    for _, prefix in specs:
        for suffix in [
            "count",
            "delay_rate",
            "avg_delay",
            "recent3_delay_rate",
            "recent5_delay_rate",
            "recent3_avg_delay",
        ]:
            payload[f"{prefix}_{suffix}"] = np.zeros(len(out), dtype=float)

    stats = {
        name: defaultdict(lambda: {"count": 0, "pos": 0, "delay_sum": 0.0, "recent": deque(maxlen=5)})
        for name, _ in specs
    }
    ordered = out.sort_values(["invoice_date", "invoice_key"], na_position="last")
    for idx, row in ordered.iterrows():
        pos = positions[idx]
        for column, prefix in specs:
            key = str(row.get(column)) if pd.notna(row.get(column)) else "Unknown"
            state = stats[column][key]
            recent = list(state["recent"])
            recent3 = recent[-3:]
            recent5 = recent[-5:]
            payload[f"{prefix}_count"][pos] = float(state["count"])
            payload[f"{prefix}_delay_rate"][pos] = float(state["pos"] / state["count"]) if state["count"] else 0.0
            payload[f"{prefix}_avg_delay"][pos] = float(state["delay_sum"] / state["count"]) if state["count"] else 0.0
            payload[f"{prefix}_recent3_delay_rate"][pos] = float(np.mean([item[0] for item in recent3])) if recent3 else 0.0
            payload[f"{prefix}_recent5_delay_rate"][pos] = float(np.mean([item[0] for item in recent5])) if recent5 else 0.0
            payload[f"{prefix}_recent3_avg_delay"][pos] = float(np.mean([item[1] for item in recent3])) if recent3 else 0.0

        target_value = pd.to_numeric(row.get("target"), errors="coerce")
        delay_value = pd.to_numeric(row.get("delay_days"), errors="coerce")
        target = int(0 if pd.isna(target_value) else target_value)
        delay = float(0.0 if pd.isna(delay_value) else delay_value)
        for column, _ in specs:
            key = str(row.get(column)) if pd.notna(row.get(column)) else "Unknown"
            state = stats[column][key]
            state["count"] += 1
            state["pos"] += target
            state["delay_sum"] += delay
            state["recent"].append((target, delay))

    out = pd.concat([out, pd.DataFrame(payload, index=out.index)], axis=1)
    interaction_payload = {}
    for left, right in [
        ("customer_hist_delay_rate", "invoice_amount_log"),
        ("customer_hist_recent3_delay_rate", "terms_days"),
        ("sales_hist_delay_rate", "invoice_amount_log"),
        ("company_hist_delay_rate", "terms_days"),
        ("route_hist_delay_rate", "exposure_to_invoice_ratio"),
        ("cust_company_hist_delay_rate", "invoice_amount_log"),
        ("cust_sales_hist_recent3_delay_rate", "days_to_due"),
        ("sales_company_hist_delay_rate", "working_capital_stress"),
    ]:
        interaction_payload[f"{left}__x__{right}"] = _safe_num_col(out[left]) * _safe_num_col(out[right])
    return pd.concat([out, pd.DataFrame(interaction_payload, index=out.index)], axis=1)


def build_risk_main_feature_frame(canonical_df: pd.DataFrame) -> pd.DataFrame:
    if canonical_df.empty:
        return canonical_df.copy()

    out = canonical_df.copy()
    for text_column, case in {
        "company": "upper",
        "currency": "upper",
        "sales_owner": "title",
        "document_type": "title",
        "shipment_mode": "lower",
        "account_type": "title",
        "inco_terms": "upper",
        "commodity": "title",
        "customer_account_type": "title",
        "customer_type": "title",
        "customer_category": "title",
        "origin_city": "title",
        "origin_state": "title",
        "origin_country": "title",
        "destination_city": "title",
        "destination_state": "title",
        "destination_country": "title",
        "clearance_status": "title",
        "gatein_status": "title",
        "tracking_status": "title",
        "tracking_location": "title",
    }.items():
        if text_column in out.columns:
            out[text_column] = _safe_text_col(out[text_column], case=case)

    numeric_columns = [
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
        "payment_installment_count",
        "gross_weight",
        "chargeable_weight",
        "volume_weight",
        "container_count",
        "days_to_due",
        "execution_gap_days",
        "customer_age_days",
        "weight_discrepancy",
        "unpaid_amount",
        "aging_total",
        "gross_to_invoice_ratio",
        "paid_to_invoice_ratio",
        "tds_to_invoice_ratio",
        "exposure_to_invoice_ratio",
        "aging_total_to_invoice_ratio",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = _safe_num_col(out[column])

    invoice_amount_log = np.log1p(out["invoice_amount"].clip(lower=0))
    gross_amount_log = np.log1p(out["gross_amount"].clip(lower=0))
    paid_amount_log = np.log1p(out["paid_amount"].clip(lower=0))
    ytd_exposure_log = np.log1p(out["ytd_exposure"].clip(lower=0))
    aging_total_log = np.log1p(out["aging_total"].clip(lower=0))
    gross_weight_log = np.log1p(out["gross_weight"].clip(lower=0))
    chargeable_weight_log = np.log1p(out["chargeable_weight"].clip(lower=0))
    volume_weight_log = np.log1p(out["volume_weight"].clip(lower=0))

    unpaid_to_invoice_ratio = pd.Series(
        [safe_ratio(unpaid, amount) for unpaid, amount in zip(out["unpaid_amount"], out["invoice_amount"])],
        index=out.index,
    )
    amount_gap = out["gross_amount"] - out["invoice_amount"]
    amount_gap_ratio = pd.Series(
        [safe_ratio(gap, amount) for gap, amount in zip(amount_gap, out["invoice_amount"])],
        index=out.index,
    )
    cash_gap = out["invoice_amount"] - out["paid_amount"]
    cash_gap_ratio = pd.Series(
        [safe_ratio(gap, amount) for gap, amount in zip(cash_gap, out["invoice_amount"])],
        index=out.index,
    )
    exposure_gap = out["ytd_exposure"] - out["invoice_amount"]
    exposure_gap_ratio = pd.Series(
        [safe_ratio(gap, amount) for gap, amount in zip(exposure_gap, out["invoice_amount"])],
        index=out.index,
    )
    aging_not_due_ratio = pd.Series(
        [safe_ratio(v, total) for v, total in zip(out["aging_not_due"], out["aging_total"])],
        index=out.index,
    )
    aging_0_15_ratio = pd.Series(
        [safe_ratio(v, total) for v, total in zip(out["aging_0_15"], out["aging_total"])],
        index=out.index,
    )
    aging_16_30_ratio = pd.Series(
        [safe_ratio(v, total) for v, total in zip(out["aging_16_30"], out["aging_total"])],
        index=out.index,
    )
    aging_31_45_ratio = pd.Series(
        [safe_ratio(v, total) for v, total in zip(out["aging_31_45"], out["aging_total"])],
        index=out.index,
    )
    aging_46_60_ratio = pd.Series(
        [safe_ratio(v, total) for v, total in zip(out["aging_46_60"], out["aging_total"])],
        index=out.index,
    )
    aging_60_90_ratio = pd.Series(
        [safe_ratio(v, total) for v, total in zip(out["aging_60_90"], out["aging_total"])],
        index=out.index,
    )
    aging_above_90_ratio = pd.Series(
        [safe_ratio(v, total) for v, total in zip(out["aging_above_90"], out["aging_total"])],
        index=out.index,
    )
    mild_aging_share = aging_0_15_ratio + aging_16_30_ratio + aging_31_45_ratio
    severe_aging_share = aging_46_60_ratio + aging_60_90_ratio + aging_above_90_ratio
    receivable_pressure_index = out["aging_total_to_invoice_ratio"] + out["exposure_to_invoice_ratio"] + unpaid_to_invoice_ratio
    working_capital_stress = severe_aging_share + cash_gap_ratio + out["tds_to_invoice_ratio"]
    realization_efficiency = out["paid_to_invoice_ratio"] - out["aging_total_to_invoice_ratio"]
    weight_density_ratio = pd.Series(
        [safe_ratio(v, g) for v, g in zip(out["volume_weight"], out["gross_weight"])],
        index=out.index,
    )
    chargeable_to_gross_ratio = pd.Series(
        [safe_ratio(c, g) for c, g in zip(out["chargeable_weight"], out["gross_weight"])],
        index=out.index,
    )
    container_intensity = pd.Series(
        [safe_ratio(c, amount) for c, amount in zip(out["container_count"], out["invoice_amount"])],
        index=out.index,
    )
    execution_to_due_gap = out["days_to_due"] - out["execution_gap_days"].fillna(0.0)

    derived_payload = {
        "invoice_amount_log": invoice_amount_log,
        "gross_amount_log": gross_amount_log,
        "paid_amount_log": paid_amount_log,
        "ytd_exposure_log": ytd_exposure_log,
        "aging_total_log": aging_total_log,
        "gross_weight_log": gross_weight_log,
        "chargeable_weight_log": chargeable_weight_log,
        "volume_weight_log": volume_weight_log,
        "unpaid_to_invoice_ratio": unpaid_to_invoice_ratio,
        "amount_gap": amount_gap,
        "amount_gap_ratio": amount_gap_ratio,
        "cash_gap": cash_gap,
        "cash_gap_ratio": cash_gap_ratio,
        "exposure_gap": exposure_gap,
        "exposure_gap_ratio": exposure_gap_ratio,
        "aging_not_due_ratio": aging_not_due_ratio,
        "aging_0_15_ratio": aging_0_15_ratio,
        "aging_16_30_ratio": aging_16_30_ratio,
        "aging_31_45_ratio": aging_31_45_ratio,
        "aging_46_60_ratio": aging_46_60_ratio,
        "aging_60_90_ratio": aging_60_90_ratio,
        "aging_above_90_ratio": aging_above_90_ratio,
        "mild_aging_share": mild_aging_share,
        "severe_aging_share": severe_aging_share,
        "receivable_pressure_index": receivable_pressure_index,
        "working_capital_stress": working_capital_stress,
        "realization_efficiency": realization_efficiency,
        "weight_density_ratio": weight_density_ratio,
        "chargeable_to_gross_ratio": chargeable_to_gross_ratio,
        "container_intensity": container_intensity,
        "execution_to_due_gap": execution_to_due_gap,
        "is_iata_document": out["document_type"].eq("Iata").astype(int),
        "is_zipa_company": out["company"].eq("ZIPA").astype(int),
        "has_paid_amount": out["paid_amount"].gt(0).astype(int),
        "has_ytd_exposure": out["ytd_exposure"].gt(0).astype(int),
        "has_execution_gap": out["execution_gap_days"].fillna(0).ne(0).astype(int),
        "has_tracking_status": out["tracking_status"].ne("Unknown").astype(int),
        "has_clearance_signal": out["clearance_status"].ne("Unknown").astype(int),
        "has_gatein_signal": out["gatein_status"].ne("Unknown").astype(int),
        "is_partial_payment": out["partial_payment_flag"].astype(int),
        "has_multiple_installments": out["payment_installment_count"].gt(1).astype(int),
        "is_air_mode": out["shipment_mode"].eq("air").astype(int),
        "is_ocean_mode": out["shipment_mode"].eq("ocean").astype(int),
    }
    out = pd.concat([out, pd.DataFrame(derived_payload, index=out.index)], axis=1).copy()

    for column, prefix in (("invoice_date", "invoice"), ("due_date", "due"), ("execution_date", "execution")):
        out = _add_calendar_features(out, column, prefix)

    out = add_point_in_time_customer_features(
        out,
        customer_col="customer_key",
        source_col="source_system",
        invoice_date_col="invoice_date",
        due_date_col="due_date",
        payment_date_col="payment_date",
        amount_col="invoice_amount",
        delay_col="delay_days",
        severe_delay_threshold=30,
    )
    out = _add_additional_customer_history_features(out)
    out = _add_frequency_features(
        out,
        [
            "company",
            "currency",
            "sales_owner",
            "document_type",
            "shipment_mode",
            "account_type",
            "inco_terms",
            "commodity",
            "customer_account_type",
            "customer_type",
            "customer_category",
            "origin_country",
            "destination_country",
            "clearance_status",
            "gatein_status",
            "tracking_status",
        ],
    )

    out["route_key"] = out["origin_country"].fillna("Unknown") + "->" + out["destination_country"].fillna("Unknown")
    out["domestic_route_flag"] = out["origin_country"].eq(out["destination_country"]).astype(int)
    out["origin_destination_same_city_flag"] = out["origin_city"].eq(out["destination_city"]).astype(int)
    out = _add_entity_history_risk_features(out)
    out["customer_invoice_amount_x_pressure"] = out["invoice_amount"] * out["receivable_pressure_index"]
    out["customer_invoice_amount_x_delay_history"] = out["invoice_amount"] * out["prior_avg_delay_days"]
    out["exposure_x_delay_rate"] = out["ytd_exposure"] * out["prior_delay_rate"]
    out["severity_x_recurring_flag"] = out["severe_aging_share"] * out["recurring_delay_flag"]
    return out.copy()

