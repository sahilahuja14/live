from __future__ import annotations

import pandas as pd

from ..features.registry import AMOUNT_COL, CUSTOMER_ID_COL, DUE_DATE_COL, PAYMENT_DATE_COL


CUSTOMER_HISTORY_PROJECTION = {
    CUSTOMER_ID_COL: 1,
    PAYMENT_DATE_COL: 1,
    "dueDate": 1,
    DUE_DATE_COL: 1,
    AMOUNT_COL: 1,
}

CUSTOMER_AGGREGATE_DEFAULTS = {
    "customer_total_invoices": 0,
    "customer_delayed_invoices": 0,
    "customer_avg_invoice": 0.0,
    "customer_invoice_std": 0.0,
    "customer_avg_delay_days": 0.0,
    "customer_max_delay_days": 0.0,
    "customer_delay_rate": 0.0,
}


def customer_ids_from_frame(df: pd.DataFrame) -> list[str]:
    if df.empty or CUSTOMER_ID_COL not in df.columns:
        return []
    return (
        df[CUSTOMER_ID_COL]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def build_customer_history_aggregates(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty or CUSTOMER_ID_COL not in history_df.columns or PAYMENT_DATE_COL not in history_df.columns:
        return pd.DataFrame()

    frame = history_df.copy()
    due_col = "dueDate" if "dueDate" in frame.columns else DUE_DATE_COL if DUE_DATE_COL in frame.columns else None
    if due_col is None:
        return pd.DataFrame()

    frame = frame[frame[PAYMENT_DATE_COL].notna()].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["delay_days"] = (
        pd.to_datetime(frame[PAYMENT_DATE_COL], errors="coerce")
        - pd.to_datetime(frame[due_col], errors="coerce")
    ).dt.days
    amount_series = frame[AMOUNT_COL] if AMOUNT_COL in frame.columns else pd.Series(0.0, index=frame.index)
    frame[AMOUNT_COL] = pd.to_numeric(amount_series, errors="coerce").fillna(0.0)
    frame["is_delayed"] = pd.to_numeric(frame["delay_days"], errors="coerce").fillna(0.0).gt(0).astype(int)

    grouped = (
        frame.groupby(CUSTOMER_ID_COL, as_index=False)
        .agg(
            customer_total_invoices=(CUSTOMER_ID_COL, "size"),
            customer_delayed_invoices=("is_delayed", "sum"),
            customer_avg_invoice=(AMOUNT_COL, "mean"),
            customer_invoice_std=(AMOUNT_COL, "std"),
            customer_avg_delay_days=("delay_days", "mean"),
            customer_max_delay_days=("delay_days", "max"),
        )
    )
    grouped["customer_delay_rate"] = (
        grouped["customer_delayed_invoices"]
        / grouped["customer_total_invoices"].replace({0: pd.NA})
    )
    for column, default in CUSTOMER_AGGREGATE_DEFAULTS.items():
        grouped[column] = pd.to_numeric(grouped[column], errors="coerce").fillna(default)
    return grouped


def merge_customer_history_aggregates(df: pd.DataFrame, aggregates: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()
    if out.empty or CUSTOMER_ID_COL not in out.columns:
        return out
    if aggregates is not None and not aggregates.empty:
        out = out.merge(aggregates, on=CUSTOMER_ID_COL, how="left")
    for column, default in CUSTOMER_AGGREGATE_DEFAULTS.items():
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
        else:
            out[column] = default
    return out


def add_customer_aggregates(df):
    frame = df.copy()
    if frame.empty or CUSTOMER_ID_COL not in frame.columns:
        return frame

    grouped = (
        frame.groupby(CUSTOMER_ID_COL, as_index=False)
        .agg(
            customer_total_invoices=("is_delayed", "count"),
            customer_delayed_invoices=("is_delayed", "sum"),
            customer_avg_invoice=(AMOUNT_COL, "mean"),
            customer_max_invoice=(AMOUNT_COL, "max"),
            customer_invoice_std=(AMOUNT_COL, "std"),
            customer_avg_execution_gap=("execution_gap_days", "mean"),
            customer_avg_delay_days=("delay_days", "mean"),
            customer_max_delay_days=("delay_days", "max"),
            customer_total_exposure=(AMOUNT_COL, "sum"),
        )
    )
    grouped["customer_delay_rate"] = (
        grouped["customer_delayed_invoices"]
        / grouped["customer_total_invoices"].replace({0: pd.NA})
    )
    grouped["customer_recurring_delay_flag"] = (grouped["customer_delayed_invoices"] >= 2).astype(int)
    grouped["customer_heavy_defaulter_flag"] = (grouped["customer_delay_rate"] > 0.5).astype(int)
    severity_mean = grouped["customer_avg_delay_days"].mean()
    grouped["customer_severity_ratio"] = (
        grouped["customer_avg_delay_days"] / severity_mean if pd.notna(severity_mean) and severity_mean != 0 else 0.0
    )

    out = frame.merge(grouped, on=CUSTOMER_ID_COL, how="left")
    out["customer_invoice_std"] = out["customer_invoice_std"].fillna(0)
    out["customer_avg_execution_gap"] = out["customer_avg_execution_gap"].fillna(0)
    out["customer_avg_delay_days"] = out["customer_avg_delay_days"].fillna(0)
    out["customer_max_delay_days"] = out["customer_max_delay_days"].fillna(0)
    out["customer_total_exposure"] = out["customer_total_exposure"].fillna(0)
    out["customer_delay_rate"] = out["customer_delay_rate"].fillna(0)
    out["customer_severity_ratio"] = out["customer_severity_ratio"].replace(
        [float("inf"), -float("inf")],
        0,
    ).fillna(0)
    return out
