from __future__ import annotations

import heapq

import numpy as np
import pandas as pd


POINT_IN_TIME_FEATURES = [
    "prior_invoice_count",
    "prior_paid_invoice_count",
    "prior_delay_rate",
    "prior_severe_delay_rate",
    "prior_avg_delay_days",
    "prior_max_delay_days",
    "prior_on_time_rate",
    "prior_late_invoice_count",
    "prior_open_invoice_count",
    "prior_avg_invoice_amount",
    "invoice_amount_vs_prior_avg",
    "days_since_last_invoice",
    "recurring_delay_flag",
]


def add_point_in_time_customer_features(
    df: pd.DataFrame,
    customer_col: str = "customer_key",
    source_col: str = "source_system",
    invoice_date_col: str = "invoice_date",
    due_date_col: str = "due_date",
    payment_date_col: str = "payment_date",
    amount_col: str = "invoice_amount",
    delay_col: str = "delay_days",
    severe_delay_threshold: int = 30,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in POINT_IN_TIME_FEATURES:
            out[col] = pd.Series(dtype=float)
        return out

    event_date = pd.to_datetime(out.get(invoice_date_col), errors="coerce")
    if due_date_col in out.columns:
        event_date = event_date.fillna(pd.to_datetime(out[due_date_col], errors="coerce"))
    if payment_date_col in out.columns:
        event_date = event_date.fillna(pd.to_datetime(out[payment_date_col], errors="coerce"))

    group_source = out.get(source_col, pd.Series("unknown", index=out.index)).fillna("unknown").astype(str)
    group_customer = out.get(customer_col, pd.Series("unknown", index=out.index)).fillna("unknown").astype(str)
    invoice_amount = pd.to_numeric(out.get(amount_col), errors="coerce").fillna(0.0)
    payment_date = pd.to_datetime(out.get(payment_date_col), errors="coerce")
    delay_days = pd.to_numeric(out.get(delay_col), errors="coerce").fillna(0.0).clip(lower=0.0)

    defaults = {feature: pd.Series(0.0, index=out.index) for feature in POINT_IN_TIME_FEATURES}
    defaults["recurring_delay_flag"] = pd.Series(0, index=out.index, dtype=int)
    features = defaults

    helper = pd.DataFrame(
        {
            "_event_date": event_date,
            "_source_group": group_source,
            "_customer_group": group_customer,
            "_invoice_amount": invoice_amount,
            "_payment_date": payment_date,
            "_delay_days": delay_days,
            "_row_id": np.arange(len(out)),
        },
        index=out.index,
    )

    helper["_sort_date"] = helper["_event_date"].fillna(pd.Timestamp("1970-01-01"))
    helper["_sort_key"] = helper.index.astype(str)

    for (_, _), group in helper.groupby(["_source_group", "_customer_group"], dropna=False, sort=False):
        group = group.sort_values(["_sort_date", "_sort_key"])
        known_outcomes: list[tuple[pd.Timestamp, float]] = []
        seen_count = 0
        invoice_sum = 0.0
        paid_count = 0
        late_count = 0
        severe_late_count = 0
        on_time_count = 0
        delay_sum = 0.0
        max_delay = 0.0
        previous_invoice_date: pd.Timestamp | None = None

        for idx, row in group.iterrows():
            current_date = row["_event_date"]
            comparison_date = current_date if pd.notna(current_date) else row["_sort_date"]

            while known_outcomes and known_outcomes[0][0] <= comparison_date:
                _, known_delay = heapq.heappop(known_outcomes)
                paid_count += 1
                delay_sum += float(known_delay)
                max_delay = max(max_delay, float(known_delay))
                if float(known_delay) > 0:
                    late_count += 1
                else:
                    on_time_count += 1
                if float(known_delay) > severe_delay_threshold:
                    severe_late_count += 1

            prior_avg_invoice = invoice_sum / seen_count if seen_count else 0.0
            features["prior_invoice_count"].at[idx] = float(seen_count)
            features["prior_paid_invoice_count"].at[idx] = float(paid_count)
            features["prior_late_invoice_count"].at[idx] = float(late_count)
            features["prior_open_invoice_count"].at[idx] = float(max(seen_count - paid_count, 0))
            features["prior_avg_invoice_amount"].at[idx] = float(prior_avg_invoice)
            features["invoice_amount_vs_prior_avg"].at[idx] = (
                float(row["_invoice_amount"]) / prior_avg_invoice if prior_avg_invoice > 0 else 0.0
            )
            features["prior_delay_rate"].at[idx] = float(late_count / paid_count) if paid_count else 0.0
            features["prior_severe_delay_rate"].at[idx] = float(severe_late_count / paid_count) if paid_count else 0.0
            features["prior_avg_delay_days"].at[idx] = float(delay_sum / paid_count) if paid_count else 0.0
            features["prior_max_delay_days"].at[idx] = float(max_delay) if paid_count else 0.0
            features["prior_on_time_rate"].at[idx] = float(on_time_count / paid_count) if paid_count else 0.0
            features["recurring_delay_flag"].at[idx] = int(late_count >= 2)
            if previous_invoice_date is not None and pd.notna(current_date):
                features["days_since_last_invoice"].at[idx] = float((current_date - previous_invoice_date).days)
            else:
                features["days_since_last_invoice"].at[idx] = 0.0

            seen_count += 1
            invoice_sum += float(row["_invoice_amount"])
            if pd.notna(current_date):
                previous_invoice_date = current_date

            row_payment_date = row["_payment_date"]
            if pd.notna(row_payment_date):
                heapq.heappush(known_outcomes, (row_payment_date, float(row["_delay_days"])))

    for col, values in features.items():
        out[col] = values
    return out
