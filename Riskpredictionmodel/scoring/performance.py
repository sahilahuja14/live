from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..logging_config import get_logger
from ..pipeline.risk_map import display_feature_name


logger = get_logger(__name__)
_FIXED_BINS = np.linspace(0.0, 1.0, 11)
_BAND_ORDER = {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2}


def _safe_float(value: Any, *, digits: int | None = 6) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    if digits is None:
        return numeric
    return round(numeric, digits)


def _safe_binary_frame(scored_df: pd.DataFrame) -> pd.DataFrame:
    if scored_df.empty:
        return pd.DataFrame(columns=["pd", "is_delayed", "approval", "approval_threshold", "risk_band", "score"])

    frame = scored_df.copy()
    frame["pd"] = pd.to_numeric(frame.get("pd"), errors="coerce")
    frame["is_delayed"] = pd.to_numeric(frame.get("is_delayed"), errors="coerce")
    valid = frame["pd"].notna() & frame["is_delayed"].notna()
    filtered = frame.loc[valid].copy()
    if filtered.empty:
        return filtered

    filtered["pd"] = filtered["pd"].clip(1e-6, 1 - 1e-6)
    filtered["is_delayed"] = (filtered["is_delayed"] > 0).astype(int)
    filtered["approval"] = filtered.get("approval", pd.Series(index=filtered.index, dtype=object)).fillna("").astype(str)
    filtered["approval_threshold"] = pd.to_numeric(filtered.get("approval_threshold"), errors="coerce")
    filtered["risk_band"] = filtered.get("risk_band", pd.Series(index=filtered.index, dtype=object)).fillna("Unknown").astype(str)
    filtered["score"] = pd.to_numeric(filtered.get("score"), errors="coerce")
    return filtered


def _safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray | None = None, y_pred: np.ndarray | None = None) -> float | None:
    try:
        if y_score is not None:
            value = metric_fn(y_true, y_score)
        elif y_pred is not None:
            value = metric_fn(y_true, y_pred)
        else:
            value = metric_fn(y_true)
    except Exception:
        return None
    return _safe_float(value)


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    if positives <= 0 or negatives <= 0:
        return None
    order = np.argsort(y_score)
    sorted_true = y_true[order]
    pos_cdf = np.cumsum(sorted_true == 1) / positives
    neg_cdf = np.cumsum(sorted_true == 0) / negatives
    return _safe_float(np.max(np.abs(pos_cdf - neg_cdf)))


def _derive_predictions(frame: pd.DataFrame, default_threshold: float) -> tuple[np.ndarray, list[float]]:
    approval = frame.get("approval", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.strip().str.lower()
    approval_mask = approval.isin(["approve", "reject"])
    threshold_values = sorted(
        {
            round(float(value), 6)
            for value in pd.to_numeric(frame.get("approval_threshold"), errors="coerce").dropna().tolist()
            if np.isfinite(value)
        }
    )
    if approval_mask.any():
        y_pred = approval.eq("reject").astype(int).to_numpy()
    else:
        y_pred = (frame["pd"].to_numpy() >= float(default_threshold)).astype(int)
        threshold_values = [round(float(default_threshold), 6)]
    return y_pred, threshold_values


def _confusion_payload(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "matrix": [
                {"actual": "Delayed", "predicted": "Reject", "count": 0, "rate": 0.0},
                {"actual": "Delayed", "predicted": "Approve", "count": 0, "rate": 0.0},
                {"actual": "On Time", "predicted": "Reject", "count": 0, "rate": 0.0},
                {"actual": "On Time", "predicted": "Approve", "count": 0, "rate": 0.0},
            ],
        }
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = max(int(len(y_true)), 1)

    def _rate(value: int) -> float:
        return round(float(value / total), 6)

    return {
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "matrix": [
            {"actual": "Delayed", "predicted": "Reject", "count": true_positive, "rate": _rate(true_positive)},
            {"actual": "Delayed", "predicted": "Approve", "count": false_negative, "rate": _rate(false_negative)},
            {"actual": "On Time", "predicted": "Reject", "count": false_positive, "rate": _rate(false_positive)},
            {"actual": "On Time", "predicted": "Approve", "count": true_negative, "rate": _rate(true_negative)},
        ],
    }


def _calibration_error(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    work = frame.copy()
    work["prob_bin"] = pd.cut(work["pd"], bins=_FIXED_BINS, include_lowest=True, duplicates="drop")
    rows: list[float] = []
    total = float(len(work))
    for _, bucket in work.groupby("prob_bin", observed=False):
        if bucket.empty:
            continue
        rows.append(abs(float(bucket["pd"].mean()) - float(bucket["is_delayed"].mean())) * (len(bucket) / total))
    return _safe_float(sum(rows))


def _calibration_bins(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    work = frame.copy()
    work["prob_bin"] = pd.cut(work["pd"], bins=_FIXED_BINS, include_lowest=True, duplicates="drop")
    rows: list[dict] = []
    for interval, bucket in work.groupby("prob_bin", observed=False):
        if bucket.empty or interval is None:
            continue
        rows.append(
            {
                "bucket": f"{int(interval.left * 100)}-{int(interval.right * 100)}%",
                "count": int(len(bucket)),
                "avg_pd": _safe_float(bucket["pd"].mean()),
                "avg_pd_pct": _safe_float(bucket["pd"].mean() * 100, digits=2),
                "actual_rate": _safe_float(bucket["is_delayed"].mean()),
                "actual_rate_pct": _safe_float(bucket["is_delayed"].mean() * 100, digits=2),
            }
        )
    return rows


def _threshold_curve(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    y_true = frame["is_delayed"].to_numpy(dtype=int)
    y_score = frame["pd"].to_numpy(dtype=float)
    rows: list[dict] = []
    for threshold in np.linspace(0.05, 0.95, 19):
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": _safe_metric(precision_score, y_true, y_pred=y_pred),
                "recall": _safe_metric(recall_score, y_true, y_pred=y_pred),
                "f1": _safe_metric(f1_score, y_true, y_pred=y_pred),
                "specificity": _safe_float(
                    np.sum((y_true == 0) & (y_pred == 0)) / max(np.sum(y_true == 0), 1),
                ),
                "reject_rate": _safe_float(float(np.mean(y_pred))),
                "approve_rate": _safe_float(float(1.0 - np.mean(y_pred))),
            }
        )
    return rows


def _pd_histogram(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    work = frame.copy()
    work["prob_bin"] = pd.cut(work["pd"], bins=_FIXED_BINS, include_lowest=True, duplicates="drop")
    rows: list[dict] = []
    for interval, bucket in work.groupby("prob_bin", observed=False):
        if bucket.empty or interval is None:
            continue
        rows.append(
            {
                "bucket": f"{int(interval.left * 100)}-{int(interval.right * 100)}%",
                "count": int(len(bucket)),
                "delayed": int(bucket["is_delayed"].sum()),
            }
        )
    return rows


def _performance_by_group(frame: pd.DataFrame, column: str, order: dict[str, int] | None = None) -> list[dict]:
    if frame.empty or column not in frame.columns:
        return []
    grouped = frame.groupby(column, dropna=False)
    rows: list[dict] = []
    for raw_name, bucket in grouped:
        name = str(raw_name or "Unknown").strip() or "Unknown"
        reject_rate = bucket.get("approval", pd.Series(dtype=object)).fillna("").astype(str).str.lower().eq("reject").mean()
        rows.append(
            {
                "name": name,
                "count": int(len(bucket)),
                "avg_pd": _safe_float(bucket["pd"].mean()),
                "avg_pd_pct": _safe_float(bucket["pd"].mean() * 100, digits=2),
                "actual_rate": _safe_float(bucket["is_delayed"].mean()),
                "actual_rate_pct": _safe_float(bucket["is_delayed"].mean() * 100, digits=2),
                "avg_score": _safe_float(bucket["score"].mean(), digits=2),
                "reject_rate": _safe_float(reject_rate),
                "reject_rate_pct": _safe_float(reject_rate * 100, digits=2),
            }
        )
    if order:
        rows.sort(key=lambda item: (order.get(item["name"], 999), item["name"]))
    else:
        rows.sort(key=lambda item: item["count"], reverse=True)
    return rows


def _comparison_metrics(live_metrics: dict, valid_metrics: dict, test_metrics: dict) -> list[dict]:
    pairs = [
        ("ROC AUC", "roc_auc"),
        ("PR AUC", "pr_auc"),
        ("Brier", "brier"),
        ("Log Loss", "log_loss"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("KS", "ks"),
        ("Calibration Error", "calibration_error"),
    ]
    return [
        {
            "metric": label,
            "valid": _safe_float(valid_metrics.get(key)),
            "test": _safe_float(test_metrics.get(key)),
            "live": _safe_float(live_metrics.get(key)),
        }
        for label, key in pairs
    ]


def _registry_payload(entry: dict) -> dict:
    metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
    nested_metrics = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}
    valid = nested_metrics.get("valid", {}) if isinstance(nested_metrics, dict) else {}
    test = nested_metrics.get("test", {}) if isinstance(nested_metrics, dict) else {}
    thresholds = nested_metrics.get("thresholds", {}) if isinstance(nested_metrics, dict) else {}
    train_profile = metrics.get("train_profile", {}) if isinstance(metrics, dict) else {}
    acceptance = metrics.get("acceptance", {}) if isinstance(metrics, dict) else {}
    iv_report = metrics.get("iv_report_top50", []) if isinstance(metrics, dict) else []
    feature_sample = metrics.get("feature_sample", []) if isinstance(metrics, dict) else []

    return {
        "valid": {key: _safe_float(value) for key, value in valid.items()},
        "test": {key: _safe_float(value) for key, value in test.items()},
        "thresholds": {key: _safe_float(value) if isinstance(value, (int, float)) else value for key, value in thresholds.items()},
        "train_profile": {
            **{key: value for key, value in train_profile.items() if key not in {"positive_rate", "actual_label_rate"}},
            "positive_rate": _safe_float(train_profile.get("positive_rate")),
            "actual_label_rate": _safe_float(train_profile.get("actual_label_rate")),
        },
        "acceptance": acceptance if isinstance(acceptance, dict) else {},
        "iv_top_features": [
            {
                "feature": str(item.get("feature") or ""),
                "display_feature": display_feature_name(str(item.get("feature") or "")) or str(item.get("feature") or ""),
                "iv": _safe_float(item.get("iv")),
            }
            for item in iv_report[:15]
            if isinstance(item, dict)
        ],
        "feature_sample": [str(item) for item in feature_sample[:25]],
    }


def _build_insights(
    *,
    live_metrics: dict,
    confusion: dict,
    registry_payload: dict,
    threshold_values: list[float],
    band_performance: list[dict],
    row_count: int,
    segment: str,
) -> list[str]:
    insights: list[str] = []
    avg_pd = live_metrics.get("average_pd")
    actual_rate = live_metrics.get("actual_delay_rate")
    if avg_pd is not None and actual_rate is not None:
        calibration_gap = abs(float(avg_pd) - float(actual_rate))
        insights.append(
            f"Live average PD is {float(avg_pd) * 100:.1f}% against an observed delay rate of {float(actual_rate) * 100:.1f}% for {row_count:,} {segment} rows."
        )
        if calibration_gap >= 0.08:
            insights.append(
                f"Calibration gap is {calibration_gap * 100:.1f} percentage points, which suggests the live lane is drifting from observed payment behavior."
            )

    test_metrics = registry_payload.get("test", {})
    live_auc = live_metrics.get("roc_auc")
    test_auc = test_metrics.get("roc_auc")
    if live_auc is not None and test_auc is not None:
        delta = float(live_auc) - float(test_auc)
        if delta <= -0.05:
            insights.append(
                f"Live ROC AUC is {abs(delta) * 100:.1f} points below the stored test ROC AUC, so discriminatory power should be reviewed."
            )
        elif delta >= 0.03:
            insights.append(
                f"Live ROC AUC is {delta * 100:.1f} points above the stored test ROC AUC, indicating the current portfolio is separating more cleanly than the training holdout."
            )
        else:
            insights.append("Live ROC AUC is broadly aligned with the stored test ROC AUC, which is a good consistency signal.")

    false_negative_rate = None
    if confusion.get("tp", 0) + confusion.get("fn", 0) > 0:
        false_negative_rate = confusion["fn"] / (confusion["tp"] + confusion["fn"])
    if false_negative_rate is not None:
        if false_negative_rate >= 0.35:
            insights.append(
                f"{false_negative_rate * 100:.1f}% of delayed invoices are still landing in the approve path, so recall at the active threshold may be too weak."
            )
        else:
            insights.append(
                f"Delayed-invoice miss rate is {false_negative_rate * 100:.1f}%, which is the main operational watch-out for false approvals."
            )

    if len(threshold_values) > 1:
        insights.append(
            f"Multiple approval thresholds are active in this view ({', '.join(f'{value:.2f}' for value in threshold_values)}), so confusion results reflect mixed segment policy rather than one universal cutoff."
        )

    band_actuals = [item.get("actual_rate") for item in band_performance if item.get("name") in _BAND_ORDER]
    cleaned_actuals = [float(value) for value in band_actuals if value is not None]
    if len(cleaned_actuals) >= 3:
        monotonic = cleaned_actuals == sorted(cleaned_actuals)
        if monotonic:
            insights.append("Observed delay rates rise monotonically from Low to High risk bands, so band ordering is behaving as expected.")
        else:
            insights.append("Observed delay rates are not perfectly monotonic across risk bands, which can point to threshold or calibration instability.")

    train_profile = registry_payload.get("train_profile", {})
    train_positive_rate = train_profile.get("positive_rate")
    if train_positive_rate is not None and actual_rate is not None:
        delta = float(actual_rate) - float(train_positive_rate)
        insights.append(
            f"Observed delay rate is {delta * 100:+.1f} points versus the registry training positive rate, which is a simple portfolio-mix drift signal."
        )

    return insights[:6]


def build_model_performance_payload(
    *,
    scored_df: pd.DataFrame,
    descriptor: dict,
    registry_entry: dict,
    segment: str,
    snapshot_meta: dict,
) -> dict:
    frame = _safe_binary_frame(scored_df)
    default_threshold = float(descriptor.get("threshold") or 0.5)
    y_true = frame["is_delayed"].to_numpy(dtype=int) if not frame.empty else np.array([], dtype=int)
    y_score = frame["pd"].to_numpy(dtype=float) if not frame.empty else np.array([], dtype=float)
    y_pred, threshold_values = _derive_predictions(frame, default_threshold) if not frame.empty else (np.array([], dtype=int), [default_threshold])

    positive_count = int(y_true.sum()) if len(y_true) else 0
    negative_count = int(len(y_true) - positive_count)
    confusion = _confusion_payload(y_true, y_pred)

    precision = _safe_metric(precision_score, y_true, y_pred=y_pred) if len(y_true) else None
    recall = _safe_metric(recall_score, y_true, y_pred=y_pred) if len(y_true) else None
    specificity = None
    if len(y_true):
        specificity = _safe_float(
            np.sum((y_true == 0) & (y_pred == 0)) / max(np.sum(y_true == 0), 1),
        )

    live_metrics = {
        "rows": int(len(frame)),
        "positive_rows": positive_count,
        "negative_rows": negative_count,
        "actual_delay_rate": _safe_float(float(np.mean(y_true))) if len(y_true) else None,
        "average_pd": _safe_float(float(np.mean(y_score))) if len(y_score) else None,
        "average_score": _safe_float(frame["score"].dropna().mean(), digits=2) if not frame.empty and not frame["score"].dropna().empty else None,
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_score=y_score) if len(np.unique(y_true)) > 1 else None,
        "pr_auc": _safe_metric(average_precision_score, y_true, y_score=y_score) if len(np.unique(y_true)) > 1 else None,
        "brier": _safe_metric(brier_score_loss, y_true, y_score=y_score) if len(y_true) else None,
        "log_loss": _safe_metric(log_loss, y_true, y_score=y_score) if len(y_true) else None,
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred=y_pred) if len(y_true) else None,
        "precision": precision,
        "recall": recall,
        "f1": _safe_metric(f1_score, y_true, y_pred=y_pred) if len(y_true) else None,
        "specificity": specificity,
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred=y_pred) if len(y_true) else None,
        "ks": _ks_statistic(y_true, y_score) if len(np.unique(y_true)) > 1 else None,
        "calibration_error": _calibration_error(frame),
        "approve_rate": _safe_float(float(np.mean(y_pred == 0))) if len(y_pred) else None,
        "reject_rate": _safe_float(float(np.mean(y_pred == 1))) if len(y_pred) else None,
    }

    registry_payload = _registry_payload(registry_entry)
    band_performance = _performance_by_group(frame, "risk_band", order=_BAND_ORDER)
    approval_performance = _performance_by_group(frame, "approval")

    payload = {
        "segment": str(segment or "").strip().lower(),
        "model": {
            "model_family": descriptor.get("model_family"),
            "model_type": descriptor.get("model_type"),
            "model_version": descriptor.get("version"),
            "threshold": _safe_float(descriptor.get("threshold")),
            "threshold_policy": descriptor.get("threshold_policy"),
            "feature_count": descriptor.get("feature_count"),
            "artifact_source": descriptor.get("artifact_source"),
            "source_path": descriptor.get("source_path"),
        },
        "snapshot": {
            "snapshot_id": snapshot_meta.get("snapshot_id"),
            "snapshot_generated_at": snapshot_meta.get("snapshot_generated_at"),
            "rows": int(snapshot_meta.get("rows", len(frame)) or 0),
            "total_available": int(snapshot_meta.get("total_available", len(frame)) or 0),
            "summary": snapshot_meta.get("summary", {}),
            "thresholds_applied": threshold_values,
            "mixed_thresholds": len(threshold_values) > 1,
        },
        "live_metrics": live_metrics,
        "confusion_matrix": confusion,
        "registry_metrics": registry_payload,
        "comparison_metrics": _comparison_metrics(live_metrics, registry_payload.get("valid", {}), registry_payload.get("test", {})),
        "charts": {
            "calibration_bins": _calibration_bins(frame),
            "threshold_curve": _threshold_curve(frame),
            "risk_band_performance": band_performance,
            "approval_performance": approval_performance,
            "pd_distribution": _pd_histogram(frame),
        },
        "analysis": {
            "insights": _build_insights(
                live_metrics=live_metrics,
                confusion=confusion,
                registry_payload=registry_payload,
                threshold_values=threshold_values,
                band_performance=band_performance,
                row_count=int(len(frame)),
                segment=str(segment or "").strip().lower(),
            ),
        },
    }
    logger.debug(
        "Built model performance payload segment=%s rows=%d snapshot=%s",
        payload["segment"],
        payload["live_metrics"]["rows"],
        payload["snapshot"]["snapshot_id"],
    )
    return payload
