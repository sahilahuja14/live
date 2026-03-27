from __future__ import annotations

import numpy as np


DEFAULT_LOW_RISK_THRESHOLD = 0.15
DEFAULT_MEDIUM_RISK_THRESHOLD = 0.65
DEFAULT_APPROVAL_THRESHOLD = 0.30


def _scale_score(prob, base_score: int = 600, pdo: int = 50):
    prob_array = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    factor = pdo / np.log(2)
    odds = prob_array / (1 - prob_array)
    return np.rint(base_score - factor * np.log(odds)).astype(float)


def _risk_band(
    prob,
    low_threshold: float = DEFAULT_LOW_RISK_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_RISK_THRESHOLD,
):
    prob_array = np.asarray(prob, dtype=float)
    return np.select(
        [prob_array < low_threshold, prob_array < medium_threshold],
        ["Low Risk", "Medium Risk"],
        default="High Risk",
    ).astype(object)


def _approval(prob, threshold: float = DEFAULT_APPROVAL_THRESHOLD):
    prob_array = np.asarray(prob, dtype=float)
    return np.where(prob_array <= threshold, "Approve", "Reject").astype(object)


def _feature_names_out(preprocessor, transformed_width: int) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        names = preprocessor.get_feature_names_out()
        if len(names) == transformed_width:
            return [str(value) for value in names]
    return [f"f_{index}" for index in range(transformed_width)]


def _base_feature_name(transformed_name: str, raw_features: list[str]) -> str:
    cleaned = transformed_name
    if "__" in cleaned:
        cleaned = cleaned.split("__", 1)[1]

    for raw in sorted(raw_features, key=len, reverse=True):
        if cleaned == raw or cleaned.startswith(f"{raw}_"):
            return raw
    return cleaned


def _top_features_tree(
    model,
    transformed,
    feature_names: list[str],
    raw_features: list[str],
    top_n: int = 5,
) -> list[list[dict]]:
    transformed_array = np.asarray(transformed)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    else:
        importances = np.std(transformed_array, axis=0)
        if np.all(importances == 0):
            importances = np.ones(transformed_array.shape[1], dtype=float)

    if len(importances) != transformed_array.shape[1]:
        importances = np.resize(importances, transformed_array.shape[1])

    top_indices = np.argsort(np.abs(importances))[::-1][:top_n]
    base_names = {
        feat_idx: _base_feature_name(
            feature_names[feat_idx] if feat_idx < len(feature_names) else f"f_{int(feat_idx)}",
            raw_features,
        )
        for feat_idx in top_indices
    }
    payload: list[list[dict]] = []
    for row in transformed_array:
        row_out: list[dict] = []
        for feat_idx in top_indices:
            contribution = float(importances[feat_idx] * row[feat_idx])
            transformed_feature = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f_{int(feat_idx)}"
            direction = "increase_pd" if contribution > 0 else "decrease_pd" if contribution < 0 else "neutral"
            row_out.append(
                {
                    "feature": transformed_feature,
                    "base_feature": base_names[feat_idx],
                    "contribution": round(contribution, 6),
                    "direction": direction,
                }
            )
        payload.append(row_out)
    return payload
