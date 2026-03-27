from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from ..logging_config import get_logger
from ..pipeline.risk_map import (
    PRODUCTION_MODEL_FAMILY,
    PRODUCTION_RISK_ACTIVE_MODEL_TYPE,
    PRODUCTION_RISK_ACTIVE_VERSION,
    PRODUCTION_RISK_REGISTRY_PATH,
)
from .utils import _approval, _feature_names_out, _risk_band, _scale_score, _top_features_tree


logger = get_logger(__name__)
PRODUCTION_MODEL_TYPE = PRODUCTION_RISK_ACTIVE_MODEL_TYPE
PRODUCTION_APPROVAL_THRESHOLD_POLICY = os.getenv("PRODUCTION_APPROVAL_THRESHOLD_POLICY", "active").strip().lower()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_artifacts_lock = Lock()
_artifacts: dict | None = None



def get_active_production_model_family() -> str:
    return PRODUCTION_MODEL_FAMILY



def _path_candidates(candidate: str | None, registry_path: str | None = None) -> list[Path]:
    if not candidate:
        return []

    raw_path = Path(candidate)
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                raw_path,
                PROJECT_ROOT / raw_path,
                PROJECT_ROOT / "models" / "production" / raw_path.name,
            ]
        )
        if registry_path:
            registry_file = PROJECT_ROOT / Path(registry_path)
            candidates.append(registry_file.parent / raw_path.name)

    unique: list[Path] = []
    seen: set[str] = set()
    for item in candidates:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _resolve_path(candidate: str | None, registry_path: str | None = None) -> str:
    if not candidate:
        raise FileNotFoundError("Artifact path is missing.")
    candidates = _path_candidates(candidate, registry_path=registry_path)
    for item in candidates:
        if item.exists():
            return str(item)
    return str(candidates[0] if candidates else Path(candidate))



def load_risk_main_registry(path: str = PRODUCTION_RISK_REGISTRY_PATH) -> dict:
    resolved_path = _resolve_path(path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Risk.Main production registry not found: {resolved_path}")
    with open(resolved_path, "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    models = registry.get("models") if isinstance(registry, dict) else None
    if not isinstance(models, list) or not models:
        raise ValueError("Risk.Main registry does not contain any models.")
    return registry



def _entry_thresholds(entry: dict) -> dict:
    metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
    nested_metrics = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}
    thresholds = nested_metrics.get("thresholds") if isinstance(nested_metrics, dict) else None
    if isinstance(thresholds, dict) and thresholds:
        return thresholds
    thresholds = metrics.get("thresholds") if isinstance(metrics, dict) else None
    return thresholds if isinstance(thresholds, dict) else {}



def _resolve_threshold_from_registry_entry(entry: dict) -> tuple[float, str]:
    thresholds = _entry_thresholds(entry)
    for key in [PRODUCTION_APPROVAL_THRESHOLD_POLICY, "active", "recall_floor_best_f1", "max_f1", "manual_default"]:
        if key in thresholds:
            try:
                return float(thresholds[key]), key
            except Exception:
                continue
    return 0.30, "manual_default"



def _sort_key(entry: dict) -> tuple[str, str]:
    return str(entry.get("created_at") or ""), str(entry.get("version") or "")



def _select_risk_main_entry(registry: dict) -> dict:
    models = [
        entry
        for entry in registry.get("models", [])
        if str(entry.get("model_type") or "").strip() == PRODUCTION_RISK_ACTIVE_MODEL_TYPE
    ]
    if not models:
        raise ValueError(f"No registry entries found for model_type '{PRODUCTION_RISK_ACTIVE_MODEL_TYPE}'.")

    requested_version = str(registry.get("active_version") or PRODUCTION_RISK_ACTIVE_VERSION or "").strip()
    if requested_version:
        for entry in models:
            if str(entry.get("version") or "").strip() == requested_version:
                return entry

    champions = [entry for entry in models if str(entry.get("status") or "").strip().lower() == "champion"]
    if champions:
        return sorted(champions, key=_sort_key, reverse=True)[0]

    approved = [
        entry
        for entry in models
        if bool(entry.get("metrics", {}).get("acceptance", {}).get("approved", False))
    ]
    if approved:
        return sorted(approved, key=_sort_key, reverse=True)[0]

    return sorted(models, key=_sort_key, reverse=True)[0]



def describe_active_production_model(force_reload: bool = False) -> dict:
    artifacts = load_production_artifacts(force_reload=force_reload)
    return {
        "model_family": artifacts["model_family"],
        "model_type": artifacts["model_type"],
        "version": artifacts["version"],
        "threshold": artifacts["approval_threshold"],
        "threshold_policy": artifacts["threshold_policy"],
        "source_path": artifacts["source_path"],
        "artifact_source": artifacts["artifact_source"],
        "model_path": artifacts["model_path"],
        "preprocessor_path": artifacts["preprocessor_path"],
        "feature_count": len(artifacts["features"]),
    }



def load_production_artifacts(force_reload: bool = False) -> dict:
    global _artifacts
    load_start = perf_counter()
    with _artifacts_lock:
        if _artifacts is not None and not force_reload:
            return _artifacts

        registry = load_risk_main_registry()
        entry = _select_risk_main_entry(registry)
        threshold, threshold_policy = _resolve_threshold_from_registry_entry(entry)
        model_path = _resolve_path(entry.get("model_path"), PRODUCTION_RISK_REGISTRY_PATH)
        preprocessor_path = _resolve_path(entry.get("preprocessor_path"), PRODUCTION_RISK_REGISTRY_PATH)
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        features = getattr(preprocessor, "feature_names_in_", None)
        if features is None:
            raise ValueError("Risk.Main production preprocessor is missing feature_names_in_.")

        _artifacts = {
            "model": model,
            "preprocessor": preprocessor,
            "features": list(features),
            "version": entry.get("version", "unknown"),
            "model_type": entry.get("model_type", PRODUCTION_RISK_ACTIVE_MODEL_TYPE),
            "model_family": PRODUCTION_MODEL_FAMILY,
            "approval_threshold": float(threshold),
            "threshold_policy": threshold_policy,
            "source_path": PRODUCTION_RISK_REGISTRY_PATH,
            "artifact_source": "registry",
            "model_path": model_path,
            "preprocessor_path": preprocessor_path,
            "registry_entry": entry,
        }

        logger.info(
            "Loaded production artifacts model_family=%s model_type=%s version=%s source=%s duration_ms=%.1f",
            _artifacts["model_family"],
            _artifacts["model_type"],
            _artifacts["version"],
            _artifacts["artifact_source"],
            (perf_counter() - load_start) * 1000.0,
        )
        return _artifacts



def score_production_frame(
    df: pd.DataFrame,
    top_n: int = 5,
    approval_threshold_override: float | None = None,
    scoring_context: str | None = None,
) -> pd.DataFrame:
    inference_start = perf_counter()
    artifacts = load_production_artifacts()
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    features = artifacts["features"]

    result = df.copy()
    for col in features:
        if col not in result.columns:
            result[col] = np.nan

    X_raw = result[features]
    X_t = preprocessor.transform(X_raw)
    X_dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
    prob = model.predict_proba(X_t)[:, 1]
    if not np.all(np.isfinite(prob)):
        bad_count = int(np.sum(~np.isfinite(prob)))
        logger.warning("Model returned %d non-finite probabilities. Clipping.", bad_count)
        prob = np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0)
        prob = np.clip(prob, 1e-6, 1 - 1e-6)

    approval_threshold = artifacts["approval_threshold"]
    threshold_policy = artifacts["threshold_policy"]
    if approval_threshold_override is not None:
        approval_threshold = float(np.clip(approval_threshold_override, 0.0, 1.0))
        threshold_policy = "override"

    result["pd"] = prob
    result["score"] = _scale_score(prob)
    result["risk_band"] = _risk_band(prob)
    result["approval"] = _approval(prob, threshold=approval_threshold)
    result["top_features"] = _top_features_tree(
        model,
        X_dense,
        feature_names=_feature_names_out(preprocessor, X_dense.shape[1]),
        raw_features=features,
        top_n=top_n,
    )
    result["model_family"] = artifacts["model_family"]
    result["model_type"] = artifacts["model_type"]
    result["model_version"] = artifacts["version"]
    result["approval_threshold_policy"] = threshold_policy
    result["approval_threshold"] = float(approval_threshold)
    result["artifact_source"] = artifacts["artifact_source"]
    if scoring_context is not None:
        result["scoring_context"] = scoring_context
    result["scoring_timestamp"] = datetime.now(timezone.utc).isoformat()
    logger.debug(
        "Scored %d rows model_family=%s model_type=%s version=%s duration_ms=%.1f context=%s",
        len(result),
        artifacts["model_family"],
        artifacts["model_type"],
        artifacts["version"],
        (perf_counter() - inference_start) * 1000.0,
        scoring_context,
    )
    return result
