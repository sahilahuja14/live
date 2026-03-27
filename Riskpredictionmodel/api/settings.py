from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from ..scoring.model import PRODUCTION_MODEL_TYPE, get_active_production_model_family


@dataclass(frozen=True)
class ApiSettings:
    auto_refresh_enabled: bool
    auto_refresh_interval_seconds: int
    dataset_cache_ttl_seconds: int
    history_cache_ttl_seconds: int
    score_all_page_default: int
    score_all_page_max: int
    scored_snapshot_retention_seconds: int
    cors_allowed_origins: tuple[str, ...]
    cors_allow_credentials: bool
    active_model_family: str
    active_model_type: str
    threshold_override_global: float | None
    threshold_override_air: float | None
    threshold_override_ocean: float | None


def _optional_env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return float(np.clip(value, 0.0, 1.0))


def _env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    values = [item.strip() for item in str(raw).split(",")]
    return tuple(item for item in values if item)


def load_api_settings() -> ApiSettings:
    auto_refresh_enabled = os.getenv("API_AUTO_REFRESH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    auto_refresh_interval_seconds = int(os.getenv("API_AUTO_REFRESH_INTERVAL_SECONDS", "1800"))
    dataset_cache_ttl_seconds = int(
        os.getenv("API_DATASET_CACHE_TTL_SECONDS", str(max(auto_refresh_interval_seconds + 300, 1800)))
    )
    history_cache_ttl_seconds = int(
        os.getenv("API_HISTORY_CACHE_TTL_SECONDS", str(max(auto_refresh_interval_seconds + 600, 2400)))
    )
    score_all_page_default = int(os.getenv("API_SCORE_ALL_PAGE_DEFAULT", "100"))
    score_all_page_max = int(os.getenv("API_SCORE_ALL_PAGE_MAX", "250"))
    scored_snapshot_retention_seconds = int(
        os.getenv("API_SCORED_SNAPSHOT_RETENTION_SECONDS", str(max(auto_refresh_interval_seconds * 2, 3600)))
    )
    cors_allowed_origins = _env_csv(
        "API_CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:5173,http://localhost:5173",
    )
    cors_allow_credentials = os.getenv("API_CORS_ALLOW_CREDENTIALS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    active_family = get_active_production_model_family()
    return ApiSettings(
        auto_refresh_enabled=auto_refresh_enabled,
        auto_refresh_interval_seconds=auto_refresh_interval_seconds,
        dataset_cache_ttl_seconds=dataset_cache_ttl_seconds,
        history_cache_ttl_seconds=history_cache_ttl_seconds,
        score_all_page_default=max(score_all_page_default, 1),
        score_all_page_max=max(score_all_page_max, 1),
        scored_snapshot_retention_seconds=max(scored_snapshot_retention_seconds, 300),
        cors_allowed_origins=cors_allowed_origins,
        cors_allow_credentials=cors_allow_credentials,
        active_model_family=active_family,
        active_model_type=PRODUCTION_MODEL_TYPE,
        threshold_override_global=_optional_env_float("API_PRODUCTION_THRESHOLD_OVERRIDE"),
        threshold_override_air=_optional_env_float("API_PRODUCTION_THRESHOLD_OVERRIDE_AIR"),
        threshold_override_ocean=_optional_env_float("API_PRODUCTION_THRESHOLD_OVERRIDE_OCEAN"),
    )
