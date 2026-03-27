from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


_ENV_INITIALIZED = False
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def init_env() -> None:
    global _ENV_INITIALIZED
    if _ENV_INITIALIZED:
        return
    # Load the workspace-level .env first so the standalone risk service and the
    # combined app share the same secrets. Keep the package-local .env as a
    # fallback for risk-only overrides.
    for env_path in (WORKSPACE_ROOT / ".env", PROJECT_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path, override=False)
    _ENV_INITIALIZED = True


def _env_value(name: str) -> str | None:
    init_env()
    value = os.getenv(name)
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def get_mongo_uri() -> str | None:
    return _env_value("MONGO_URI")


def get_database_name() -> str | None:
    return _env_value("DATABASE_NAME") or _env_value("PRODUCTION_RISK_DB_NAME")


def get_production_risk_db_name() -> str:
    return _env_value("PRODUCTION_RISK_DB_NAME") or get_database_name() or "Risk"


def get_production_risk_collection() -> str:
    return _env_value("PRODUCTION_RISK_COLLECTION") or "Main"
