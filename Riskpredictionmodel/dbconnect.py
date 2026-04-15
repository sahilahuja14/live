from __future__ import annotations

import os
from threading import Lock

from pymongo import MongoClient

from .config import (
    get_database_name,
    get_live_db_name,
    get_live_mongo_uri,
    get_mongo_uri,
)
from .logging_config import get_logger


logger = get_logger(__name__)
_client: MongoClient | None = None
_live_client: MongoClient | None = None
_client_lock = Lock()
_live_client_lock = Lock()


def _mongo_options() -> dict:
    return {
        "serverSelectionTimeoutMS": int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "10000")),
        "connectTimeoutMS": int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "10000")),
        "socketTimeoutMS": int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "30000")),
        "maxPoolSize": int(os.getenv("MONGO_MAX_POOL_SIZE", "20")),
    }


def _create_client(mongo_uri: str, resolved_name: str, label: str) -> MongoClient:
    try:
        client = MongoClient(mongo_uri, **_mongo_options())
        logger.info("MongoClient initialized label=%s database=%s", label, resolved_name)
        return client
    except Exception:
        logger.exception("Failed to initialize MongoClient label=%s database=%s", label, resolved_name)
        raise


def get_database(db_name: str | None = None):
    global _client

    mongo_uri = get_mongo_uri()
    resolved_name = db_name or get_database_name()
    if not mongo_uri:
        raise RuntimeError("MONGO_URI is not configured.")
    if not resolved_name:
        raise RuntimeError("DATABASE_NAME or PRODUCTION_RISK_DB_NAME is not configured.")

    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _create_client(mongo_uri, resolved_name, label="risk_main")
    return _client[resolved_name]


def get_live_database(db_name: str | None = None):
    global _live_client

    mongo_uri = get_live_mongo_uri()
    resolved_name = db_name or get_live_db_name()
    if not mongo_uri:
        raise RuntimeError("LIVE_MONGO_URI is not configured.")
    if not resolved_name:
        raise RuntimeError("LIVE_DB_NAME is not configured.")

    if _live_client is None:
        with _live_client_lock:
            if _live_client is None:
                _live_client = _create_client(mongo_uri, resolved_name, label="live_collections")
    return _live_client[resolved_name]
