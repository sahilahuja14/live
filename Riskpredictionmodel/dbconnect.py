from __future__ import annotations

import os
from threading import Lock

from pymongo import MongoClient

from .config import get_database_name, get_mongo_uri
from .logging_config import get_logger


logger = get_logger(__name__)
_client: MongoClient | None = None
_client_lock = Lock()



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
                try:
                    _client = MongoClient(
                        mongo_uri,
                        serverSelectionTimeoutMS=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000")),
                        connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "5000")),
                        socketTimeoutMS=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "10000")),
                        maxPoolSize=int(os.getenv("MONGO_MAX_POOL_SIZE", "20")),
                    )
                    logger.info("MongoClient initialized for database=%s", resolved_name)
                except Exception:
                    logger.exception("Failed to initialize MongoClient for database=%s", resolved_name)
                    raise
    return _client[resolved_name]
