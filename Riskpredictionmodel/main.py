from __future__ import annotations

import os

import uvicorn

from api.scoring_api import app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    reload = os.getenv("ENV", "development").strip().lower() == "development"
    uvicorn.run("main:app", host=host, port=port, reload=reload)