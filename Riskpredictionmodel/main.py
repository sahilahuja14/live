from __future__ import annotations

import os

import uvicorn

try:
    from .api.scoring_api import app
except ImportError:  # pragma: no cover
    from api.scoring_api import app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    reload = os.getenv("ENV", "development").strip().lower() == "development"
    target = "Riskpredictionmodel.main:app" if __package__ else "main:app"
    uvicorn.run(target, host=host, port=port, reload=reload)
