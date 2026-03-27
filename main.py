from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from Dashboardbackend.app.core.logger import setup_logging
from Dashboardbackend.app.database import close_db, connect_db
from Dashboardbackend.app.routers import analytics, auth, dashboard, finance, streams
from Dashboardbackend.app.services.stream_manager import stream_manager

from Riskpredictionmodel.api import scoring_api as risk_scoring_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    connect_db()
    risk_scoring_api._startup_checks()
    risk_scoring_api._api_cache.start()
    await stream_manager.start_watching()
    try:
        yield
    finally:
        await stream_manager.stop_watching()
        risk_scoring_api._api_cache.stop()
        close_db()


app = FastAPI(lifespan=lifespan)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
origins = [origin.strip() for origin in allowed_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Dashboard Dashboard-Project API is running"}


@app.get("/api/health")
def health_check():
    return {"status": "ok", "database": "connected"}


app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(streams.router, prefix="/api/streams", tags=["streams"])
app.include_router(finance.router, prefix="/api/finance", tags=["finance"])
app.include_router(risk_scoring_api.router, prefix="/api/risk", tags=["risk"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = logging.getLogger("app.main")
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc)},
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENV", "development") == "development"
    uvicorn.run("main:app", host=host, port=port, reload=reload)
