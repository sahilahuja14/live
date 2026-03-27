# RiskPred

RiskPred is the production scoring service for the `Risk.Main` credit-risk lane. The backend reads a single MongoDB source, builds the serving feature frame for the accepted `Risk.Main` XGBoost model, and exposes FastAPI endpoints for manual scoring, paginated bulk scoring, customer drilldown, cache refresh, and health monitoring.

## Current Production Scope

- Data source: MongoDB database `Risk`, collection `Main`
- Active model family: `risk_main`
- Active model type: `risk_main_xgb`
- Default live model version: `risk_main_xgb_20260319_094014`
- Target: `delay_days > 30`
- Serving mode: `Risk.Main` only
- Legacy `invoicemasters` / `customermasters` / payment-simulator path: removed from runtime

## Repository Layout

```text
riskpred/
??? api/                    FastAPI routes, request/response shaping, cache, auth, settings
??? credit-risk-dashboard/  Frontend dashboard wired to this backend
??? data/                   Shared data helpers that still support the active lane
??? features/               Risk.Main feature engineering and customer-history helpers
??? models/production/      Production registry and model artifacts
??? pipeline/               Risk.Main fetch, parsing, canonicalization, and scoring-frame build
??? scoring/                Artifact loading, scoring math, decisions, explainability
??? tests/                  Backend production tests
??? config.py               Environment accessors
??? dbconnect.py            MongoDB connection singleton
??? logging_config.py       Shared logger setup
??? requirements.txt        Production dependencies
??? requirements-dev.txt    Test and local-dev extras
```

## Runtime Flow

1. FastAPI starts and loads `.env`.
2. Startup validation checks `MONGO_URI`, the default database mapping, and the production model artifacts.
3. The API pre-warms the active model from `models/production/risk_main_registry.json`.
4. `ApiCache` refreshes a `Risk.Main` snapshot from MongoDB with explicit field projection.
5. The pipeline canonicalizes `Risk.Main`, builds the serving frame, and scores with the active XGBoost model.
6. `/score-all/{segment}` serves cursor-paginated rows from a scored snapshot.
7. `/score-customer/{segment}` scores live by customer, or uses a pinned snapshot when `snapshotId` is supplied.
8. `/score/{segment}` scores an ad hoc payload after mapping it to the `Risk.Main` request shape.

## API Endpoints

- `POST /score/{segment}`
  - Manual scoring for one request payload.
  - Accepts either flat request fields or nested `Risk.Main`-style payloads.
- `GET /score-all/{segment}`
  - Cursor-paginated bulk scoring from the scored snapshot.
  - Query params: `limit`, `cursor`, `refresh`.
- `POST /score-customer/{segment}`
  - Customer drilldown scoring.
  - Optional `snapshotId` for exact alignment with a `score-all` snapshot.
- `POST /cache/refresh`
  - Forces a fresh dataset + scored snapshot rebuild.
- `GET /health`
  - Reports API, Mongo, model, cache, and snapshot state.

## Environment

Copy `.env.example` to `.env` and fill the values for your environment.

Minimum required values:

```ini
MONGO_URI=
DATABASE_NAME=Risk
PRODUCTION_RISK_DB_NAME=Risk
PRODUCTION_RISK_COLLECTION=Main
PRODUCTION_RISK_REGISTRY_PATH=models/production/risk_main_registry.json
PRODUCTION_RISK_ACTIVE_MODEL_TYPE=risk_main_xgb
PRODUCTION_RISK_ACTIVE_VERSION=risk_main_xgb_20260319_094014
```

Important notes:

- `DATABASE_NAME` should align with the production `Risk` database.
- `PRODUCTION_RISK_DB_NAME` and `PRODUCTION_RISK_COLLECTION` control the active source collection.
- Setting `API_KEY` enables API-key enforcement on scoring and cache-refresh routes.
- `LOG_LEVEL` defaults to `INFO`.

## Local Run

### Backend

```bash
python -m venv venv
venv\Scriptsctivate
pip install -r requirements.txt
uvicorn api.scoring_api:app --reload --host 0.0.0.0 --port 8000
```

Swagger: `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd credit-risk-dashboard
npm install
npm run dev
```

Frontend defaults should point to the backend through its own `.env` file.

## Testing

Run the production backend tests:

```bash
venv\Scripts\python.exe -m unittest discover -s tests -p test_production_risk_main.py
```

Optional compile sanity check:

```bash
venv\Scripts\python.exe -m py_compile api\scoring_api.py api\cache.py scoring\model.py pipeline
isk_main.py pipeline
isk_canonical.py
```

## Operational Notes

- MongoDB connection timeouts and pool size are env-configurable in `dbconnect.py`.
- `/health` returns `503` when MongoDB is unreachable.
- Snapshot refresh errors are counted and surfaced in the health payload.
- The backend logs full internal exceptions but returns sanitized `500` responses to clients.
- Bulk scoring is snapshot-based; manual scoring and customer scoring remain lightweight live paths.

## What Was Removed

The production runtime no longer uses:

- `invoicemasters`
- `customermasters`
- payment simulator logic
- old multisource preprocessing path
- manifest-based legacy model loading

Those older approaches are documented historically in `doc.txt`, but they are not part of the live serving path anymore.
