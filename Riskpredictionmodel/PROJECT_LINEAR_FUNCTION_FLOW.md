# RiskPred Linear Function Flow Guide

This document explains the live `riskpred` project in one top-down, function-by-function order.
The goal is simple: if a request starts in the dashboard or Swagger, this file tells you exactly which function runs next, what DataFrame is created, and how the final JSON is returned.

## 1. The Big Picture

There are 6 active runtime flows in this project:

1. App startup and background refresh
2. `POST /score/{segment}` for one manual invoice
3. `GET /score-all/{segment}` for paginated bulk scoring
4. `POST /score-customer/{segment}` for customer drilldown
5. `GET /model-performance/{segment}` for live model diagnostics
6. `GET /health` and `POST /cache/refresh` for operations

The frontend never talks to Mongo directly.
Every frontend page calls `credit-risk-dashboard/src/services/api.js`, which calls FastAPI, and FastAPI either:
- hits the live Mongo -> pipeline -> scoring path, or
- reuses the scored snapshot in `ApiCache`

---

## 2. Shared Foundation: Environment, DB, Settings, Auth

### 2.1 Environment load
File: `config.py`

Top-level helpers:
- `init_env()`
- `_env_value(name)`
- `get_mongo_uri()`
- `get_database_name()`
- `get_production_risk_db_name()`
- `get_production_risk_collection()`

What happens:
1. `init_env()` loads `.env` once using `load_dotenv()`.
2. Every config getter reads the env lazily after that.
3. `DATABASE_NAME`, `PRODUCTION_RISK_DB_NAME`, and `PRODUCTION_RISK_COLLECTION` decide where Mongo is read from.

### 2.2 Mongo connection
File: `dbconnect.py`

Main function:
- `get_database(db_name=None)`

What happens:
1. Reads `MONGO_URI` and DB name from `config.py`.
2. Creates one singleton `MongoClient` with timeouts and pool size.
3. Returns `_client[resolved_name]`.

This is the first real DB connection point in the project.

### 2.3 API settings
File: `api/settings.py`

Main functions:
- `load_api_settings()`
- `_optional_env_float(name)`
- `_env_csv(name)`

What happens:
1. Loads page size, snapshot TTL, auto-refresh, CORS, and threshold override settings.
2. Returns one `ApiSettings` dataclass used by `api/scoring_api.py`.

### 2.4 API auth
File: `api/auth.py`

Main function:
- `require_api_key(x_api_key=None)`

What happens:
1. If `API_KEY` is empty, requests are allowed.
2. If `API_KEY` is set, every protected route must send matching `x-api-key`.

---

## 3. Startup Flow: What Happens When the Backend Starts

File: `api/scoring_api.py`

Main startup functions:
- `_startup_checks()`
- `_check_mongo_live()`
- `lifespan(app)`

Linear order:
1. `scoring_api.py` imports `init_env()` and runs it.
2. `SETTINGS = load_api_settings()` builds runtime config.
3. `_api_cache = ApiCache(...)` creates the cache manager.
4. FastAPI starts and enters `lifespan()`.
5. `lifespan()` calls `_startup_checks()`.
6. `_startup_checks()` validates:
   - `get_mongo_uri()`
   - `get_database_name()`
   - `load_production_artifacts()`
7. `load_production_artifacts()` loads the active model + preprocessor from the registry into memory.
8. `lifespan()` then calls `_api_cache.start()`.
9. `ApiCache.start()` launches the background refresh thread.

Result:
- env is ready
- Mongo config is validated
- champion model artifacts are warm
- background cache refresh is running

---

## 4. Mongo Fetch Flow: From Projection to Canonical DataFrame

These are the lowest-level data loading steps that all scoring flows depend on.

### 4.1 Projection fetch from `Risk.Main`
File: `pipeline/risk_canonical.py`

Main constants:
- `RISK_MAIN_FETCH_PROJECTION`
- `RISK_MAIN_FETCH_BATCH_SIZE`

Main functions:
- `fetch_risk_main_frame(query=None, limit=None)`
- `inspect_risk_main_indexes()`

Linear order inside `fetch_risk_main_frame()`:
1. `get_database(PRODUCTION_RISK_DB_NAME)` gets the DB handle.
2. `collection = db[PRODUCTION_RISK_COLLECTION]` resolves the `Main` collection.
3. `collection.find(query or {}, RISK_MAIN_FETCH_PROJECTION)` runs the projected Mongo query.
4. `.batch_size(RISK_MAIN_FETCH_BATCH_SIZE)` keeps fetches chunked.
5. Each Mongo document is flattened by `flatten_dict(doc)`.
6. The flattened rows become `pd.DataFrame(rows)`.

Output:
- A raw flattened DataFrame directly from Mongo documents.

### 4.2 Canonicalization
File: `pipeline/risk_canonical.py`

Main function:
- `canonicalize_risk_main_frame(df, target_delay_days=..., as_of_date=None)`

Linear order inside canonicalization:
1. Create empty `canonical` DataFrame on same index.
2. Copy raw flattened fields into canonical names via `NORMALIZED_TO_CANONICAL_FIELD_MAP`.
3. Parse dates with `parse_main_date()`:
   - `invoice_date`
   - `due_date`
   - `payment_date`
   - `execution_date`
   - `customer_onboard_date`
4. Coerce numeric columns using `pd.to_numeric()`.
5. Normalize booleans and string defaults.
6. Derive business columns:
   - `delay_days`
   - `target`
   - `label_quality`
   - `execution_gap_days`
   - `days_to_due`
   - `customer_age_days`
   - `weight_discrepancy`
   - `aging_total`
   - `unpaid_amount`
7. Derive safe ratios with `safe_ratio()`:
   - `gross_to_invoice_ratio`
   - `paid_to_invoice_ratio`
   - `tds_to_invoice_ratio`
   - `exposure_to_invoice_ratio`
   - `aging_total_to_invoice_ratio`
8. Normalize `customer_key`.

Output:
- Canonical invoice-level DataFrame ready for feature engineering.

### 4.3 Convenience wrapper
File: `pipeline/risk_canonical.py`

Function:
- `load_canonical_risk_main_dataset(limit=None)`

What it does:
1. calls `fetch_risk_main_frame()`
2. then calls `canonicalize_risk_main_frame()`

---

## 5. Feature and Scoring Frame Flow

### 5.1 Manual payload into one-row frame
File: `pipeline/risk_main.py`

Function:
- `build_risk_main_manual_request_frame(segment, payload)`

Linear order:
1. Convert Pydantic request into dict with `_payload_dict(payload)`.
2. Flatten nested request fields with `flatten_payload(...)`.
3. Build one invoice record with required core fields.
4. Map incoming payload keys through `PRODUCTION_RISK_REQUEST_FIELD_MAP`.
5. Fill defaults for missing business fields like:
   - segment
   - document type
   - account type
   - currency
   - commodity
6. Parse invoice, execution, due, payment dates using:
   - `parse_main_date()`
   - `parse_payment_value()`
7. Normalize invoice and gross amounts.
8. Return `pd.DataFrame([record])`.

Output:
- One-row raw request DataFrame for `/score/{segment}`.

### 5.2 Full live scoring frame
File: `pipeline/risk_main.py`

Functions:
- `frame_invoice_keys(df)`
- `build_risk_main_scoring_frame(target_df, history_df=None)`

Linear order inside `build_risk_main_scoring_frame()`:
1. Copy `target_df`.
2. Create stable invoice keys using `frame_invoice_keys()`.
3. Mark current rows with:
   - `_scoring_row_flag = 1`
   - `_scoring_row_order = range(...)`
4. If `history_df` exists:
   - copy it
   - create invoice keys
   - remove rows whose keys already exist in current rows
   - mark history rows with `_scoring_row_flag = 0`
5. Concatenate history + current into one combined frame.
6. Call `canonicalize_risk_main_frame(combined)`.
7. Copy the current-row markers back onto the canonical frame.
8. Call `build_risk_main_feature_frame(canonical)`.
9. Filter only current rows back out.
10. Drop scratch columns.

Output:
- Feature-engineered scoring frame for only the rows being scored now.

### 5.3 Customer aggregate helper
File: `pipeline/risk_main.py`

Function:
- `build_risk_main_customer_aggregates(full_df, customer_ids)`

Linear order:
1. Filter `full_df` by requested `customer.customerId` values.
2. Canonicalize the filtered history.
3. Group by `customer_key`.
4. Build aggregates like:
   - `customer_total_invoices`
   - `customer_delayed_invoices`
   - `customer_avg_invoice`
   - `customer_avg_delay_days`
   - `customer_max_delay_days`
   - `customer_delay_rate`
5. Fill defaults from `CUSTOMER_AGGREGATE_DEFAULTS`.

Output:
- One row per customer with aggregate history metrics.

---

## 6. Feature Engineering Flow

File: `features/engineering.py`

Master entry point:
- `build_risk_main_feature_frame(canonical_df)`

This is the main feature builder used before model inference.

Linear order inside `build_risk_main_feature_frame()`:
1. Copy canonical DataFrame.
2. Normalize text columns with `_safe_text_col()`.
3. Normalize numeric columns with `_safe_num_col()`.
4. Create numeric derived fields, including logs, ratios, gaps, and pressure indices.
5. Add calendar features with `_add_calendar_features()` for:
   - invoice date
   - due date
   - execution date
6. Add point-in-time customer features with `add_point_in_time_customer_features()`.
7. Add additional customer-history features with `_add_additional_customer_history_features()`.
8. Add categorical frequency features with `_add_frequency_features()`.
9. Add route and entity roll-up features with `_add_entity_history_risk_features()`.
10. Add final interaction features.

Important sub-functions used here:
- `_add_calendar_features()`
- `_add_frequency_features()`
- `_add_additional_customer_history_features()`
- `_terms_bucket()`
- `_amount_bucket()`
- `_compose_key()`
- `_add_entity_history_risk_features()`

Output:
- Wide feature DataFrame containing model inputs and supporting descriptive columns.

### 6.1 Point-in-time features
File: `features/point_in_time.py`

Function:
- `add_point_in_time_customer_features(...)`

What it does:
1. Groups rows by source and customer.
2. Sorts them by event date.
3. For each row, calculates only history that would have been known before that row.
4. Adds PIT-safe columns like:
   - `prior_invoice_count`
   - `prior_delay_rate`
   - `prior_avg_delay_days`
   - `prior_open_invoice_count`
   - `days_since_last_invoice`
   - `recurring_delay_flag`

Why it matters:
- This is the no-leakage layer.
- Each row only sees prior behavior, not future outcomes.

### 6.2 Customer aggregate merge helpers
File: `features/customer_aggregates.py`

Functions:
- `customer_ids_from_frame(df)`
- `build_customer_history_aggregates(history_df)`
- `merge_customer_history_aggregates(df, aggregates)`
- `add_customer_aggregates(df)`

What they do:
- extract customer IDs
- build aggregate history stats
- merge those stats onto scoring rows

In the live API flow, `ApiCache.enrich_with_customer_history()` mainly uses:
- `customer_ids_from_frame()`
- `build_risk_main_customer_aggregates()` from `pipeline/risk_main.py`
- `merge_customer_history_aggregates()`

---

## 7. Model Loading and Scoring Flow

File: `scoring/model.py`

Main functions:
- `load_risk_main_registry()`
- `_select_risk_main_entry()`
- `load_production_artifacts()`
- `describe_active_production_model()`
- `score_production_frame()`
- `get_active_production_model_family()`

### 7.1 Artifact selection
Linear order inside `load_production_artifacts()`:
1. `load_risk_main_registry()` opens `models/production/risk_main_registry.json`.
2. `_select_risk_main_entry()` chooses the active/champion registry entry.
3. `_resolve_threshold_from_registry_entry()` finds threshold policy.
4. `_resolve_path()` resolves model and preprocessor file paths.
5. `joblib.load(model_path)` loads the XGBoost model.
6. `joblib.load(preprocessor_path)` loads the sklearn preprocessor.
7. `feature_names_in_` becomes the active feature list.
8. Everything is cached in `_artifacts`.

### 7.2 Model inference
Linear order inside `score_production_frame(df, ...)`:
1. Copy the incoming scoring frame.
2. Ensure all model feature columns exist.
3. Slice raw model inputs: `X_raw = result[features]`.
4. Transform with `preprocessor.transform(X_raw)`.
5. Convert transformed matrix to dense if needed.
6. Run `model.predict_proba(X_t)[:, 1]`.
7. Clip non-finite outputs if needed.
8. Decide threshold:
   - registry threshold by default
   - override if route/cache supplies one
9. Add prediction columns:
   - `pd`
   - `score` via `_scale_score()`
   - `risk_band` via `_risk_band()`
   - `approval` via `_approval()`
   - `top_features` via `_top_features_tree()`
10. Add metadata:
   - `model_family`
   - `model_type`
   - `model_version`
   - `approval_threshold_policy`
   - `approval_threshold`
   - `artifact_source`
   - `scoring_context`
   - `scoring_timestamp`

Output:
- Scored DataFrame with predictions and explanation payload.

---

## 8. Response Shaping Flow

File: `api/response_builder.py`

Main functions:
- `response_from_raw(raw_df, scored_df)`
- `_shape_response_frame(df, response_mode="lean")`
- `_derive_contract_fields(record)`
- `_normalize_response_records(df)`
- `_filter_customer_rows(df, customer_id)`
- `_build_scored_summary(df)`
- `_build_customer_summary_payload(records, ...)`
- `_aggregate_customer_top_features(records_df)`

Linear order for response shaping:
1. `response_from_raw()` merges raw invoice fields with scored output fields.
2. `_shape_response_frame(..., response_mode="lean")` converts merged rows into API contract fields.
3. `_derive_contract_fields()` maps internal fields into display/output fields like:
   - `invoice_key`
   - `customer.customerId`
   - `pd`
   - `score`
   - `risk_band`
   - `approval`
4. `build_credit_suggestions()` adds recommended actions.
5. `_normalize_top_features()` cleans explanation features.
6. `_normalize_response_records()` makes values JSON-safe.

Customer-specific shaping:
1. `_filter_customer_rows()` narrows invoice rows to one customer.
2. `_build_customer_summary_payload()` computes one customer summary card from the record list.
3. `_aggregate_customer_top_features()` combines invoice-level top features into customer-level drivers.

---

## 9. Cache and Snapshot Flow

File: `api/cache.py`

Class:
- `ApiCache`

Important methods:
- `load_full_dataset()`
- `fetch_customer_aggregates()`
- `enrich_with_customer_history()`
- `_build_scored_snapshot()`
- `load_scored_snapshot()`
- `get_scored_page()`
- `get_snapshot_customer_frame()`
- `get_scored_segment_frame()`
- `refresh()`
- `snapshot()`
- `_worker()`
- `start()` / `stop()`

### 9.1 Live dataset cache
Linear order inside `load_full_dataset()`:
1. Check current model key.
2. If cache is warm and not expired, return cached dataset copy.
3. Otherwise call `fetch_production_risk_main_dataset()`.
4. That calls `fetch_risk_main_frame()`.
5. Result is passed through `_prepare_snapshot_frame()` to add:
   - normalized segment helper column
   - parsed invoice timestamp helper column
   - sortable `_id` helper column
6. `inspect_risk_main_indexes()` is refreshed.
7. Dataset cache is updated.

Output:
- Full Mongo-backed DataFrame cached for reuse.

### 9.2 Customer history enrichment
Linear order inside `enrich_with_customer_history(df)`:
1. `customer_ids_from_frame(df)` extracts unique customer IDs.
2. `fetch_customer_aggregates(customer_ids)` either returns cached aggregates or builds them.
3. `merge_customer_history_aggregates(df, aggregates)` left-joins those aggregates onto `df`.

Output:
- Same invoice rows, now enriched with customer-level history stats.

### 9.3 Scored snapshot build
Linear order inside `_build_scored_snapshot(full_df, model_key)`:
1. Prepare the full frame again for segment and invoice sorting helpers.
2. Enrich the entire full frame with customer history.
3. Split rows by segment.
4. For each segment:
   - call `score_mongo_frame(current_df, history_df=enriched_full, ...)`
   - call `response_from_raw(current_df, scored)`
5. Concatenate all segment frames.
6. Sort globally by invoice timestamp and `_id`.
7. Convert to lean response frame via `_shape_response_frame(..., "lean")`.
8. Serialize once via `_normalize_response_records()`.
9. Build:
   - `records`
   - `segment_positions`
   - `segment_counts`
   - `segment_summaries`
   - `snapshot_id`
   - `generated_at`

Output:
- One reusable scored snapshot used by `/score-all`, snapshot-pinned `/score-customer`, and `/model-performance`.

### 9.4 Cursor paging
Linear order inside `get_scored_page(segment, page_size, cursor=None, refresh=False)`:
1. If cursor exists, decode it with `_decode_cursor()`.
2. Validate cursor segment and page size.
3. Resolve the correct snapshot by `snapshot_id`.
4. If no cursor, call `load_scored_snapshot()`.
5. Use `segment_positions` to find row indexes for the requested segment.
6. Slice only the requested page of `records`.
7. Build page-level `summary` with `_build_scored_summary()`.
8. Build `next_cursor` with `_encode_cursor()` if more rows exist.
9. Return:
   - `count`
   - `limit`
   - `summary`
   - `snapshot_summary`
   - `total_available`
   - `records`
   - `pagination`

### 9.5 Snapshot customer drilldown
Linear order inside `get_snapshot_customer_frame(snapshot_id, segment, customer_id, limit=None)`:
1. Load or resolve the requested snapshot.
2. Use `segment_positions` to scope rows.
3. Build DataFrame from scoped `records`.
4. `_filter_customer_rows()` narrows to one customer.
5. Apply limit if present.

Output:
- Customer-specific DataFrame from an existing snapshot.

### 9.6 Background refresh worker
Linear order:
1. `ApiCache.start()` spawns background thread `_worker()`.
2. `_worker()` loops until stopped.
3. Each cycle runs `refresh()`.
4. `refresh()` calls:
   - `load_full_dataset(force_refresh=True)`
   - `load_scored_snapshot(force_refresh=True, source_df=fresh)`
5. If refresh succeeds, error counter resets.
6. If refresh fails, consecutive error count is raised and logged.

Result:
- Cache stays warm without the frontend having to rebuild everything manually.

---

## 10. Backend Route Flow: `POST /score/{segment}`

File: `api/scoring_api.py`
Frontend caller: `credit-risk-dashboard/src/views/ManualScore.jsx` -> `credit-risk-dashboard/src/services/api.js` -> `api.scoreManual()`

### 10.1 Frontend path
1. User fills the Manual Score form in `ManualScore.jsx`.
2. `submit()` builds a payload.
3. `api.scoreManual(segment, payload)` sends `POST /score/{segment}`.

### 10.2 Backend linear flow
1. Route function `score(segment, payload)` starts.
2. `_prepare_history_frame()` calls `_api_cache.load_full_dataset()`.
3. `_build_manual_request_frame(segment, payload)` creates one-row invoice DataFrame.
4. `_enrich_with_customer_history(raw_df)` merges customer history aggregates.
5. `score_mongo_frame(raw_df, history_df=history_df, ...)` runs scoring.
6. `build_risk_main_scoring_frame()` builds the model-ready feature frame.
7. `score_production_frame()` appends predictions and top features.
8. `_response_from_raw(raw_df, result)` merges raw + scored output.
9. `_shape_response_frame(..., "lean")` maps to API contract columns.
10. `_normalize_response_records(...)[0]` returns one JSON object.

Final output:
- One invoice risk-scoring JSON response.

---

## 11. Backend Route Flow: `GET /score-all/{segment}`

File: `api/scoring_api.py`
Frontend callers:
- `Dashboard.jsx` via `api.fetchScoreAllWindow()`
- `ScoreAll.jsx` via `api.scoreAll()`

### 11.1 Frontend path
1. Dashboard or Score All page loads.
2. `api.scoreAll(segment, { limit, cursor, refresh })` calls the endpoint.
3. `ScoreAll.jsx` appends `records` page by page.
4. `Dashboard.jsx` usually reads summary + a limited sample window.

### 11.2 Backend linear flow
1. Route `score_all(segment, limit, cursor, refresh)` starts.
2. `describe_active_production_model()` reads model metadata.
3. `_api_cache.get_scored_page(...)` is called.
4. `get_scored_page()` either:
   - reuses an existing scored snapshot, or
   - builds a new one through `load_scored_snapshot()` -> `_build_scored_snapshot()`
5. The route wraps the page payload with model metadata.
6. Response returns:
   - `segment`
   - `model_type`
   - `model_family`
   - `model_version`
   - `count`
   - `limit`
   - `summary`
   - `snapshot_summary`
   - `total_available`
   - `pagination`
   - `records`

Final output:
- Paginated JSON response built from the scored snapshot.

---

## 12. Backend Route Flow: `POST /score-customer/{segment}`

File: `api/scoring_api.py`
Frontend caller: `CustomerLookup.jsx` via `api.scoreCustomer()`

This route has two valid paths.

### 12.1 Frontend path
1. `CustomerLookup.jsx` stores:
   - `customerId`
   - `snapshotId`
2. Manual search clears `snapshotId`.
3. Clicking a customer from Score All passes both `customerId` and `snapshotId`.
4. `api.scoreCustomer(segment, customerId, { limit, snapshotId })` sends the request.

### 12.2 Snapshot-pinned route path
If `payload.snapshotId` is present:
1. `score_customer()` calls `_api_cache.get_snapshot_customer_frame(...)`.
2. Snapshot rows are resolved from cached `records`.
3. Customer rows are filtered from that snapshot only.
4. `_normalize_response_records(snapshot_df)` serializes them.
5. `_build_customer_summary_payload(...)` creates the customer summary JSON.

Use case:
- When user clicked a customer from Score All and wants the exact same snapshot context.

### 12.3 Live scoring path
If `payload.snapshotId` is empty:
1. `score_customer()` calls `build_scored_dataset(segment, limit, customer_id)`.
2. `build_scored_dataset()` calls `build_scored_frame()`.
3. `build_scored_frame()`:
   - loads full dataset from Mongo cache
   - filters by segment using `filter_segment()`
   - filters to one customer using `_filter_customer_rows()`
   - enriches rows with customer history
   - calls `score_mongo_frame()`
   - calls `_response_from_raw()`
4. `_shape_response_frame(..., "lean")` shapes rows.
5. `_normalize_response_records()` serializes them.
6. `_build_customer_summary_payload(...)` aggregates them into one customer summary.

Use case:
- Manual customer lookup independent of Score All.

Final output:
- One customer summary JSON payload, optionally snapshot-pinned.

---

## 13. Backend Route Flow: `GET /model-performance/{segment}`

File: `api/scoring_api.py`
Frontend caller: `ModelPerformance.jsx` via `api.modelPerformance()`

### 13.1 Backend linear flow
1. Route `model_performance(segment, refresh=False)` starts.
2. `describe_active_production_model()` loads model metadata.
3. `load_production_artifacts()` loads registry entry and artifacts.
4. Route tries `_api_cache.get_scored_segment_frame(segment, refresh=refresh)`.
5. That reuses or rebuilds the scored snapshot and returns the segment-scoped scored DataFrame.
6. `build_model_performance_payload(...)` is called.

### 13.2 Inside `build_model_performance_payload()`
File: `scoring/performance.py`

Linear order:
1. `_safe_binary_frame(scored_df)` keeps only rows with valid `pd` and label.
2. `_derive_predictions()` decides predicted classes from approvals or threshold.
3. `_confusion_payload()` builds TP / FP / TN / FN payload.
4. Calculate live metrics:
   - ROC AUC
   - PR AUC
   - Brier
   - Log loss
   - Accuracy
   - Precision
   - Recall
   - F1
   - Specificity
   - KS
   - Calibration error
5. `_registry_payload(registry_entry)` parses stored validation and test metrics.
6. Build chart payloads:
   - `_calibration_bins()`
   - `_threshold_curve()`
   - `_performance_by_group()` for risk bands and approval groups
   - `_pd_histogram()`
7. `_comparison_metrics()` aligns valid/test/live metrics into one table.
8. `_build_insights()` creates text insights about drift, calibration, and confusion behavior.

### 13.3 Fallback behavior
If live snapshot cannot be loaded:
1. Route catches the failure.
2. Builds empty `scored_frame` and minimal `snapshot_meta`.
3. Returns registry-only performance payload with:
   - `status = degraded`
   - `live_status = unavailable`
   - `live_error = ...`

Final output:
- JSON payload for the model-performance dashboard.

---

## 14. Backend Route Flow: `GET /health`

File: `api/scoring_api.py`
Frontend callers:
- `App.jsx` on mount to test API status
- `SystemHealth.jsx` for full ops dashboard

Linear order:
1. Route `health()` starts.
2. `_api_cache.snapshot(now=time())` returns cache state.
3. `describe_active_production_model()` returns active model metadata.
4. `_check_mongo_live()` runs `get_database().command("ping")`.
5. The route builds one large operational JSON payload containing:
   - overall status
   - mongo status
   - production model metadata
   - dataset cache state
   - history cache state
   - scored snapshot state
   - mongo index report
   - auto-refresh state
6. If Mongo is down, route returns `503` with degraded payload.
7. Otherwise route returns `200` with the same payload structure.

Final output:
- Operational health JSON.

---

## 15. Backend Route Flow: `POST /cache/refresh`

File: `api/scoring_api.py`
Frontend callers:
- `Dashboard.jsx`
- `SystemHealth.jsx`

Linear order:
1. Route `refresh_cache()` starts.
2. Calls `_api_cache.refresh()`.
3. `refresh()` forces:
   - full dataset refresh from Mongo
   - scored snapshot rebuild
4. Route then calls `_api_cache.snapshot()`.
5. Returns JSON with refresh confirmation and current snapshot ID.

Final output:
- Small operational response confirming cache refresh.

---

## 16. Frontend Flow: Page by Page

File: `credit-risk-dashboard/src/App.jsx`

App-level flow:
1. `App.jsx` decides which page is active.
2. Each page uses `api.js` for all HTTP calls.
3. `API_BASE_URL` from frontend env decides which FastAPI server is hit.

### 16.1 Shared frontend HTTP layer
File: `credit-risk-dashboard/src/services/api.js`

Main functions:
- `request(path, options)`
- `api.scoreAll(...)`
- `api.fetchScoreAllWindow(...)`
- `api.scoreCustomer(...)`
- `api.scoreManual(...)`
- `api.health()`
- `api.refreshCache()`
- `api.modelPerformance()`

Common order:
1. Build URL and query params.
2. Run `fetch()`.
3. If not ok, throw backend detail.
4. Return parsed JSON.

### 16.2 Dashboard page
File: `credit-risk-dashboard/src/views/Dashboard.jsx`

Flow:
1. `load()` calls `api.fetchScoreAllWindow(segment, { limit, maxPages })`.
2. `fetchScoreAllWindow()` may request page 1 and optionally more pages using `next_cursor`.
3. Dashboard reads:
   - `snapshot_summary`
   - `total_available`
   - `records`
4. Builds cards, charts, and top-defaulters table.
5. Clicking a row navigates to customer lookup with `{ customerId, snapshotId }`.

### 16.3 Score All page
File: `credit-risk-dashboard/src/views/ScoreAll.jsx`

Flow:
1. `loadFirstPage()` calls `api.scoreAll(segment, { limit, refresh })`.
2. `applySnapshot()` stores page metadata and rows.
3. `loadMore()` uses `pagination.next_cursor` to fetch the next page.
4. Rows are merged client-side.
5. Search, risk filter, approval filter, and sorting are applied on loaded rows in the browser.
6. Clicking a row navigates to customer lookup with `{ customerId, snapshotId }`.

### 16.4 Customer Lookup page
File: `credit-risk-dashboard/src/views/CustomerLookup.jsx`

Flow:
1. On mount or change, `useEffect()` triggers customer fetch.
2. It calls `api.scoreCustomer(segment, customerId, { limit, snapshotId })`.
3. If user typed the customer manually, `handleSearch()` clears `snapshotId`.
4. If user came from Score All, `snapshotId` is preserved.
5. Page renders:
   - profile
   - feature bars
   - suggested actions
   - history summary

### 16.5 Manual Score page
File: `credit-risk-dashboard/src/views/ManualScore.jsx`

Flow:
1. User fills ad-hoc invoice fields.
2. `submit()` builds one JSON payload.
3. `api.scoreManual(segment, payload)` hits `/score/{segment}`.
4. Returned JSON is shown as:
   - PD
   - score
   - risk band
   - approval
   - top features
   - suggestions
   - customer history if present

### 16.6 System Health page
File: `credit-risk-dashboard/src/views/SystemHealth.jsx`

Flow:
1. `load()` calls `api.health()`.
2. Page renders operational cards from the health payload.
3. `handleRefreshCache()` calls `api.refreshCache()` and reloads health.

### 16.7 Model Performance page
File: `credit-risk-dashboard/src/views/ModelPerformance.jsx`

Flow:
1. `load()` calls `api.modelPerformance(segment, { refresh })`.
2. Page renders:
   - live metrics
   - confusion matrix
   - comparison metrics
   - calibration curve
   - threshold sweep
   - risk band / approval performance
   - PD distribution
   - registry metadata and IV features

---

## 17. If You Want To Trace One Request Quickly

Use this reading order.

### For `/score-all`
1. `credit-risk-dashboard/src/views/ScoreAll.jsx`
2. `credit-risk-dashboard/src/services/api.js`
3. `api/scoring_api.py` -> `score_all()`
4. `api/cache.py` -> `get_scored_page()`
5. `api/cache.py` -> `load_scored_snapshot()`
6. `api/cache.py` -> `_build_scored_snapshot()`
7. `pipeline/runner.py` -> `score_mongo_frame()`
8. `pipeline/risk_main.py` -> `build_risk_main_scoring_frame()`
9. `pipeline/risk_canonical.py` -> `fetch_risk_main_frame()` + `canonicalize_risk_main_frame()`
10. `features/engineering.py` -> `build_risk_main_feature_frame()`
11. `scoring/model.py` -> `score_production_frame()`
12. `api/response_builder.py` -> `response_from_raw()` + `_shape_response_frame()`

### For `/score-customer`
1. `credit-risk-dashboard/src/views/CustomerLookup.jsx`
2. `credit-risk-dashboard/src/services/api.js`
3. `api/scoring_api.py` -> `score_customer()`
4. then either:
   - snapshot path -> `api/cache.py` -> `get_snapshot_customer_frame()`
   - live path -> `build_scored_dataset()` -> `build_scored_frame()` -> pipeline -> scoring
5. `api/response_builder.py` -> `_build_customer_summary_payload()`

### For `/score`
1. `credit-risk-dashboard/src/views/ManualScore.jsx`
2. `credit-risk-dashboard/src/services/api.js`
3. `api/scoring_api.py` -> `score()`
4. `api/request_builder.py` -> `build_manual_request_frame()`
5. `pipeline/risk_main.py` -> `build_risk_main_manual_request_frame()`
6. `pipeline/runner.py` -> `score_mongo_frame()`
7. `scoring/model.py` -> `score_production_frame()`
8. `api/response_builder.py` -> lean response normalization

---

## 18. One Sentence Summary Per Layer

- `config.py`: reads env safely
- `dbconnect.py`: gives one Mongo DB handle
- `pipeline/risk_canonical.py`: fetches projected Mongo docs and canonicalizes them
- `pipeline/risk_main.py`: builds the live scoring frame and manual request frame
- `features/engineering.py`: creates model features
- `scoring/model.py`: loads the champion model and scores rows
- `api/cache.py`: keeps full dataset and scored snapshots warm
- `api/response_builder.py`: turns raw + scored DataFrames into JSON-ready contract rows
- `api/scoring_api.py`: exposes routes and chooses which path to run
- `credit-risk-dashboard/src/services/api.js`: frontend HTTP wrapper
- `credit-risk-dashboard/src/views/*.jsx`: UI pages that consume the JSON

---

## 19. The Simplest End-to-End Mental Model

For most flows, the project is doing this:

1. read env
2. connect to Mongo
3. fetch projected `Risk.Main` documents
4. flatten them
5. canonicalize them
6. add customer history
7. build features
8. load champion model + preprocessor
9. predict PD
10. derive score / risk band / approval / top features
11. shape the response contract
12. return JSON to the dashboard

The only difference between endpoints is where they start and whether they use:
- live scoring from Mongo, or
- cached scored snapshots
