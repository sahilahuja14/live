from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_live_db_name, get_live_mongo_uri, init_env

init_env()

from .analysis import (
    build_customer_history_response as _build_customer_history_response_impl,
    build_scored_dataset as _build_scored_dataset_impl,
    build_scored_frame as _build_scored_frame_impl,
    canonical_snapshot_for_rows as _canonical_snapshot_for_rows_impl,
    clean_customer_portfolio_frame as _clean_customer_portfolio_frame_impl,
    enrich_with_customer_history as _enrich_with_customer_history_impl,
    feature_snapshot_for_rows as _feature_snapshot_for_rows_impl,
    history_preview_limit as _history_preview_limit_impl,
    load_customer_portfolio_page_from_store as _load_customer_portfolio_page_from_store_impl,
    load_customer_summary_from_store as _load_customer_summary_from_store_impl,
    load_customer_summary_or_bootstrap as _load_customer_summary_or_bootstrap_impl,
    prepare_history_frame as _prepare_history_frame_impl,
    resolve_customer_lookup_input as _resolve_customer_lookup_input_impl,
    resolve_customer_lookup_key as _resolve_customer_lookup_key_impl,
)
from .cache import ApiCache
from .routers import (
    register_customer_routes,
    register_scoring_routes,
    register_system_routes,
)
from .services import CustomerService
from .settings import load_api_settings
from .ws import score_broadcaster
from ..dbconnect import get_live_database
from ..logging_config import get_logger
from ..scoring.model import load_production_artifacts


logger = get_logger(__name__)
SETTINGS = load_api_settings()
router = APIRouter(tags=["risk"])

# Keep a compatibility alias so existing tests and callers that still patch
# get_database do not break while the runtime stays live-db only.
get_database = get_live_database


def _resolve_threshold_override(segment: str) -> float | None:
    segment = (segment or "").strip().lower()
    if segment == "air" and SETTINGS.threshold_override_air is not None:
        return SETTINGS.threshold_override_air
    if segment == "ocean" and SETTINGS.threshold_override_ocean is not None:
        return SETTINGS.threshold_override_ocean
    return SETTINGS.threshold_override_global


_api_cache = ApiCache(
    dataset_ttl_seconds=SETTINGS.dataset_cache_ttl_seconds,
    history_ttl_seconds=SETTINGS.history_cache_ttl_seconds,
    auto_refresh_enabled=SETTINGS.auto_refresh_enabled,
    auto_refresh_interval_seconds=SETTINGS.auto_refresh_interval_seconds,
    scored_snapshot_retention_seconds=SETTINGS.scored_snapshot_retention_seconds,
    threshold_resolver=_resolve_threshold_override,
    broadcaster=score_broadcaster,
)
_customer_service = CustomerService(
    api_cache=_api_cache,
    threshold_resolver=_resolve_threshold_override,
)


def _prepare_history_frame(force_refresh: bool = False):
    return _prepare_history_frame_impl(_api_cache, force_refresh=force_refresh)


def _enrich_with_customer_history(df, force_refresh: bool = False):
    return _enrich_with_customer_history_impl(_api_cache, df, force_refresh=force_refresh)


def _history_preview_limit(payload):
    return _history_preview_limit_impl(payload)


def _feature_snapshot_for_rows(scoring_frame, page_indices, expected_features):
    return _feature_snapshot_for_rows_impl(scoring_frame, page_indices, expected_features)


def _canonical_snapshot_for_rows(raw_frame, page_indices):
    return _canonical_snapshot_for_rows_impl(raw_frame, page_indices)


def _load_customer_portfolio_page_from_store(*, segment: str, search: str | None, refresh: bool):
    return _load_customer_portfolio_page_from_store_impl(
        api_cache=_api_cache,
        segment=segment,
        search=search,
        refresh=refresh,
    )


def _load_customer_summary_from_store(*, segment: str, customer_id: str, refresh: bool):
    return _load_customer_summary_from_store_impl(
        api_cache=_api_cache,
        segment=segment,
        customer_id=customer_id,
        refresh=refresh,
    )


def _load_customer_summary_or_bootstrap(*, segment: str, customer_id: str, refresh: bool):
    return _load_customer_summary_or_bootstrap_impl(
        api_cache=_api_cache,
        customer_service=_customer_service,
        segment=segment,
        customer_id=customer_id,
        refresh=refresh,
    )


def _resolve_customer_lookup_input(*, customer_id: str | None = None, customer_name: str | None = None, invoice_no: str | None = None, query: str | None = None):
    return _resolve_customer_lookup_input_impl(
        customer_id=customer_id,
        customer_name=customer_name,
        invoice_no=invoice_no,
        query=query,
    )


def _resolve_customer_lookup_key(*, segment: str, lookup_value: str, refresh: bool):
    return _resolve_customer_lookup_key_impl(
        customer_service=_customer_service,
        segment=segment,
        lookup_value=lookup_value,
        refresh=refresh,
    )


def _clean_customer_portfolio_frame(customer_frame):
    return _clean_customer_portfolio_frame_impl(customer_frame)


def _build_customer_history_response(
    *,
    segment: str,
    customer_key: str,
    page_size: int,
    offset: int,
    include_features: bool,
    include_canonical: bool,
    refresh: bool,
    customer_summary_override: dict | None = None,
):
    return _build_customer_history_response_impl(
        customer_service=_customer_service,
        threshold_resolver=_resolve_threshold_override,
        segment=segment,
        customer_key=customer_key,
        page_size=page_size,
        offset=offset,
        include_features=include_features,
        include_canonical=include_canonical,
        refresh=refresh,
        customer_summary_override=customer_summary_override,
    )


def build_scored_frame(segment: str, limit: int | None = None, customer_id: str | None = None, force_refresh: bool = False):
    return _build_scored_frame_impl(
        api_cache=_api_cache,
        threshold_resolver=_resolve_threshold_override,
        segment=segment,
        limit=limit,
        customer_id=customer_id,
        force_refresh=force_refresh,
    )


def build_scored_dataset(segment: str, limit: int | None = None, customer_id: str | None = None, force_refresh: bool = False):
    return _build_scored_dataset_impl(
        api_cache=_api_cache,
        threshold_resolver=_resolve_threshold_override,
        segment=segment,
        limit=limit,
        customer_id=customer_id,
        force_refresh=force_refresh,
    )


def _startup_checks() -> None:
    if not get_live_mongo_uri():
        raise RuntimeError("LIVE_MONGO_URI is required but not set.")
    if not get_live_db_name():
        raise RuntimeError("LIVE_DB_NAME is required but not set.")
    load_production_artifacts()
    logger.info("Startup checks completed successfully source_mode=live_collections")


def _check_live_mongo_live() -> bool:
    try:
        get_database().command("ping")
        return True
    except Exception:
        logger.warning("Live MongoDB ping failed", exc_info=True)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_checks()
    score_broadcaster.set_event_loop(asyncio.get_running_loop())
    await score_broadcaster.start()
    _api_cache.start()
    try:
        yield
    finally:
        _api_cache.stop()
        await score_broadcaster.stop()


register_scoring_routes(
    router=router,
    logger=logger,
    settings=SETTINGS,
    api_cache=_api_cache,
    threshold_resolver=_resolve_threshold_override,
    prepare_history_frame=_prepare_history_frame_impl,
    enrich_with_customer_history=_enrich_with_customer_history_impl,
    build_scored_frame=build_scored_frame,
    build_scored_dataset=build_scored_dataset,
)

register_customer_routes(
    router=router,
    logger=logger,
    settings=SETTINGS,
    api_cache=_api_cache,
    customer_service=_customer_service,
    threshold_resolver=_resolve_threshold_override,
    history_preview_limit=_history_preview_limit_impl,
    load_customer_portfolio_page_from_store=_load_customer_portfolio_page_from_store_impl,
    load_customer_summary_or_bootstrap=_load_customer_summary_or_bootstrap_impl,
    resolve_customer_lookup_input=_resolve_customer_lookup_input_impl,
    resolve_customer_lookup_key=_resolve_customer_lookup_key_impl,
    clean_customer_portfolio_frame=_clean_customer_portfolio_frame_impl,
    build_customer_history_response=_build_customer_history_response_impl,
)

register_system_routes(
    router=router,
    logger=logger,
    settings=SETTINGS,
    api_cache=_api_cache,
    score_broadcaster=score_broadcaster,
    check_live_mongo_live=_check_live_mongo_live,
)


app = FastAPI(lifespan=lifespan)
if SETTINGS.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.cors_allowed_origins),
        allow_credentials=SETTINGS.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.include_router(router)
