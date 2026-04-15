import asyncio
import json
import os
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, date
from fastapi import WebSocket
from ..database import get_analytics_database, get_all_async_queryFor_databases
from .query_builder import QueryBuilder
from ..schemas.analytics import PivotConfig
from .dashboard_stats import calculate_dashboard_stats, calculate_financial_stats


logger = logging.getLogger(__name__)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

class StreamManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Store client config: WebSocket -> widgetId -> PivotConfig (as dict)
        self.client_configs: Dict[WebSocket, Dict[str, dict]] = {} 
        # Store client stats config: WebSocket -> StatsConfig (dict)
        self.client_stats_configs: Dict[WebSocket, dict] = {}
        # Store client finance stats config: WebSocket -> FinanceConfig (dict)
        self.client_finance_configs: Dict[WebSocket, dict] = {}
        # Store risk customer subscriptions: WebSocket -> set[(customer_id, segment)]
        self.client_risk_configs: Dict[WebSocket, set[tuple[str, str]]] = {}

        self.is_watching = False
        self._watch_task = None
        self._loop = None
        self.qb = QueryBuilder()
        self._risk_refresh_callback: Callable[[], None] | None = None
        self._risk_snapshot_meta: dict = {}

        # --- API SCHEDULING (Polling) ---
        self.POLL_INTERVAL_SECONDS = max(
            int(os.getenv("DASHBOARD_STREAM_POLL_INTERVAL_SECONDS", "300")),
            30,
        )
        
        # Debounce/Throttle state
        self.DEBOUNCE_SECONDS = 2.0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_risk_configs[websocket] = set()
        logger.info("WebSocket connected total=%d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if websocket in self.client_configs:
                del self.client_configs[websocket]
            if websocket in self.client_stats_configs:
                del self.client_stats_configs[websocket]
            if websocket in self.client_finance_configs:
                del self.client_finance_configs[websocket]
            if websocket in self.client_risk_configs:
                del self.client_risk_configs[websocket]
            logger.info("WebSocket disconnected total=%d", len(self.active_connections))

    def configure_risk_runtime(self, refresh_callback: Callable[[], None] | None = None):
        self._risk_refresh_callback = refresh_callback

    def set_risk_event_loop(self, loop):
        self._loop = loop

    async def handle_client_message(self, websocket: WebSocket, message_text: str):
        try:
            data = json.loads(message_text)
            msg_type = data.get("type")
            
            if msg_type == "SET_PIVOT_CONFIG":
                payload = data.get("payload")
                widget_id = payload.get("widgetId")
                config_data = payload.get("config")
                
                if not widget_id or not config_data:
                    logger.warning("Missing widgetId or config in payload payload=%s", payload)
                    return
                
                logger.info("Received pivot config widget_id=%s", widget_id)
                
                # Initialize nested dict if needed
                if websocket not in self.client_configs:
                    self.client_configs[websocket] = {}
                
                self.client_configs[websocket][widget_id] = config_data
                config_obj = PivotConfig(**config_data)
                stats = await self._run_aggregation(config_obj)
                response = { 
                    "type": "PIVOT_DATA", 
                    "payload": {
                        "widgetId": widget_id,
                        "data": stats
                    }
                }
                await websocket.send_text(json.dumps(response, default=json_serial))
                
            elif msg_type == "SUBSCRIBE_DASHBOARD_STATS":
                payload = data.get("payload", {})
                logger.info("Subscribed to dashboard stats payload=%s", payload)
                self.client_stats_configs[websocket] = payload
                
                # Send Initial Stats
                stats = await self._run_stats_calculation(payload)
                response = { "type": "STATS_DATA", "payload": stats }
                await websocket.send_text(json.dumps(response, default=json_serial))
                
            elif msg_type == "SUBSCRIBE_FINANCE_STATS":
                payload = data.get("payload", {})
                logger.info("Subscribed to finance stats payload=%s", payload)
                self.client_finance_configs[websocket] = payload
                
                # Send Initial Finance Stats
                stats = await self._run_finance_calculation(payload)
                response = { "type": "FINANCE_DATA", "payload": stats }
                await websocket.send_text(json.dumps(response, default=json_serial))

            elif msg_type == "SUBSCRIBE_CUSTOMER":
                payload = data.get("payload", {})
                customer_id = str(payload.get("customer_id") or "").strip()
                segment = str(payload.get("segment") or "all").strip().lower()
                if customer_id:
                    self.client_risk_configs.setdefault(websocket, set()).add((customer_id, segment))

            elif msg_type == "UNSUBSCRIBE_CUSTOMER":
                payload = data.get("payload", {})
                customer_id = str(payload.get("customer_id") or "").strip()
                segment = str(payload.get("segment") or "all").strip().lower()
                if customer_id:
                    self.client_risk_configs.setdefault(websocket, set()).discard((customer_id, segment))

            elif msg_type == "REQUEST_REFRESH":
                await self._broadcast_risk_refresh_started(triggered_by="client_request")
                if self._risk_refresh_callback is not None:
                    asyncio.create_task(self._run_risk_refresh())
                
        except Exception:
            logger.exception("Error handling client websocket message")

    async def _run_risk_refresh(self):
        if self._risk_refresh_callback is None:
            return
        try:
            await asyncio.to_thread(self._risk_refresh_callback)
        except Exception:
            logger.exception("Error running risk refresh")

    async def _run_stats_calculation(self, config: dict) -> dict:
        try:
            # Extract filters from config
            # config matches FetchShipmentsParams-ish or just range/type/queryType
            range_param = config.get("range", "weekly")
            shipment_type = config.get("type", "all")
            query_type = config.get("queryType", "all")
            
            # Use existing service
            return await calculate_dashboard_stats(
                range_param=range_param,
                shipment_type=shipment_type,
                query_type=query_type
            )
        except Exception as exc:
            logger.exception("Stats calculation error")
            return {"error": str(exc)}

    async def _run_finance_calculation(self, config: dict) -> dict:
        try:
            range_param = config.get("range", "monthly")
            modes = config.get("modes", [])
            
            return await calculate_financial_stats(
                range_param=range_param,
                modes=modes
            )
        except Exception as exc:
            logger.exception("Finance calculation error")
            return {"error": str(exc)}

    async def _run_aggregation(self, config: PivotConfig) -> dict:
        # Re-use the existing logic from analytics router, but calling it directly
        try:
             # We need to use the sync PyMongo for aggregation or ensure we have async driver
             # The existing QueryBuilder builds a pipeline. 
             # We can use the async driver to run it.
             
             queryFor_dbs = get_all_async_queryFor_databases()
             pipeline = self.qb.build_pipeline(config)
             
             # DEBUG: Log Pipeline
             logger.debug("Pivot config rows=%s", [r.id for r in config.rows])
             logger.debug("Pivot config columns=%s", [c.id for c in config.columns])
             logger.debug("Pivot config values=%s", [v.fieldId for v in config.values])
             logger.debug("Generated pipeline tail=%s", pipeline[-1] if pipeline else "EMPTY")
             
             async def run_query(db):
                 cursor = await db.queries.aggregate(pipeline)
                 return await cursor.to_list(length=None)
                 
             tasks = [run_query(db) for db in queryFor_dbs.values()]
             all_results = await asyncio.gather(*tasks)
             
             results = []
             for res_list in all_results:
                 results.extend(res_list)
             
             # Flatten results (same logic as router)
             flattened_results = []
             for doc in results:
                flat = {}
                if isinstance(doc.get('_id'), dict):
                    for k, v in doc['_id'].items():
                        flat[k] = v
                elif doc.get('_id') is not None:
                    flat['group'] = doc['_id']
                
                for k, v in doc.items():
                    if k != '_id':
                        flat[k] = v
                flattened_results.append(flat)
                
             return {"data": flattened_results}
             
        except Exception as exc:
            logger.exception("Aggregation error")
            return {"error": str(exc), "data": []}

    async def start_watching(self):
        if self.is_watching:
            return
        self.is_watching = True
        self._loop = asyncio.get_running_loop()
        
        # --- API SCHEDULING (Polling) ---
        self._watch_task = asyncio.create_task(self._api_scheduling_loop())
        logger.info("Started API scheduling loop poll_interval_seconds=%d", self.POLL_INTERVAL_SECONDS)

    async def stop_watching(self):
        self.is_watching = False
        if getattr(self, '_watch_task', None):
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Stopped API scheduling loop")

    # --- API SCHEDULING (Polling) ---
    async def _api_scheduling_loop(self):
        logger.info("Starting scheduled polling poll_interval_seconds=%d", self.POLL_INTERVAL_SECONDS)
        heartbeat_interval_seconds = 30
        last_poll_ts = 0.0
        while self.is_watching:
            try:
                if self.active_connections:
                    now = time.time()
                    await self._broadcast_risk_heartbeat()
                    if self._has_scheduled_poll_subscribers() and (now - last_poll_ts) >= self.POLL_INTERVAL_SECONDS:
                        last_poll_ts = now
                        await self._handle_change("scheduled_poll")
            except Exception:
                logger.exception("Error in API scheduling loop")
            await asyncio.sleep(heartbeat_interval_seconds)

    def _has_scheduled_poll_subscribers(self) -> bool:
        return bool(
            self.client_configs
            or self.client_stats_configs
            or self.client_finance_configs
        )

    async def _handle_change(self, source: str):
        # Unified change handler for all collections
        logger.info("Detected change source=%s", source)
        
        # Mark update as needed
        self._needs_update = True
        
        # If already a debounce loop running, let it handle it
        if getattr(self, '_is_debouncing', False):
            return

        self._is_debouncing = True
        
        try:
            while self._needs_update:
                # Clear flag, wait, then check/broadcast
                # We wait FIRST to gather multiple quick updates
                await asyncio.sleep(self.DEBOUNCE_SECONDS)
                
                # Check if more updates came in while sleeping? 
                # Actually, we just broadcast current state. 
                # If update came in at T=0.1, we sleep till T=2.0. Broadcast.
                # If update came at T=1.9, we sleep till T=2.0. Broadcast.
                # Ideally we want to broadcast recent state.
                
                if self._needs_update:
                    # Clear it before broadcasting (so new changes separate)
                    self._needs_update = False
                    logger.info("Broadcasting buffered updates")
                    await self._broadcast_dynamic_updates()
                    await self._broadcast_stats()
                    await self._broadcast_finance_stats()
        finally:
            self._is_debouncing = False

    async def _broadcast_dynamic_updates(self):
        # 1. Identify unique configs to avoid redundant queries
        # Map: ConfigHash -> Result
        config_cache = {}
        
        # 2. Iterate all active clients
        for websocket in self.active_connections:
            # Check if client has configs
            widget_configs = self.client_configs.get(websocket)
            if not widget_configs:
                continue
            
            # Iterate over all widget configs for this client
            for widget_id, raw_config in widget_configs.items():
                # Create a hashable key for the config
                config_json = json.dumps(raw_config, sort_keys=True)
                config_hash = hashlib.md5(config_json.encode('utf-8')).hexdigest()
                
                # Check Cache
                if config_hash in config_cache:
                    result = config_cache[config_hash]
                else:
                    # Run Aggregation
                    try:
                        config_obj = PivotConfig(**raw_config)
                        result = await self._run_aggregation(config_obj)
                        config_cache[config_hash] = result
                    except Exception:
                        logger.exception("Error executing config widget_id=%s", widget_id)
                        continue
                
                # Send Update with widgetId
                try:
                    response = {
                        "type": "PIVOT_DATA",
                        "payload": {
                            "widgetId": widget_id,
                            "data": result
                        }
                    }
                    await websocket.send_text(json.dumps(response, default=json_serial))
                except Exception:
                    logger.exception("Error sending dynamic update widget_id=%s", widget_id)

    async def _broadcast_stats(self):
        # Broadcast Dashboard Stats to subscribed clients
        stats_cache = {}
        
        for websocket in self.active_connections:
            # Check if client subscribed to stats
            stats_config = self.client_stats_configs.get(websocket)
            if not stats_config:
                continue
            
            # Hash Config for Caching
            config_json = json.dumps(stats_config, sort_keys=True)
            config_hash = hashlib.md5(config_json.encode('utf-8')).hexdigest()
            
            if config_hash in stats_cache:
                stats = stats_cache[config_hash]
            else:
                try:
                    stats = await self._run_stats_calculation(stats_config)
                    stats_cache[config_hash] = stats
                except Exception:
                    logger.exception("Error calculating dashboard stats config=%s", stats_config)
                    continue
            
            # Send
            try:
                response = {
                    "type": "STATS_DATA",
                    "payload": stats
                }
                await websocket.send_text(json.dumps(response, default=json_serial))
            except Exception:
                logger.exception("Error sending stats update")

    async def _broadcast_finance_stats(self):
        # Broadcast Finance Stats to subscribed clients
        finance_cache = {}
        
        for websocket in self.active_connections:
            # Check if client subscribed to finance stats
            finance_config = self.client_finance_configs.get(websocket)
            if not finance_config:
                continue
            
            # Hash Config for Caching
            config_json = json.dumps(finance_config, sort_keys=True)
            config_hash = hashlib.md5(config_json.encode('utf-8')).hexdigest()
            
            if config_hash in finance_cache:
                stats = finance_cache[config_hash]
            else:
                try:
                    stats = await self._run_finance_calculation(finance_config)
                    finance_cache[config_hash] = stats
                except Exception:
                    logger.exception("Error calculating finance stats config=%s", finance_config)
                    continue
            
            # Send
            try:
                response = {
                    "type": "FINANCE_DATA",
                    "payload": stats
                }
                await websocket.send_text(json.dumps(response, default=json_serial))
            except Exception:
                logger.exception("Error sending finance stats update")

    async def _broadcast_risk_refresh_started(self, triggered_by: str = "auto_refresh"):
        await self._broadcast_risk_message(
            {
                "type": "REFRESH_STARTED",
                "payload": {"triggered_by": triggered_by},
            }
        )

    async def publish_risk_snapshot(self, snapshot: dict):
        self._risk_snapshot_meta = {
            "snapshot_id": snapshot.get("snapshot_id"),
            "generated_at": snapshot.get("generated_at"),
            "segment_counts": snapshot.get("segment_counts", {}),
            "rows": snapshot.get("rows", 0),
            "ts": snapshot.get("ts"),
        }
        await self._broadcast_risk_message(
            {
                "type": "SNAPSHOT_READY",
                "payload": {
                    "snapshot_id": snapshot.get("snapshot_id"),
                    "generated_at": snapshot.get("generated_at"),
                    "segment_counts": snapshot.get("segment_counts", {}),
                    "rows": snapshot.get("rows", 0),
                },
            }
        )
        await self._broadcast_risk_customer_updates(snapshot)

    def notify_risk_snapshot_ready_threadsafe(self, snapshot: dict):
        if self._loop is None or not self.active_connections:
            return
        asyncio.run_coroutine_threadsafe(self.publish_risk_snapshot(snapshot), self._loop)

    def notify_risk_refresh_started_threadsafe(self, triggered_by: str = "auto_refresh"):
        if self._loop is None or not self.active_connections:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_risk_refresh_started(triggered_by),
            self._loop,
        )

    async def _broadcast_risk_customer_updates(self, snapshot: dict):
        if not self.active_connections:
            return

        wanted: set[tuple[str, str]] = set()
        for subscriptions in self.client_risk_configs.values():
            wanted.update(subscriptions)
        if not wanted:
            return

        try:
            from Riskpredictionmodel.api.cache.customer_risk_store import CustomerRiskStore
        except Exception:
            logger.exception("Unable to import CustomerRiskStore for risk websocket updates")
            return

        store = CustomerRiskStore()
        customer_lookup: dict[tuple[str, str], dict] = {}
        for customer_id, segment in wanted:
            record = store.load_customer_record(segment=segment, customer_id=customer_id)
            if record is None:
                continue
            customer_lookup[(customer_id, segment)] = {
                "pd": record.get("pd"),
                "risk_band": record.get("risk_band"),
                "approval": record.get("approval"),
            }

        for websocket, subscriptions in list(self.client_risk_configs.items()):
            for customer_id, segment in subscriptions:
                risk_payload = customer_lookup.get((customer_id, segment))
                if risk_payload is None:
                    continue
                try:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "CUSTOMER_UPDATED",
                                "payload": {
                                    "customer_id": customer_id,
                                    "segment": segment,
                                    "snapshot_id": snapshot.get("snapshot_id"),
                                    **risk_payload,
                                },
                            },
                            default=json_serial,
                        )
                    )
                except Exception:
                    logger.exception("Error sending risk customer update customer_id=%s segment=%s", customer_id, segment)

    async def _broadcast_risk_heartbeat(self):
        if not self.active_connections:
            return

        snapshot_ts = self._risk_snapshot_meta.get("ts")
        snapshot_age_seconds = None
        if snapshot_ts:
            snapshot_age_seconds = max(int(time.time() - float(snapshot_ts)), 0)

        await self._broadcast_risk_message(
            {
                "type": "HEARTBEAT",
                "payload": {
                    "status": "ok",
                    "snapshot_age_seconds": snapshot_age_seconds,
                    "snapshot_id": self._risk_snapshot_meta.get("snapshot_id"),
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                },
            }
        )

    async def _broadcast_risk_message(self, message: dict):
        dead_connections = []
        for websocket in list(self.active_connections):
            try:
                await websocket.send_text(json.dumps(message, default=json_serial))
            except Exception:
                dead_connections.append(websocket)

        for websocket in dead_connections:
            self.disconnect(websocket)

stream_manager = StreamManager()
