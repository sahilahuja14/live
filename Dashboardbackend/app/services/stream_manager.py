import asyncio
import json
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, date
from fastapi import WebSocket
from ..database import get_analytics_database, get_all_async_queryFor_databases
from .query_builder import QueryBuilder
from ..schemas.analytics import PivotConfig
from .dashboard_stats import calculate_dashboard_stats, calculate_financial_stats

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
        print(f"WebSocket connected. Total: {len(self.active_connections)}")

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
            print(f"WebSocket disconnected. Total: {len(self.active_connections)}")

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
                    print(f"Error: Missing widgetId or config in payload: {payload}")
                    return
                
                print(f"Received Pivot Config from client for widget {widget_id}")
                
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
                print(f"Subscribed to Dashboard Stats: {payload}")
                self.client_stats_configs[websocket] = payload
                
                # Send Initial Stats
                stats = await self._run_stats_calculation(payload)
                response = { "type": "STATS_DATA", "payload": stats }
                await websocket.send_text(json.dumps(response, default=json_serial))
                
            elif msg_type == "SUBSCRIBE_FINANCE_STATS":
                payload = data.get("payload", {})
                print(f"Subscribed to Finance Stats: {payload}")
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
                
        except Exception as e:
            print(f"Error handling client message: {e}")

    async def _run_risk_refresh(self):
        if self._risk_refresh_callback is None:
            return
        try:
            await asyncio.to_thread(self._risk_refresh_callback)
        except Exception as e:
            print(f"Error running risk refresh: {e}")

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
        except Exception as e:
            print(f"Stats Calculation Error: {e}")
            return {"error": str(e)}

    async def _run_finance_calculation(self, config: dict) -> dict:
        try:
            range_param = config.get("range", "monthly")
            modes = config.get("modes", [])
            
            return await calculate_financial_stats(
                range_param=range_param,
                modes=modes
            )
        except Exception as e:
            print(f"Finance Calculation Error: {e}")
            return {"error": str(e)}

    async def _run_aggregation(self, config: PivotConfig) -> dict:
        # Re-use the existing logic from analytics router, but calling it directly
        try:
             # We need to use the sync PyMongo for aggregation or ensure we have async driver
             # The existing QueryBuilder builds a pipeline. 
             # We can use the async driver to run it.
             
             queryFor_dbs = get_all_async_queryFor_databases()
             pipeline = self.qb.build_pipeline(config)
             
             # DEBUG: Log Pipeline
             print(f"DEBUG: Pivot Config Rows: {[r.id for r in config.rows]}")
             print(f"DEBUG: Pivot Config Columns: {[c.id for c in config.columns]}")
             print(f"DEBUG: Pivot Config Values: {[v.fieldId for v in config.values]}")
             print(f"DEBUG: Generated Pipeline Project Stage: {pipeline[-1] if pipeline else 'EMPTY'}")
             
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
             
        except Exception as e:
            print(f"Aggregation Error: {e}")
            return {"error": str(e), "data": []}

    async def start_watching(self):
        if self.is_watching:
            return
        self.is_watching = True
        self._loop = asyncio.get_running_loop()
        
        # --- API SCHEDULING (Polling) ---
        self._watch_task = asyncio.create_task(self._api_scheduling_loop())
        print(f"Started API Scheduling Loop (Polling every {self.POLL_INTERVAL_SECONDS}s)")

    async def stop_watching(self):
        self.is_watching = False
        if getattr(self, '_watch_task', None):
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            
        print("Stopped API Scheduling Loop")

    # --- API SCHEDULING (Polling) ---
    async def _api_scheduling_loop(self):
        print(f"Starting scheduled polling of database every {self.POLL_INTERVAL_SECONDS} seconds...")
        
        # In the future, you could query the database for Max(updatedAt) here.
        # For now, we will simply trigger the broadcasts on a schedule. 
        # The caching logic in the broadcast functions handles deduplication.
        while self.is_watching:
            try:
                if self.active_connections:
                    if self._has_scheduled_poll_subscribers():
                        await self._handle_change("scheduled_poll")
                    await self._broadcast_risk_heartbeat()
            except Exception as e:
                print(f"Error in API Scheduling Loop: {e}")
            
            # Sleep for the interval before checking again
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

    def _has_scheduled_poll_subscribers(self) -> bool:
        return bool(
            self.client_configs
            or self.client_stats_configs
            or self.client_finance_configs
        )

    async def _handle_change(self, source: str):
        # Unified change handler for all collections
        print(f"Detected change in {source}")
        
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
                    print(f"Broadcasting buffered updates...")
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
                    except Exception as e:
                        print(f"Error executing config for widget {widget_id}: {e}")
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
                except Exception as e:
                    print(f"Error sending dynamic update for widget {widget_id}: {e}")

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
                except Exception as e:
                    print(f"Error calc stats for {stats_config}: {e}")
                    continue
            
            # Send
            try:
                response = {
                    "type": "STATS_DATA",
                    "payload": stats
                }
                await websocket.send_text(json.dumps(response, default=json_serial))
            except Exception as e:
                print(f"Error sending stats update: {e}")

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
                except Exception as e:
                    print(f"Error calc finance stats for {finance_config}: {e}")
                    continue
            
            # Send
            try:
                response = {
                    "type": "FINANCE_DATA",
                    "payload": stats
                }
                await websocket.send_text(json.dumps(response, default=json_serial))
            except Exception as e:
                print(f"Error sending finance stats update: {e}")

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

        customer_lookup: dict[tuple[str, str], dict] = {}
        for record in snapshot.get("records", []):
            customer_id = str(record.get("customer.customerId") or "").strip()
            segment = str(record.get("shipmentDetails.queryFor") or record.get("segment") or "all").strip().lower()
            if not customer_id:
                continue

            risk_payload = {
                "pd": record.get("pd"),
                "risk_band": record.get("risk_band"),
                "approval": record.get("approval"),
            }
            customer_lookup.setdefault((customer_id, segment), risk_payload)
            customer_lookup.setdefault((customer_id, "all"), risk_payload)

        for websocket in list(self.active_connections):
            subscriptions = self.client_risk_configs.get(websocket, set())
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
                except Exception as e:
                    print(f"Error sending risk customer update: {e}")

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
