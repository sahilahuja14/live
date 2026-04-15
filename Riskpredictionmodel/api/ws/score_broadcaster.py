from __future__ import annotations

from Dashboardbackend.app.services.stream_manager import stream_manager


class ScoreBroadcasterBridge:
    def set_event_loop(self, loop) -> None:
        stream_manager.set_risk_event_loop(loop)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def notify_snapshot_ready_threadsafe(self, snapshot: dict) -> None:
        stream_manager.notify_risk_snapshot_ready_threadsafe(snapshot)

    def notify_refresh_started_threadsafe(self, triggered_by: str = "auto_refresh") -> None:
        stream_manager.notify_risk_refresh_started_threadsafe(triggered_by)


score_broadcaster = ScoreBroadcasterBridge()
