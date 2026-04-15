from __future__ import annotations

class ScoreBroadcasterBridge:
    def __init__(self, stream_manager=None) -> None:
        self._stream_manager = stream_manager

    def set_stream_manager(self, stream_manager) -> None:
        self._stream_manager = stream_manager

    def set_event_loop(self, loop) -> None:
        if self._stream_manager is not None:
            self._stream_manager.set_risk_event_loop(loop)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def notify_snapshot_ready_threadsafe(self, snapshot: dict) -> None:
        if self._stream_manager is not None:
            self._stream_manager.notify_risk_snapshot_ready_threadsafe(snapshot)

    def notify_refresh_started_threadsafe(self, triggered_by: str = "auto_refresh") -> None:
        if self._stream_manager is not None:
            self._stream_manager.notify_risk_refresh_started_threadsafe(triggered_by)


score_broadcaster = ScoreBroadcasterBridge()
