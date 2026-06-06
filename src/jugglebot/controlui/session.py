"""Live controller session state and transport orchestration."""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue

from .history import TelemetryHistory
from .models import SessionConfig, TelemetryFrame, normalize_telemetry
from .transport import CommandClient, TelemetryListener, queue_put_latest


class LiveRobotSession:
    def __init__(self, config: SessionConfig, history: TelemetryHistory, status_cb=None):
        self.config = config
        self.history = history
        self.status_cb = status_cb
        self._cmd_queue: Queue = Queue(maxsize=1)
        self._raw_telem_queue: Queue = Queue()
        self._status_lock = threading.Lock()
        self._command_status = "TCP idle"
        self._telemetry_status = "Telemetry idle"
        self._latest_frame: TelemetryFrame | None = None
        self._last_telem_received_s = 0.0
        self._command_client: CommandClient | None = None
        self._telemetry_listener: TelemetryListener | None = None

    def start(self) -> None:
        if self._telemetry_listener is None:
            self._telemetry_listener = TelemetryListener(
                self.config.udp_port,
                self._raw_telem_queue,
                status_cb=self._set_telemetry_status,
            )
            self._telemetry_listener.start()
        if self._command_client is None:
            self._command_client = CommandClient(
                self.config.host,
                self.config.tcp_port,
                self._cmd_queue,
                status_cb=self._set_command_status,
            )
            self._command_client.start()

    def stop(self) -> None:
        if self._telemetry_listener is not None:
            self._telemetry_listener.stop()
            self._telemetry_listener.join(timeout=1.0)
            self._telemetry_listener = None
        if self._command_client is not None:
            self._command_client.stop()
            self._command_client.join(timeout=1.0)
            self._command_client = None

    def send_command(self, cmd: dict) -> None:
        queue_put_latest(self._cmd_queue, cmd)

    def poll(self) -> int:
        updated = 0
        while True:
            try:
                payload = self._raw_telem_queue.get_nowait()
            except Empty:
                break
            frame = normalize_telemetry(payload, source_id=self.config.session_id, receipt_time_s=time.time())
            self.history.append(frame)
            self._latest_frame = frame
            self._last_telem_received_s = frame.receipt_time_s
            updated += 1
        return updated

    @property
    def latest_frame(self) -> TelemetryFrame | None:
        if self._latest_frame is not None:
            return self._latest_frame
        return self.history.latest_frame

    @property
    def command_status(self) -> str:
        with self._status_lock:
            return self._command_status

    @property
    def telemetry_status(self) -> str:
        with self._status_lock:
            return self._telemetry_status

    @property
    def last_telem_received_s(self) -> float:
        return float(self._last_telem_received_s)

    def has_recent_telemetry(self, timeout_s: float) -> bool:
        last = self._last_telem_received_s
        if last <= 0.0:
            return False
        return (time.time() - last) <= float(timeout_s)

    def _set_command_status(self, text: str) -> None:
        with self._status_lock:
            self._command_status = str(text)
        if self.status_cb is not None:
            self.status_cb(str(text))

    def _set_telemetry_status(self, text: str) -> None:
        with self._status_lock:
            self._telemetry_status = str(text)
        if self.status_cb is not None:
            self.status_cb(str(text))
