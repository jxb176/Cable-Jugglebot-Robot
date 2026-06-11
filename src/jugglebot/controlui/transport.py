"""Transport primitives for the controller GUI."""

from __future__ import annotations

import json
import socket
import threading
import time
from queue import Empty, Queue


UDP_RECV_BUFFER_BYTES = 65535


def queue_put_latest(queue: Queue, item) -> None:
    """Keep only the newest item in a queue."""
    try:
        while True:
            queue.get_nowait()
    except Empty:
        pass
    queue.put(item)


class CommandClient(threading.Thread):
    """TCP client that sends commands to the robot with auto-reconnect."""

    def __init__(self, host: str, port: int, cmd_queue: Queue, manual_sample_queue: Queue, status_cb=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = int(port)
        self.cmd_queue = cmd_queue
        self.manual_sample_queue = manual_sample_queue
        self.status_cb = status_cb
        self._stop_event = threading.Event()
        self._sock = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._emit_status("Connecting to robot (TCP)...")
                self._sock = socket.create_connection((self.host, self.port), timeout=5)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._emit_status("Connected (TCP)")

                while not self._stop_event.is_set():
                    manual_sample = self._drain_manual_sample()
                    if manual_sample is not None:
                        self._send_cmd(manual_sample)
                        continue
                    try:
                        cmd = self.cmd_queue.get(timeout=0.005)
                        self._send_cmd(cmd)
                        continue
                    except Empty:
                        pass
                    manual_sample = self._drain_manual_sample()
                    if manual_sample is not None:
                        self._send_cmd(manual_sample)
            except Exception as exc:
                if not self._stop_event.is_set():
                    self._emit_status(f"TCP disconnected: {exc}. Reconnecting in 1s...")
                    time.sleep(1.0)
            finally:
                self._close()

    def stop(self) -> None:
        self._stop_event.set()
        self._close()

    def _send_cmd(self, cmd_value) -> None:
        if not self._sock or not isinstance(cmd_value, dict):
            return
        msg = json.dumps(cmd_value) + "\n"
        self._sock.sendall(msg.encode("utf-8"))

    def _drain_manual_sample(self):
        manual_sample = None
        try:
            while True:
                manual_sample = self.manual_sample_queue.get_nowait()
        except Empty:
            pass
        return manual_sample

    def _close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    def _emit_status(self, text: str) -> None:
        if self.status_cb is not None:
            self.status_cb(text)


class TelemetryListener(threading.Thread):
    """UDP listener that forwards telemetry payloads to a queue."""

    def __init__(self, udp_port: int, telem_queue: Queue, status_cb=None):
        super().__init__(daemon=True)
        self.udp_port = int(udp_port)
        self.telem_queue = telem_queue
        self.status_cb = status_cb
        self._stop_event = threading.Event()
        self._sock = None

    def run(self) -> None:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", self.udp_port))
            sock.settimeout(0.2)
        except OSError as exc:
            self._emit_status(f"Telemetry bind failed: {exc}")
            return

        self._sock = sock
        self._emit_status(f"Telemetry: listening UDP :{self.udp_port}")

        try:
            while not self._stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(UDP_RECV_BUFFER_BYTES)
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self._stop_event.is_set():
                        self._emit_status(f"Telemetry socket error: {exc}")
                        time.sleep(0.05)
                    continue
                try:
                    telem = json.loads(data.decode("utf-8"))
                    self.telem_queue.put(telem)
                except Exception as exc:
                    self._emit_status(f"Telemetry error: {exc}")
                    time.sleep(0.05)
        finally:
            self._close()

    def stop(self) -> None:
        self._stop_event.set()
        self._close()

    def _close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None

    def _emit_status(self, text: str) -> None:
        if self.status_cb is not None:
            self.status_cb(text)
