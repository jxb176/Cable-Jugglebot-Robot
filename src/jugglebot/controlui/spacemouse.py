"""Optional SpaceMouse backend integration for the control UI."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
import timeit


@dataclass(frozen=True, slots=True)
class SpaceMouseSample:
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float
    device_time_s: float | None = None
    device_age_ms: float | None = None
    reports_drained: int = 0
    poll_interval_ms: float | None = None


class SpaceMouseBackend:
    def is_available(self) -> bool:
        raise NotImplementedError

    def status_text(self) -> str:
        raise NotImplementedError

    def open(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def read_sample(self) -> SpaceMouseSample | None:
        raise NotImplementedError


class SpaceMouseWorker(threading.Thread):
    def __init__(self, backend: SpaceMouseBackend, sample_cb=None, *, poll_interval_s: float = 0.005):
        super().__init__(daemon=True)
        self.backend = backend
        self.sample_cb = sample_cb
        self.poll_interval_s = max(0.0, float(poll_interval_s))
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_sample: SpaceMouseSample | None = None
        self._error_text: str | None = None
        self._last_poll_perf_s: float | None = None

    def stop(self) -> None:
        self._stop_event.set()

    def latest_sample(self) -> SpaceMouseSample | None:
        with self._lock:
            return self._latest_sample

    def latest_error(self) -> str | None:
        with self._lock:
            return self._error_text

    def run(self) -> None:
        while not self._stop_event.is_set():
            now_perf_s = time.perf_counter()
            poll_interval_ms = None if self._last_poll_perf_s is None else 1000.0 * (now_perf_s - self._last_poll_perf_s)
            self._last_poll_perf_s = now_perf_s
            try:
                sample = self.backend.read_sample()
            except Exception as exc:
                with self._lock:
                    self._error_text = str(exc)
                return
            if sample is not None:
                sample = SpaceMouseSample(
                    tx=float(sample.tx),
                    ty=float(sample.ty),
                    tz=float(sample.tz),
                    rx=float(sample.rx),
                    ry=float(sample.ry),
                    rz=float(sample.rz),
                    device_time_s=sample.device_time_s,
                    device_age_ms=sample.device_age_ms,
                    reports_drained=int(sample.reports_drained),
                    poll_interval_ms=poll_interval_ms,
                )
                with self._lock:
                    self._latest_sample = sample
                    self._error_text = None
                if self.sample_cb is not None:
                    try:
                        self.sample_cb(sample)
                    except Exception:
                        pass
            if self.poll_interval_s > 0.0:
                self._stop_event.wait(self.poll_interval_s)


class NullSpaceMouseBackend(SpaceMouseBackend):
    def __init__(self, reason: str):
        self.reason = str(reason)

    def is_available(self) -> bool:
        return False

    def status_text(self) -> str:
        return self.reason

    def open(self) -> None:
        raise RuntimeError(self.reason)

    def close(self) -> None:
        return

    def read_sample(self) -> SpaceMouseSample | None:
        return None


class PySpaceMouseBackend(SpaceMouseBackend):
    def __init__(self, module):
        self._module = module
        self._device_ctx = None
        self._device = None
        self._last_device_time_s: float | None = None

    def is_available(self) -> bool:
        return True

    def status_text(self) -> str:
        if self._device is None:
            return "SpaceMouse backend ready."
        return "SpaceMouse backend connected."

    def open(self) -> None:
        if self._device is not None:
            return
        ctx = self._module.open()
        if ctx is None:
            raise RuntimeError("pyspacemouse.open() returned no device")
        self._device_ctx = ctx
        if hasattr(ctx, "__enter__"):
            self._device = ctx.__enter__()
        else:
            self._device = ctx
        if self._device is None:
            raise RuntimeError("SpaceMouse device could not be opened")

    def close(self) -> None:
        if self._device_ctx is not None and hasattr(self._device_ctx, "__exit__"):
            try:
                self._device_ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._device_ctx = None
        self._device = None
        self._last_device_time_s = None

    def read_sample(self) -> SpaceMouseSample | None:
        if self._device is None:
            return None
        latest_state = None
        latest_time_s = self._last_device_time_s
        reports_drained = 0
        for _ in range(64):
            state = self._device.read()
            if state is None:
                break
            latest_state = state
            state_time_s = getattr(state, "t", None)
            try:
                state_time_s = float(state_time_s)
            except Exception:
                state_time_s = None
            if state_time_s is None or not math.isfinite(state_time_s):
                latest_time_s = None
                break
            if self._last_device_time_s is None or state_time_s > float(self._last_device_time_s) + 1e-9:
                reports_drained += 1
                latest_time_s = state_time_s
                self._last_device_time_s = state_time_s
                continue
            break

        if latest_state is None:
            return None
        device_age_ms = None
        if latest_time_s is not None:
            try:
                device_age_ms = max(0.0, 1000.0 * (timeit.default_timer() - float(latest_time_s)))
            except Exception:
                device_age_ms = None
        return SpaceMouseSample(
            tx=float(getattr(latest_state, "x", 0.0)),
            ty=float(getattr(latest_state, "y", 0.0)),
            tz=float(getattr(latest_state, "z", 0.0)),
            rx=float(getattr(latest_state, "roll", 0.0)),
            ry=float(getattr(latest_state, "pitch", 0.0)),
            rz=float(getattr(latest_state, "yaw", 0.0)),
            device_time_s=latest_time_s,
            device_age_ms=device_age_ms,
            reports_drained=reports_drained,
        )


def create_spacemouse_backend() -> SpaceMouseBackend:
    try:
        import pyspacemouse
    except ImportError:
        return NullSpaceMouseBackend("SpaceMouse backend unavailable: install with pip install -e .[spacemouse].")
    except Exception as exc:
        return NullSpaceMouseBackend(f"SpaceMouse backend unavailable: {exc}")
    return PySpaceMouseBackend(pyspacemouse)
