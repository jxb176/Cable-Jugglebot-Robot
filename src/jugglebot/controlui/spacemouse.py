"""Optional SpaceMouse backend integration for the control UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpaceMouseSample:
    tx: float
    ty: float
    tz: float
    rx: float
    ry: float
    rz: float


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

    def read_sample(self) -> SpaceMouseSample | None:
        if self._device is None:
            return None
        state = self._device.read()
        if state is None:
            return None
        return SpaceMouseSample(
            tx=float(getattr(state, "x", 0.0)),
            ty=float(getattr(state, "y", 0.0)),
            tz=float(getattr(state, "z", 0.0)),
            rx=float(getattr(state, "roll", 0.0)),
            ry=float(getattr(state, "pitch", 0.0)),
            rz=float(getattr(state, "yaw", 0.0)),
        )


def create_spacemouse_backend() -> SpaceMouseBackend:
    try:
        import pyspacemouse
    except ImportError:
        return NullSpaceMouseBackend("SpaceMouse backend unavailable: install with pip install -e .[spacemouse].")
    except Exception as exc:
        return NullSpaceMouseBackend(f"SpaceMouse backend unavailable: {exc}")
    return PySpaceMouseBackend(pyspacemouse)
