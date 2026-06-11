from __future__ import annotations

from dataclasses import dataclass
import math

from jugglebot.controlui.spacemouse import PySpaceMouseBackend


@dataclass
class _FakeState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    t: float = 0.0


class _FakeDevice:
    def __init__(self, responses):
        self._responses = list(responses)
        self._last = self._responses[-1] if self._responses else None

    def read(self):
        if self._responses:
            self._last = self._responses.pop(0)
        return self._last


def test_read_sample_drains_pending_reports_and_returns_latest(monkeypatch) -> None:
    monkeypatch.setattr("jugglebot.controlui.spacemouse.timeit.default_timer", lambda: 10.0)
    backend = PySpaceMouseBackend(object())
    state1 = _FakeState(x=0.1, t=9.90)
    state2 = _FakeState(x=0.6, y=-0.2, t=9.95)
    backend._device = _FakeDevice([state1, state2, state2])

    sample = backend.read_sample()

    assert sample is not None
    assert sample.tx == 0.6
    assert sample.ty == -0.2
    assert sample.reports_drained == 2
    assert sample.device_time_s == 9.95
    assert math.isclose(sample.device_age_ms or 0.0, 50.0, rel_tol=0.0, abs_tol=1e-9)


def test_read_sample_reports_zero_drained_when_state_is_stale(monkeypatch) -> None:
    monkeypatch.setattr("jugglebot.controlui.spacemouse.timeit.default_timer", lambda: 20.0)
    backend = PySpaceMouseBackend(object())
    backend._last_device_time_s = 19.8
    stale = _FakeState(x=-0.4, t=19.8)
    backend._device = _FakeDevice([stale])

    sample = backend.read_sample()

    assert sample is not None
    assert sample.tx == -0.4
    assert sample.reports_drained == 0
    assert sample.device_time_s == 19.8
    assert math.isclose(sample.device_age_ms or 0.0, 200.0, rel_tol=0.0, abs_tol=1e-9)
