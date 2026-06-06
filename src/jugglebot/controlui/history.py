"""History storage for controller GUI telemetry."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math

from .channels import ChannelDefinition
from .models import TelemetryFrame


class TelemetryHistory:
    def __init__(self, channels: Mapping[str, ChannelDefinition], history_seconds: float = 20.0):
        self._channels = dict(channels)
        self.history_seconds = max(1.0, float(history_seconds))
        self._times = deque()
        self._frames = deque()
        self._series = {key: deque() for key in self._channels}
        self._time_origin_s: float | None = None

    def clear(self) -> None:
        self._times.clear()
        self._frames.clear()
        for values in self._series.values():
            values.clear()
        self._time_origin_s = None

    def append(self, frame: TelemetryFrame) -> None:
        absolute_time_s = float(frame.preferred_time_s())
        if not math.isfinite(absolute_time_s):
            absolute_time_s = float(frame.receipt_time_s)
        if self._time_origin_s is None:
            self._time_origin_s = absolute_time_s
        relative_time_s = absolute_time_s - self._time_origin_s
        self._times.append(relative_time_s)
        self._frames.append(frame)
        for key, channel in self._channels.items():
            self._series[key].append(float(channel.extractor(frame)))
        self._trim()

    def _trim(self) -> None:
        if not self._times:
            return
        latest_time_s = self._times[-1]
        while self._times and (latest_time_s - self._times[0]) > self.history_seconds:
            self._times.popleft()
            self._frames.popleft()
            for values in self._series.values():
                values.popleft()

    def times(self) -> list[float]:
        return list(self._times)

    def values(self, key: str) -> list[float]:
        return list(self._series[key])

    @property
    def latest_frame(self) -> TelemetryFrame | None:
        if not self._frames:
            return None
        return self._frames[-1]

    def __len__(self) -> int:
        return len(self._times)
