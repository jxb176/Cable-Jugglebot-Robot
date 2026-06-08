"""History storage for controller GUI telemetry."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from .channels import ChannelDefinition
from .models import TelemetryFrame


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    version: int
    times: list[float]
    series_by_key: Mapping[str, list[float]]


class TelemetryHistory:
    def __init__(self, channels: Mapping[str, ChannelDefinition], history_seconds: float = 60.0):
        self._channels = dict(channels)
        self.history_seconds = max(1.0, float(history_seconds))
        self._times = deque()
        self._frames = deque()
        self._series = {key: deque() for key in self._channels}
        self._time_origin_s: float | None = None
        self._version = 0
        self._times_cache: list[float] = []
        self._times_cache_version = 0
        self._series_cache = {key: [] for key in self._channels}
        self._series_cache_version = {key: 0 for key in self._channels}
        self._snapshot_cache: dict[tuple[str, ...], HistorySnapshot] = {}

    def clear(self) -> None:
        self._times.clear()
        self._frames.clear()
        for values in self._series.values():
            values.clear()
        self._time_origin_s = None
        self._mark_changed()

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
        self._mark_changed()

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
        if self._times_cache_version != self._version:
            self._times_cache = list(self._times)
            self._times_cache_version = self._version
        return self._times_cache

    def values(self, key: str) -> list[float]:
        if self._series_cache_version[key] != self._version:
            self._series_cache[key] = list(self._series[key])
            self._series_cache_version[key] = self._version
        return self._series_cache[key]

    def snapshot(self, keys: Iterable[str] | Mapping[str, object]) -> HistorySnapshot:
        if isinstance(keys, Mapping):
            cache_key = tuple(keys.keys())
        else:
            cache_key = tuple(keys)
        cached = self._snapshot_cache.get(cache_key)
        if cached is not None and cached.version == self._version:
            return cached

        snapshot = HistorySnapshot(
            version=self._version,
            times=self.times(),
            series_by_key={key: self.values(key) for key in cache_key},
        )
        self._snapshot_cache[cache_key] = snapshot
        return snapshot

    @property
    def version(self) -> int:
        return self._version

    @property
    def latest_frame(self) -> TelemetryFrame | None:
        if not self._frames:
            return None
        return self._frames[-1]

    def __len__(self) -> int:
        return len(self._times)

    def _mark_changed(self) -> None:
        self._version += 1
        self._times_cache_version = -1
        self._snapshot_cache.clear()
