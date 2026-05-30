"""Clock abstractions for the real-time runtime layer."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


class RuntimeClock(ABC):
    """Abstract time source used by the runtime loop."""

    @abstractmethod
    def now_monotonic(self) -> float:
        """Return a monotonic timestamp in seconds."""

    @abstractmethod
    def now_wall_time(self) -> float:
        """Return a wall-clock timestamp in seconds."""

    @abstractmethod
    def sleep_until(self, monotonic_deadline_s: float):
        """Sleep until the given monotonic deadline."""


class WallClock(RuntimeClock):
    """Wall-clock backed runtime clock."""

    def now_monotonic(self) -> float:
        return time.perf_counter()

    def now_wall_time(self) -> float:
        return time.time()

    def sleep_until(self, monotonic_deadline_s: float):
        sleep_s = float(monotonic_deadline_s) - time.perf_counter()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
