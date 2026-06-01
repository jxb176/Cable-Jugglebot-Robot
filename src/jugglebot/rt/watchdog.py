"""Passive runtime watchdog for loop timing and feedback health."""

from __future__ import annotations

from dataclasses import dataclass

from jugglebot.core.types import RuntimeHealthLevel, WatchdogStatus


@dataclass(slots=True, frozen=True)
class WatchdogSample:
    now_perf_s: float
    mode: str
    loop_period_s: float | None
    total_loop_duration_s: float
    deadline_margin_s: float | None
    missed_deadline_count: int
    feedback_age_s: float | None


@dataclass(slots=True, frozen=True)
class WatchdogEvaluation:
    status: WatchdogStatus
    report_due: bool
    log_as_warning: bool
    missed_since_last_report: int


@dataclass(slots=True, frozen=True)
class _ModePolicy:
    deadline_margin_warning_s: float
    missed_per_report_warning: int
    feedback_age_warning_s: float


class RuntimeWatchdog:
    """Classify runtime timing health and decide when to emit status reports."""

    def __init__(
        self,
        *,
        status_report_period_s: float,
        deadline_warning_margin_s: float,
        deadline_warning_missed_per_report: int,
        feedback_age_warning_s: float,
        transition_grace_s: float,
    ):
        self._status_report_period_s = max(0.1, float(status_report_period_s))
        self._deadline_warning_margin_s = max(0.0, float(deadline_warning_margin_s))
        self._deadline_warning_missed_per_report = max(1, int(deadline_warning_missed_per_report))
        self._feedback_age_warning_s = max(0.0, float(feedback_age_warning_s))
        self._transition_grace_s = max(0.0, float(transition_grace_s))
        self._last_report_perf_s: float | None = None
        self._last_report_missed_deadline_count = 0
        self._last_mode: str | None = None
        self._last_transition_perf_s: float | None = None
        self._consecutive_missed_deadlines = 0

    def observe(self, sample: WatchdogSample) -> WatchdogEvaluation:
        mode = str(sample.mode).lower()
        if self._last_mode != mode:
            self._last_mode = mode
            self._last_transition_perf_s = float(sample.now_perf_s)

        if sample.deadline_margin_s is not None and float(sample.deadline_margin_s) < 0.0:
            self._consecutive_missed_deadlines += 1
        else:
            self._consecutive_missed_deadlines = 0

        policy = self._policy_for_mode(mode, sample.loop_period_s)
        transition_grace_active = self._in_transition_grace(sample.now_perf_s)
        missed_since_last_report = max(
            0,
            int(sample.missed_deadline_count) - int(self._last_report_missed_deadline_count),
        )
        low_deadline_margin = (
            sample.deadline_margin_s is not None
            and float(sample.deadline_margin_s) <= policy.deadline_margin_warning_s
        )
        excessive_missed_deadlines = missed_since_last_report >= policy.missed_per_report_warning
        stale_feedback = (
            sample.feedback_age_s is not None
            and float(sample.feedback_age_s) >= policy.feedback_age_warning_s
        )

        if transition_grace_active:
            low_deadline_margin = False
            excessive_missed_deadlines = False

        level = RuntimeHealthLevel.HEALTHY
        reason_parts: list[str] = []
        if stale_feedback:
            level = RuntimeHealthLevel.VIOLATION
            reason_parts.append("stale feedback")
        if excessive_missed_deadlines:
            if level is RuntimeHealthLevel.HEALTHY:
                level = RuntimeHealthLevel.WARNING
            reason_parts.append("deadline misses above threshold")
        if low_deadline_margin:
            if level is RuntimeHealthLevel.HEALTHY:
                level = RuntimeHealthLevel.WARNING
            reason_parts.append("low deadline margin")
        if transition_grace_active:
            reason_parts.append("transition grace")

        status = WatchdogStatus(
            level=level,
            mode=mode,
            message=", ".join(reason_parts) if reason_parts else None,
            transition_grace_active=transition_grace_active,
            deadline_margin_s=None if sample.deadline_margin_s is None else float(sample.deadline_margin_s),
            feedback_age_s=None if sample.feedback_age_s is None else float(sample.feedback_age_s),
            missed_deadline_count=int(sample.missed_deadline_count),
            missed_deadline_delta=int(missed_since_last_report),
            consecutive_missed_deadlines=int(self._consecutive_missed_deadlines),
            low_deadline_margin=bool(low_deadline_margin),
            excessive_missed_deadlines=bool(excessive_missed_deadlines),
            stale_feedback=bool(stale_feedback),
        )

        report_due = (
            self._last_report_perf_s is None
            or (float(sample.now_perf_s) - float(self._last_report_perf_s)) >= self._status_report_period_s
        )
        if report_due:
            self._last_report_perf_s = float(sample.now_perf_s)
            self._last_report_missed_deadline_count = int(sample.missed_deadline_count)

        log_as_warning = level is not RuntimeHealthLevel.HEALTHY
        return WatchdogEvaluation(
            status=status,
            report_due=bool(report_due),
            log_as_warning=bool(log_as_warning),
            missed_since_last_report=int(missed_since_last_report),
        )

    def _in_transition_grace(self, now_perf_s: float) -> bool:
        if self._last_transition_perf_s is None:
            return False
        return (float(now_perf_s) - float(self._last_transition_perf_s)) < self._transition_grace_s

    def _policy_for_mode(self, mode: str, loop_period_s: float | None) -> _ModePolicy:
        base_feedback_age_warning_s = self._feedback_age_warning_s
        if loop_period_s is not None:
            base_feedback_age_warning_s = max(base_feedback_age_warning_s, 2.0 * float(loop_period_s))

        if mode == "enable":
            return _ModePolicy(
                deadline_margin_warning_s=self._deadline_warning_margin_s,
                missed_per_report_warning=self._deadline_warning_missed_per_report,
                feedback_age_warning_s=base_feedback_age_warning_s,
            )
        if mode == "pretension":
            return _ModePolicy(
                deadline_margin_warning_s=1.5 * self._deadline_warning_margin_s,
                missed_per_report_warning=max(1, 2 * self._deadline_warning_missed_per_report),
                feedback_age_warning_s=1.5 * base_feedback_age_warning_s,
            )
        return _ModePolicy(
            deadline_margin_warning_s=4.0 * self._deadline_warning_margin_s,
            missed_per_report_warning=max(1000, 10 * self._deadline_warning_missed_per_report),
            feedback_age_warning_s=2.0 * base_feedback_age_warning_s,
        )
