"""Runtime lifecycle state machine and transition planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from jugglebot.core.types import ActuatorCommand, ActuatorControlMode


class RuntimeMode(str, Enum):
    ENABLE = "enable"
    DISABLE = "disable"
    ESTOP = "estop"
    PRETENSION = "pretension"

    @classmethod
    def from_state_value(cls, value: str) -> "RuntimeMode":
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ValueError(f"Unknown runtime mode: {value}") from exc


@dataclass(slots=True, frozen=True)
class RuntimeTransitionAction:
    mode: RuntimeMode
    commands: tuple[ActuatorCommand, ...]
    reset_runtime_state: bool = False


@dataclass(slots=True, frozen=True)
class RuntimeStateMachineResult:
    mode: RuntimeMode
    transition: RuntimeTransitionAction | None = None
    apply_home: bool = False
    apply_pretension_mode: bool = False
    apply_spool_gain_update: bool = False

    @property
    def allow_taskspace_streaming(self) -> bool:
        return self.mode is RuntimeMode.ENABLE

    @property
    def allow_pretension_streaming(self) -> bool:
        return self.mode is RuntimeMode.PRETENSION


@dataclass(slots=True)
class RuntimeStateMachine:
    axis_ids: tuple[int, ...]
    applied_state_version: int
    applied_home_version: int
    applied_pretension_version: int
    applied_spool_gain_version: int

    @classmethod
    def from_mailbox(cls, state, axis_ids) -> "RuntimeStateMachine":
        return cls(
            axis_ids=tuple(int(aid) for aid in axis_ids),
            applied_state_version=int(state.get_state_version()),
            applied_home_version=int(state.get_home_version()),
            applied_pretension_version=int(state.get_pretension_version()),
            applied_spool_gain_version=-1,
        )

    def step(self, state) -> RuntimeStateMachineResult:
        mode = RuntimeMode.from_state_value(state.get_state())
        transition = self._consume_state_transition(state, mode)
        apply_home = self._consume_home_request(state)
        apply_pretension_mode = self._consume_pretension_request(state)
        apply_spool_gain_update = self._consume_spool_gain_request(state)
        return RuntimeStateMachineResult(
            mode=mode,
            transition=transition,
            apply_home=apply_home,
            apply_pretension_mode=apply_pretension_mode,
            apply_spool_gain_update=apply_spool_gain_update,
        )

    def _consume_state_transition(self, state, mode: RuntimeMode) -> RuntimeTransitionAction | None:
        state_version = int(state.get_state_version())
        if state_version == self.applied_state_version:
            return None
        self.applied_state_version = state_version
        return RuntimeTransitionAction(
            mode=mode,
            commands=self.build_mode_commands(mode),
            reset_runtime_state=mode in (RuntimeMode.DISABLE, RuntimeMode.ESTOP),
        )

    def _consume_home_request(self, state) -> bool:
        home_version = int(state.get_home_version())
        if home_version == self.applied_home_version:
            return False
        self.applied_home_version = home_version
        return True

    def _consume_pretension_request(self, state) -> bool:
        pretension_version = int(state.get_pretension_version())
        if pretension_version == self.applied_pretension_version:
            return False
        self.applied_pretension_version = pretension_version
        return True

    def _consume_spool_gain_request(self, state) -> bool:
        spool_gain_version = int(state.get_spool_gain_version())
        if spool_gain_version == self.applied_spool_gain_version:
            return False
        self.applied_spool_gain_version = spool_gain_version
        return True

    def build_mode_commands(self, mode: RuntimeMode) -> tuple[ActuatorCommand, ...]:
        commands: list[ActuatorCommand] = []
        if mode is RuntimeMode.ENABLE:
            for axis_id in self.axis_ids:
                commands.append(
                    ActuatorCommand(
                        axis_id=axis_id,
                        control_mode=ActuatorControlMode.POSITION,
                        apply_control_mode=True,
                        enable=True,
                    )
                )
        elif mode is RuntimeMode.PRETENSION:
            for axis_id in self.axis_ids:
                commands.append(
                    ActuatorCommand(
                        axis_id=axis_id,
                        control_mode=ActuatorControlMode.TORQUE,
                        apply_control_mode=True,
                        enable=True,
                    )
                )
        elif mode in (RuntimeMode.DISABLE, RuntimeMode.ESTOP):
            for axis_id in self.axis_ids:
                commands.append(
                    ActuatorCommand(
                        axis_id=axis_id,
                        control_mode=ActuatorControlMode.DISABLED,
                        enable=False,
                    )
                )
        return tuple(commands)
