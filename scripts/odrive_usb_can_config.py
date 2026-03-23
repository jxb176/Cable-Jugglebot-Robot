#!/usr/bin/env python3
"""Configure ODrive CAN settings over USB with serial-number verification.

Example:
  python scripts/odrive_usb_can_config.py \
      --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Any


GLOBAL_AXIS_CAN_SETTINGS = {
    "bus_voltage_msg_rate_ms": 1000,
    "encoder_msg_rate_ms": 4,
    "error_msg_rate_ms": 250,
    "heartbeat_msg_rate_ms": 250,
    "iq_msg_rate_ms": 100,
    "powers_msg_rate_ms": 0,
    "temperature_msg_rate_ms": 1000,
    "torques_msg_rate_ms": 4,
    "version_msg_rate_ms": 0,
}

GLOBAL_AXIS_CONTROLLER_SETTINGS = {
    "input_filter_bandwidth": 150.0,
    "input_mode": 3,
    "pos_gain": 50.0,
    "torque_ramp_rate": 0.01,
    "vel_gain": 0.01,
    "vel_integrator_gain": 0.0,
    "vel_integrator_limit": float("inf"),
    "vel_limit": 60.0,
    "vel_limit_tolerance": 1.667,
    "vel_ramp_rate": 10.0,
}

HARDCODED_ODRIVES = [
    {
        "name": "odrive_0",
        "serial_number": "364D33643432",
        "axis": "axis0",
        "axis_can_overrides": {
            "node_id": 0,
        },
    },
    {
        "name": "odrive_1",
        "serial_number": "3676334D3432",
        "axis": "axis0",
        "axis_can_overrides": {
            "node_id": 1,
        },
    },
    {
        "name": "odrive_2",
        "serial_number": "367633723432",
        "axis": "axis0",
        "axis_can_overrides": {
            "node_id": 2,
        },
    },
    {
        "name": "odrive_3",
        "serial_number": "367B33653432",
        "axis": "axis0",
        "axis_can_overrides": {
            "node_id": 3,
        },
    },
    {
        "name": "odrive_4",
        "serial_number": "3667336A3432",
        "axis": "axis0",
        "axis_can_overrides": {
            "node_id": 4,
        },
    },
    {
        "name": "odrive_5",
        "serial_number": "367733663432",
        "axis": "axis0",
        "axis_can_overrides": {
            "node_id": 5,
        },
    },
]


@dataclass(frozen=True)
class PendingChange:
    path: str
    value: Any


def _normalize_serial(serial: str) -> str:
    value = str(serial).strip().lower()
    if value.startswith("0x"):
        value = value[2:]
    return value


def _serial_int_variants(serial: str) -> set[int]:
    normalized = _normalize_serial(serial)
    values: set[int] = set()
    try:
        values.add(int(normalized, 10))
    except ValueError:
        pass
    try:
        values.add(int(normalized, 16))
    except ValueError:
        pass
    return values


def _serial_variants(serial: str) -> list[Any]:
    normalized = _normalize_serial(serial)
    variants: list[Any] = [serial, normalized]
    ints = sorted(_serial_int_variants(serial))
    for value in ints:
        variants.append(value)
        variants.append(str(value))
        variants.append(format(value, "x"))
        variants.append("0x" + format(value, "x"))

    deduped: list[Any] = []
    seen: set[tuple[type, str]] = set()
    for item in variants:
        key = (type(item), str(item))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _config_serial_int(entry: dict[str, Any]) -> int:
    serial = str(entry["serial_number"])
    try:
        return int(_normalize_serial(serial), 16)
    except ValueError as exc:
        raise ValueError(
            f"Invalid hardcoded serial number for {entry.get('name', '<unnamed>')}: {serial}"
        ) from exc


def _device_serial_text(device: Any) -> str | None:
    serial = getattr(device, "serial_number", None)
    if serial is None:
        return None
    return str(serial).strip()


def _device_serial_int(device: Any) -> int | None:
    serial = getattr(device, "serial_number", None)
    if serial is None:
        return None
    try:
        return int(serial)
    except Exception:
        pass
    variants = _serial_int_variants(str(serial))
    if len(variants) == 1:
        return next(iter(variants))
    return None


def _serial_matches(device: Any, expected_serial: str) -> bool:
    actual_int = _device_serial_int(device)
    if actual_int is not None:
        return actual_int in _serial_int_variants(expected_serial)

    actual = getattr(device, "serial_number", None)
    if actual is None:
        return False
    actual_text = _normalize_serial(str(actual))
    expected_texts = {str(v) for v in _serial_int_variants(expected_serial)}
    expected_texts.add(_normalize_serial(expected_serial))
    return actual_text in expected_texts


def _connect_odrive(expected_serial: str | None, timeout_s: float):
    try:
        import odrive
    except ImportError as exc:
        raise RuntimeError(
            "The 'odrive' Python package is required. Install it in the environment used for this script."
        ) from exc

    device = None
    last_error = None
    if expected_serial is None:
        try:
            device = odrive.find_any(timeout=timeout_s)
        except Exception as exc:
            last_error = exc
            device = None
    else:
        deadline = time.monotonic() + timeout_s
        candidates = _serial_variants(expected_serial)
        for idx, candidate in enumerate(candidates, start=1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            print(
                f"  Looking for serial {expected_serial} "
                f"(candidate {idx}/{len(candidates)}: {candidate}, timeout {remaining:.1f}s)"
            )
            try:
                device = odrive.find_any(serial_number=candidate, timeout=remaining)
            except Exception as exc:
                last_error = exc
                device = None
            if device is not None:
                break

    if device is None:
        raise RuntimeError(
            f"Unable to find ODrive"
            + (f" with serial number '{expected_serial}'" if expected_serial is not None else "")
            + (f": {last_error}" if last_error else "")
        )

    actual_serial = _device_serial_text(device)
    if actual_serial is None:
        _disconnect_odrive(device)
        raise RuntimeError("Connected device does not expose a serial_number property")

    if expected_serial is not None and not _serial_matches(device, expected_serial):
        _disconnect_odrive(device)
        raise RuntimeError(
            f"Connected wrong ODrive: expected serial {expected_serial}, got {actual_serial}"
        )

    return device


def _disconnect_odrive(device: Any) -> None:
    destroy = getattr(device, "_destroy", None)
    if callable(destroy):
        try:
            destroy()
        except Exception:
            pass


def _resolve_attr(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _read_value(root: Any, path: str) -> Any:
    return _resolve_attr(root, path)


def _write_value(root: Any, path: str, value: Any) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = _resolve_attr(root, parent_path) if parent_path else root
    setattr(parent, leaf, value)


def _find_device_config(serial_text: str) -> dict[str, Any] | None:
    serial_ints = _serial_int_variants(serial_text)
    for entry in HARDCODED_ODRIVES:
        if _config_serial_int(entry) in serial_ints:
            return entry
    return None


def _resolve_axis_name(args: argparse.Namespace, device_cfg: dict[str, Any]) -> str:
    axis = args.axis or device_cfg.get("axis", "axis0")
    if axis not in ("axis0", "axis1"):
        raise ValueError("axis must be axis0 or axis1")
    return axis


def _build_changes(args: argparse.Namespace, device_cfg: dict[str, Any]) -> list[PendingChange]:
    axis = _resolve_axis_name(args, device_cfg)

    can_values: dict[str, int] = {}
    controller_values: dict[str, Any] = {}
    if args.apply_default_can_rates or not args.no_globals:
        can_values.update(GLOBAL_AXIS_CAN_SETTINGS)
        controller_values.update(GLOBAL_AXIS_CONTROLLER_SETTINGS)
    can_values.update({k: int(v) for k, v in (device_cfg.get("axis_can_overrides") or {}).items()})
    controller_values.update(device_cfg.get("axis_controller_overrides") or {})

    can_overrides = {
        "bus_voltage_msg_rate_ms": args.bus_voltage_msg_rate_ms,
        "encoder_msg_rate_ms": args.encoder_msg_rate_ms,
        "error_msg_rate_ms": args.error_msg_rate_ms,
        "heartbeat_msg_rate_ms": args.heartbeat_msg_rate_ms,
        "iq_msg_rate_ms": args.iq_msg_rate_ms,
        "node_id": args.node_id,
        "powers_msg_rate_ms": args.powers_msg_rate_ms,
        "temperature_msg_rate_ms": args.temperature_msg_rate_ms,
        "torques_msg_rate_ms": args.torques_msg_rate_ms,
        "version_msg_rate_ms": args.version_msg_rate_ms,
    }
    for key, value in can_overrides.items():
        if value is not None:
            can_values[key] = int(value)

    controller_overrides = {
        "input_filter_bandwidth": args.input_filter_bandwidth,
        "input_mode": args.input_mode,
        "pos_gain": args.pos_gain,
        "torque_ramp_rate": args.torque_ramp_rate,
        "vel_gain": args.vel_gain,
        "vel_integrator_gain": args.vel_integrator_gain,
        "vel_integrator_limit": args.vel_integrator_limit,
        "vel_limit": args.vel_limit,
        "vel_limit_tolerance": args.vel_limit_tolerance,
        "vel_ramp_rate": args.vel_ramp_rate,
    }
    for key, value in controller_overrides.items():
        if value is not None:
            controller_values[key] = value

    changes = [
        PendingChange(path=f"{axis}.config.can.{name}", value=int(value))
        for name, value in sorted(can_values.items())
    ]
    changes.extend(
        PendingChange(path=f"{axis}.controller.config.{name}", value=value)
        for name, value in sorted(controller_values.items())
    )
    return changes


def _print_change_table(device: Any, changes: list[PendingChange]) -> None:
    print("Requested changes:")
    for change in changes:
        current = _read_value(device, change.path)
        print(f"  {change.path}: current={current} -> target={change.value}")


def _apply_and_verify(device: Any, changes: list[PendingChange], dry_run: bool) -> None:
    if dry_run:
        return

    for change in changes:
        _write_value(device, change.path, change.value)

    for change in changes:
        actual = _read_value(device, change.path)
        if not _values_match(actual, change.value):
            raise RuntimeError(
                f"Verification failed for {change.path}: expected {change.value}, got {actual}"
            )


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except Exception:
            return False
    try:
        actual_f = float(actual)
        expected_f = float(expected)
    except Exception:
        return actual == expected
    if math.isnan(expected_f):
        return math.isnan(actual_f)
    if math.isinf(expected_f):
        return math.isinf(actual_f) and ((actual_f > 0) == (expected_f > 0))
    return math.isclose(actual_f, expected_f, rel_tol=1e-6, abs_tol=1e-9)


def _save_configuration(device: Any) -> None:
    save = getattr(device, "save_configuration", None)
    if not callable(save):
        raise RuntimeError("Connected ODrive object does not provide save_configuration()")
    save()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial-number",
        default=None,
        help="Optional expected ODrive serial number. If set, it must also appear in HARDCODED_ODRIVES.",
    )
    parser.add_argument("--axis", default=None, help="Override axis selection: axis0 or axis1")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1.0,
        help="USB discovery timeout in seconds",
    )
    parser.add_argument(
        "--apply-default-can-rates",
        action="store_true",
        help="Apply the global CAN telemetry settings explicitly",
    )
    parser.add_argument(
        "--no-globals",
        action="store_true",
        help="Skip global CAN settings and apply only per-device settings plus explicit CLI overrides",
    )
    parser.add_argument("--list-devices", action="store_true", help="Print the hardcoded device table and exit")
    parser.add_argument("--bus-voltage-msg-rate-ms", type=int, default=None)
    parser.add_argument("--encoder-msg-rate-ms", type=int, default=None)
    parser.add_argument("--error-msg-rate-ms", type=int, default=None)
    parser.add_argument("--heartbeat-msg-rate-ms", type=int, default=None)
    parser.add_argument("--iq-msg-rate-ms", type=int, default=None)
    parser.add_argument("--node-id", type=int, default=None)
    parser.add_argument("--powers-msg-rate-ms", type=int, default=None)
    parser.add_argument("--temperature-msg-rate-ms", type=int, default=None)
    parser.add_argument("--torques-msg-rate-ms", type=int, default=None)
    parser.add_argument("--version-msg-rate-ms", type=int, default=None)
    parser.add_argument("--input-filter-bandwidth", type=float, default=None)
    parser.add_argument("--input-mode", type=int, default=None)
    parser.add_argument("--pos-gain", type=float, default=None)
    parser.add_argument("--torque-ramp-rate", type=float, default=None)
    parser.add_argument("--vel-gain", type=float, default=None)
    parser.add_argument("--vel-integrator-gain", type=float, default=None)
    parser.add_argument("--vel-integrator-limit", type=float, default=None)
    parser.add_argument("--vel-limit", type=float, default=None)
    parser.add_argument("--vel-limit-tolerance", type=float, default=None)
    parser.add_argument("--vel-ramp-rate", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing them")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not call save_configuration() after applying verified changes",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first device that fails instead of continuing through the device list",
    )
    return parser.parse_args(argv)


def _print_device_table() -> None:
    for entry in HARDCODED_ODRIVES:
        serial_hex = _normalize_serial(str(entry["serial_number"])).upper()
        serial_dec = _config_serial_int(entry)
        print(
            f"{entry['name']}: serial=0x{serial_hex} ({serial_dec}), "
            f"axis={entry.get('axis', 'axis0')}, overrides={entry.get('axis_can_overrides', {})}"
        )


def _selected_device_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.serial_number is None:
        return list(HARDCODED_ODRIVES)

    requested_cfg = _find_device_config(args.serial_number)
    if requested_cfg is None:
        raise RuntimeError(
            f"serial number {args.serial_number} is not present in HARDCODED_ODRIVES"
        )
    return [requested_cfg]


def _configure_one_device(args: argparse.Namespace, device_cfg: dict[str, Any]) -> None:
    device = None
    try:
        expected_serial = str(device_cfg["serial_number"])
        device = _connect_odrive(expected_serial, args.timeout_s)
        serial_text = _device_serial_text(device)
        serial_int = _device_serial_int(device)
        if serial_int is not None:
            print(
                f"[{device_cfg['name']}] Connected to ODrive serial {serial_text} "
                f"(0x{serial_int:x})"
            )
        else:
            print(f"[{device_cfg['name']}] Connected to ODrive serial {serial_text}")

        matched_cfg = _find_device_config(serial_text or "")
        if matched_cfg is None:
            raise RuntimeError(
                f"Connected ODrive serial {serial_text} is not present in HARDCODED_ODRIVES"
            )
        if matched_cfg is not device_cfg:
            raise RuntimeError(
                f"Connected config mismatch: expected {device_cfg['name']}, matched {matched_cfg['name']}"
            )

        changes = _build_changes(args, device_cfg)
        if not changes:
            raise RuntimeError("No parameter changes requested. Check global/per-device config and CLI overrides.")

        _print_change_table(device, changes)
        _apply_and_verify(device, changes, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[{device_cfg['name']}] Dry run complete. No settings were written.")
            return

        print(f"[{device_cfg['name']}] Verified updated parameters in RAM.")
        if not args.no_save:
            print(f"[{device_cfg['name']}] Saving configuration...")
            try:
                _save_configuration(device)
                print(f"[{device_cfg['name']}] Configuration saved.")
            except Exception as exc:
                # ODrive may drop the USB connection or reboot during save.
                print(f"[{device_cfg['name']}] save_configuration() returned with exception: {exc}")
                print(f"[{device_cfg['name']}] If the device rebooted, reconnect and verify the saved values.")
        else:
            print(f"[{device_cfg['name']}] Skipped save_configuration() due to --no-save.")
    finally:
        if device is not None:
            _disconnect_odrive(device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_devices:
        _print_device_table()
        return 0

    try:
        selected = _selected_device_configs(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    failures: list[tuple[str, str]] = []
    for device_cfg in selected:
        print(f"[{device_cfg['name']}] Starting configuration")
        try:
            _configure_one_device(args, device_cfg)
        except Exception as exc:
            failures.append((device_cfg["name"], str(exc)))
            print(f"[{device_cfg['name']}] ERROR: {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    successes = len(selected) - len(failures)
    print(f"Summary: {successes} succeeded, {len(failures)} failed.")
    for name, message in failures:
        print(f"  {name}: {message}", file=sys.stderr)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
