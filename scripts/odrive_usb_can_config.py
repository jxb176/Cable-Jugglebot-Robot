#!/usr/bin/env python3
"""Configure ODrive CAN settings over USB with serial-number verification.

Example:
  python scripts/odrive_usb_can_config.py \
      --dry-run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any


GLOBAL_AXIS_CAN_SETTINGS = {
    "bus_voltage_msg_rate_ms": 1000,
    "encoder_msg_rate_ms": 10,
    "error_msg_rate_ms": 1000,
    "heartbeat_msg_rate_ms": 0,
    "iq_msg_rate_ms": 100,
    "powers_msg_rate_ms": 0,
    "temperature_msg_rate_ms": 1000,
    "torques_msg_rate_ms": 0,
    "version_msg_rate_ms": 0,
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
    value: int


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
    variants.extend(sorted(_serial_int_variants(serial)))
    return variants


def _device_serial_text(device: Any) -> str | None:
    serial = getattr(device, "serial_number", None)
    if serial is None:
        return None
    return str(serial).strip()


def _serial_matches(device: Any, expected_serial: str) -> bool:
    actual = getattr(device, "serial_number", None)
    if actual is None:
        return False

    actual_text = _normalize_serial(str(actual))
    expected_texts = {_normalize_serial(expected_serial), str(expected_serial).strip().lower()}
    if actual_text in expected_texts:
        return True

    try:
        actual_int = int(actual)
    except Exception:
        return False
    return actual_int in _serial_int_variants(expected_serial)


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
            try:
                device = odrive.find_any(timeout=timeout_s)
            except TypeError:
                device = odrive.find_any()
        except Exception as exc:
            last_error = exc
            device = None
    else:
        for candidate in _serial_variants(expected_serial):
            try:
                try:
                    device = odrive.find_any(serial_number=candidate, timeout=timeout_s)
                except TypeError:
                    device = odrive.find_any(serial_number=candidate)
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
    for entry in HARDCODED_ODRIVES:
        if _normalize_serial(entry["serial_number"]) == _normalize_serial(serial_text):
            return entry
        if _normalize_serial(entry["serial_number"]).startswith("placeholder_serial_"):
            continue
        if _serial_int_variants(entry["serial_number"]) & _serial_int_variants(serial_text):
            return entry
    return None


def _resolve_axis_name(args: argparse.Namespace, device_cfg: dict[str, Any]) -> str:
    axis = args.axis or device_cfg.get("axis", "axis0")
    if axis not in ("axis0", "axis1"):
        raise ValueError("axis must be axis0 or axis1")
    return axis


def _build_changes(args: argparse.Namespace, device_cfg: dict[str, Any]) -> list[PendingChange]:
    axis = _resolve_axis_name(args, device_cfg)

    values: dict[str, int] = {}
    if args.apply_default_can_rates or not args.no_globals:
        values.update(GLOBAL_AXIS_CAN_SETTINGS)
    values.update({k: int(v) for k, v in (device_cfg.get("axis_can_overrides") or {}).items()})

    overrides = {
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
    for key, value in overrides.items():
        if value is not None:
            values[key] = int(value)

    return [
        PendingChange(path=f"{axis}.config.can.{name}", value=int(value))
        for name, value in sorted(values.items())
    ]


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
        if int(actual) != int(change.value):
            raise RuntimeError(
                f"Verification failed for {change.path}: expected {change.value}, got {actual}"
            )


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
        default=10.0,
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
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing them")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not call save_configuration() after applying verified changes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_devices:
        for entry in HARDCODED_ODRIVES:
            print(
                f"{entry['name']}: serial={entry['serial_number']}, "
                f"axis={entry.get('axis', 'axis0')}, overrides={entry.get('axis_can_overrides', {})}"
            )
        return 0

    if args.serial_number is not None:
        requested_cfg = _find_device_config(args.serial_number)
        if requested_cfg is None:
            print(
                f"ERROR: serial number {args.serial_number} is not present in HARDCODED_ODRIVES",
                file=sys.stderr,
            )
            return 1

    device = None
    try:
        device = _connect_odrive(args.serial_number, args.timeout_s)
        serial_text = _device_serial_text(device)
        print(f"Connected to ODrive serial {serial_text}")

        device_cfg = _find_device_config(serial_text or "")
        if device_cfg is None:
            raise RuntimeError(
                f"Connected ODrive serial {serial_text} is not present in HARDCODED_ODRIVES"
            )

        print(f"Matched device config: {device_cfg['name']}")
        changes = _build_changes(args, device_cfg)
        if not changes:
            print("No parameter changes requested. Check global/per-device config and CLI overrides.")
            return 2

        _print_change_table(device, changes)
        _apply_and_verify(device, changes, dry_run=args.dry_run)
        if args.dry_run:
            print("Dry run complete. No settings were written.")
            return 0

        print("Verified updated parameters in RAM.")
        if not args.no_save:
            print("Saving configuration...")
            try:
                _save_configuration(device)
                print("Configuration saved.")
            except Exception as exc:
                # ODrive may drop the USB connection or reboot during save.
                print(f"save_configuration() returned with exception: {exc}")
                print("If the device rebooted, reconnect and verify the saved values.")
        else:
            print("Skipped save_configuration() due to --no-save.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if device is not None:
            _disconnect_odrive(device)


if __name__ == "__main__":
    raise SystemExit(main())
