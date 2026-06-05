#!/usr/bin/env python3
"""Dump persistent ODrive configuration to flat JSON and tree YAML files.

This script is intended for firmware migration audits. It walks persistent
configuration roots by parameter name, not by opaque binary/json export, so
version-to-version diffs stay meaningful and missing parameters fail clearly.

Outputs for each selected drive:
  - <device_name>__config_flat.json
  - <device_name>__config_tree.yaml
  - <device_name>__full_flat.json
  - <device_name>__full_tree.yaml

By default dumps all devices listed in HARDCODED_ODRIVES from
scripts/odrive_usb_can_config.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from odrive_usb_can_config import (
    HARDCODED_ODRIVES,
    _config_serial_int,
    _connect_odrive,
    _device_serial_int,
    _device_serial_text,
    _disconnect_odrive,
    _find_device_config,
    _normalize_serial,
    _print_device_table,
)


@dataclass(frozen=True)
class DumpResult:
    device_name: str
    serial_number: str
    axis: str
    flat_json_path: Path
    tree_yaml_path: Path
    parameter_count: int
    full_flat_json_path: Path
    full_tree_yaml_path: Path
    full_parameter_count: int


_SKIP_NAMES = {
    "parent",
    "property_exchange",
}


def _safe_getattr(obj: Any, name: str):
    try:
        return getattr(obj, name)
    except Exception as exc:
        return exc


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str))


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _insert_tree(tree: dict[str, Any], dotted_path: str, value: Any) -> None:
    node = tree
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _iter_named_children(obj: Any):
    for name in sorted(dir(obj)):
        if name.startswith("_") or name in _SKIP_NAMES:
            continue
        value = _safe_getattr(obj, name)
        if isinstance(value, Exception):
            continue
        if callable(value):
            continue
        yield name, value


def _collect_config_roots(device: Any, axis_name: str) -> OrderedDict[str, Any]:
    roots: OrderedDict[str, Any] = OrderedDict()

    if hasattr(device, "config"):
        roots["odrv.config"] = device.config

    can_obj = _safe_getattr(device, "can")
    can_cfg = None if isinstance(can_obj, Exception) else _safe_getattr(can_obj, "config")
    if can_cfg is not None and not isinstance(can_cfg, Exception):
        roots["odrv.can.config"] = can_cfg

    axis_obj = _safe_getattr(device, axis_name)
    if isinstance(axis_obj, Exception):
        raise RuntimeError(f"Connected ODrive does not expose {axis_name}: {axis_obj}")

    axis_cfg = _safe_getattr(axis_obj, "config")
    if isinstance(axis_cfg, Exception):
        raise RuntimeError(f"Connected ODrive does not expose {axis_name}.config: {axis_cfg}")
    roots[f"odrv.{axis_name}.config"] = axis_cfg

    for child_name, child_obj in _iter_named_children(device):
        if child_name in {"config", "can", axis_name}:
            continue
        child_cfg = _safe_getattr(child_obj, "config")
        if child_cfg is not None and not isinstance(child_cfg, Exception) and not callable(child_cfg):
            roots[f"odrv.{child_name}.config"] = child_cfg

    for child_name, child_obj in _iter_named_children(axis_obj):
        if child_name == "config":
            continue
        child_cfg = _safe_getattr(child_obj, "config")
        if child_cfg is not None and not isinstance(child_cfg, Exception) and not callable(child_cfg):
            roots[f"odrv.{axis_name}.{child_name}.config"] = child_cfg
        # Some persistent config objects live one level deeper under runtime
        # containers such as axis.motor.motor_thermistor.config. Discover
        # those nested config leaves without broadening the dump to all
        # runtime-state subtrees.
        for grandchild_name, grandchild_obj in _iter_named_children(child_obj):
            grandchild_cfg = _safe_getattr(grandchild_obj, "config")
            if grandchild_cfg is not None and not isinstance(grandchild_cfg, Exception) and not callable(grandchild_cfg):
                roots[f"odrv.{axis_name}.{child_name}.{grandchild_name}.config"] = grandchild_cfg

    return roots


def _collect_full_roots(device: Any, axis_name: str) -> OrderedDict[str, Any]:
    roots: OrderedDict[str, Any] = OrderedDict()
    roots["odrv"] = device

    axis_obj = _safe_getattr(device, axis_name)
    if not isinstance(axis_obj, Exception):
        roots[f"odrv.{axis_name}"] = axis_obj
    return roots


def _walk_config_obj(
    obj: Any,
    prefix: str,
    flat: dict[str, Any],
    errors: list[dict[str, str]],
    *,
    max_depth: int = 12,
    _depth: int = 0,
    _visited: set[int] | None = None,
) -> None:
    if _visited is None:
        _visited = set()
    if _depth > max_depth:
        errors.append({"path": prefix, "error": f"max_depth_exceeded:{max_depth}"})
        return

    obj_id = id(obj)
    if obj_id in _visited:
        return
    _visited.add(obj_id)

    for name in sorted(dir(obj)):
        if name.startswith("_") or name in _SKIP_NAMES:
            continue

        path = f"{prefix}.{name}" if prefix else name
        value = _safe_getattr(obj, name)

        if isinstance(value, Exception):
            errors.append({"path": path, "error": str(value)})
            continue
        if callable(value):
            continue

        if _is_scalar(value):
            flat[path] = _normalize_scalar(value)
            continue

        if isinstance(value, (list, tuple)):
            flat[path] = [_normalize_scalar(v) if _is_scalar(v) else str(v) for v in value]
            continue

        _walk_config_obj(
            value,
            path,
            flat,
            errors,
            max_depth=max_depth,
            _depth=_depth + 1,
            _visited=_visited,
        )


def _walk_full_obj(
    obj: Any,
    prefix: str,
    flat: dict[str, Any],
    errors: list[dict[str, str]],
    *,
    max_depth: int = 16,
    _depth: int = 0,
    _visited: set[int] | None = None,
) -> None:
    if _visited is None:
        _visited = set()
    if _depth > max_depth:
        errors.append({"path": prefix, "error": f"max_depth_exceeded:{max_depth}"})
        return

    obj_id = id(obj)
    if obj_id in _visited:
        return
    _visited.add(obj_id)

    for name in sorted(dir(obj)):
        if name.startswith("_") or name in _SKIP_NAMES:
            continue

        path = f"{prefix}.{name}" if prefix else name
        value = _safe_getattr(obj, name)

        if isinstance(value, Exception):
            errors.append({"path": path, "error": str(value)})
            continue
        if callable(value):
            continue

        if _is_scalar(value):
            flat[path] = _normalize_scalar(value)
            continue

        if isinstance(value, (list, tuple)):
            flat[path] = [_normalize_scalar(v) if _is_scalar(v) else str(v) for v in value]
            continue

        _walk_full_obj(
            value,
            path,
            flat,
            errors,
            max_depth=max_depth,
            _depth=_depth + 1,
            _visited=_visited,
        )


def _metadata(device: Any, device_cfg: dict[str, Any], axis_name: str) -> dict[str, Any]:
    fw_major = _safe_getattr(device, "fw_version_major")
    fw_minor = _safe_getattr(device, "fw_version_minor")
    fw_revision = _safe_getattr(device, "fw_version_revision")
    fw_unreleased = _safe_getattr(device, "fw_version_unreleased")
    commit_hash = _safe_getattr(device, "commit_hash")

    return {
        "device_name": str(device_cfg["name"]),
        "serial_number_text": _device_serial_text(device),
        "serial_number_hex": None if _device_serial_int(device) is None else f"0x{_device_serial_int(device):x}",
        "configured_serial_hex": "0x" + _normalize_serial(str(device_cfg["serial_number"])),
        "configured_serial_int": _config_serial_int(device_cfg),
        "axis": str(axis_name),
        "firmware": {
            "major": None if isinstance(fw_major, Exception) else int(fw_major),
            "minor": None if isinstance(fw_minor, Exception) else int(fw_minor),
            "revision": None if isinstance(fw_revision, Exception) else int(fw_revision),
            "unreleased": None if isinstance(fw_unreleased, Exception) else int(fw_unreleased),
            "commit_hash": None if isinstance(commit_hash, Exception) else int(commit_hash),
        },
        "dumped_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


def _dump_one_device(device_cfg: dict[str, Any], output_dir: Path, axis_override: str | None) -> DumpResult:
    axis_name = axis_override or str(device_cfg.get("axis", "axis0"))
    device = None
    try:
        expected_serial = str(device_cfg["serial_number"])
        device = _connect_odrive(expected_serial, timeout_s=2.0)
        matched_cfg = _find_device_config(_device_serial_text(device) or "")
        if matched_cfg is None:
            raise RuntimeError("Connected ODrive serial is not present in HARDCODED_ODRIVES")
        if matched_cfg is not device_cfg:
            raise RuntimeError(
                f"Connected config mismatch: expected {device_cfg['name']}, matched {matched_cfg['name']}"
            )

        flat: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        roots = _collect_config_roots(device, axis_name)
        for root_path, root_obj in roots.items():
            _walk_config_obj(root_obj, root_path, flat, errors)

        tree: dict[str, Any] = {}
        for path, value in sorted(flat.items()):
            _insert_tree(tree, path, value)

        metadata = _metadata(device, device_cfg, axis_name)
        flat_doc = {
            "_metadata": metadata,
            "_roots": list(roots.keys()),
            "_errors": errors,
            "parameters": {k: flat[k] for k in sorted(flat.keys())},
        }
        tree_doc = {
            "_metadata": metadata,
            "_roots": list(roots.keys()),
            "_errors": errors,
            "config": tree,
        }

        full_flat: dict[str, Any] = {}
        full_errors: list[dict[str, str]] = []
        full_roots = _collect_full_roots(device, axis_name)
        for root_path, root_obj in full_roots.items():
            _walk_full_obj(root_obj, root_path, full_flat, full_errors)

        full_tree: dict[str, Any] = {}
        for path, value in sorted(full_flat.items()):
            _insert_tree(full_tree, path, value)

        full_flat_doc = {
            "_metadata": metadata,
            "_roots": list(full_roots.keys()),
            "_errors": full_errors,
            "parameters": {k: full_flat[k] for k in sorted(full_flat.keys())},
        }
        full_tree_doc = {
            "_metadata": metadata,
            "_roots": list(full_roots.keys()),
            "_errors": full_errors,
            "config": full_tree,
        }

        base = f"{device_cfg['name']}__{axis_name}"
        flat_path = output_dir / f"{base}__config_flat.json"
        tree_path = output_dir / f"{base}__config_tree.yaml"
        full_flat_path = output_dir / f"{base}__full_flat.json"
        full_tree_path = output_dir / f"{base}__full_tree.yaml"
        flat_path.write_text(json.dumps(flat_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tree_path.write_text(yaml.safe_dump(tree_doc, sort_keys=False, allow_unicode=False), encoding="utf-8")
        full_flat_path.write_text(json.dumps(full_flat_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        full_tree_path.write_text(yaml.safe_dump(full_tree_doc, sort_keys=False, allow_unicode=False), encoding="utf-8")

        return DumpResult(
            device_name=str(device_cfg["name"]),
            serial_number=str(device_cfg["serial_number"]),
            axis=axis_name,
            flat_json_path=flat_path,
            tree_yaml_path=tree_path,
            parameter_count=len(flat),
            full_flat_json_path=full_flat_path,
            full_tree_yaml_path=full_tree_path,
            full_parameter_count=len(full_flat),
        )
    finally:
        if device is not None:
            _disconnect_odrive(device)


def _selected_device_configs(serial_number: str | None) -> list[dict[str, Any]]:
    if serial_number is None:
        return list(HARDCODED_ODRIVES)
    requested_cfg = _find_device_config(serial_number)
    if requested_cfg is None:
        raise RuntimeError(f"serial number {serial_number} is not present in HARDCODED_ODRIVES")
    return [requested_cfg]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial-number",
        default=None,
        help="Optional expected ODrive serial number. If omitted, dump all HARDCODED_ODRIVES.",
    )
    parser.add_argument("--axis", default=None, help="Override axis selection, e.g. axis0 or axis1")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to Logs/odrive_config_dumps/<timestamp>/",
    )
    parser.add_argument("--list-devices", action="store_true", help="Print HARDCODED_ODRIVES and exit")
    return parser.parse_args(argv)


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("Logs") / "odrive_config_dumps" / stamp


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.list_devices:
        _print_device_table()
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        selected = _selected_device_configs(args.serial_number)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results: list[DumpResult] = []
    failures: list[tuple[str, str]] = []
    for device_cfg in selected:
        print(f"[{device_cfg['name']}] Dumping configuration")
        try:
            result = _dump_one_device(device_cfg, output_dir, args.axis)
            results.append(result)
            print(
                f"[{device_cfg['name']}] Wrote {result.parameter_count} config parameters and "
                f"{result.full_parameter_count} full parameters:\n"
                f"  JSON: {result.flat_json_path}\n"
                f"  YAML: {result.tree_yaml_path}\n"
                f"  Full JSON: {result.full_flat_json_path}\n"
                f"  Full YAML: {result.full_tree_yaml_path}"
            )
        except Exception as exc:
            failures.append((str(device_cfg["name"]), str(exc)))
            print(f"[{device_cfg['name']}] ERROR: {exc}", file=sys.stderr)

    manifest = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "results": [
            {
                "device_name": r.device_name,
                "serial_number": r.serial_number,
                "axis": r.axis,
                "flat_json_path": str(r.flat_json_path),
                "tree_yaml_path": str(r.tree_yaml_path),
                "parameter_count": r.parameter_count,
                "full_flat_json_path": str(r.full_flat_json_path),
                "full_tree_yaml_path": str(r.full_tree_yaml_path),
                "full_parameter_count": r.full_parameter_count,
            }
            for r in results
        ],
        "failures": [{"device_name": name, "error": message} for name, message in failures],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"Summary: {len(results)} succeeded, {len(failures)} failed.")
    for name, message in failures:
        print(f"  {name}: {message}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
