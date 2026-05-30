"""Shared winch and unit-conversion helpers."""

from __future__ import annotations

import math

from jugglebot.core.cable_ik import WinchCalibration


# +turns (ODrive) reduces cable length, so use negative mm/turn such that +mm command extends cable.
MM_PER_TURN = [-62.832] * 6  # 2*pi*10mm = 62.832 mm/turn, with sign convention applied

DEFAULT_WINCH_CAL = WinchCalibration(
    spool_radius_mm=[10.0] * 6,
    gear_ratio=[1.0] * 6,
    sign=[-1.0] * 6,
    zero_length_mm=[0.0] * 6,
)


def turns_to_mm(turns_list, cal: WinchCalibration = DEFAULT_WINCH_CAL):
    """Convert [turns] -> [mm] elementwise using calibration."""
    if not isinstance(turns_list, (list, tuple)) or len(turns_list) != 6:
        raise ValueError("turns_list must be length-6 list/tuple")
    out = []
    for i in range(6):
        trn = float(turns_list[i])
        r = float(cal.spool_radius_mm[i])
        if r <= 0.0:
            raise ValueError(f"spool_radius_mm[{i}] must be > 0")

        spool_turns = trn / float(cal.sign[i]) / float(cal.gear_ratio[i])
        dL = spool_turns * 2.0 * math.pi * r
        L = dL + float(cal.zero_length_mm[i])
        out.append(L)
    return out


def mm_to_turns(mm_list, cal: WinchCalibration = DEFAULT_WINCH_CAL):
    """Convert [mm] -> [turns] elementwise using calibration."""
    if not isinstance(mm_list, (list, tuple)) or len(mm_list) != 6:
        raise ValueError("mm_list must be length-6 list/tuple")
    out = []
    for i in range(6):
        mm = float(mm_list[i])
        r = float(cal.spool_radius_mm[i])
        if r <= 0.0:
            raise ValueError(f"spool_radius_mm[{i}] must be > 0")

        l0 = float(cal.zero_length_mm[i])
        dL = mm - l0
        spool_turns = dL / (2.0 * math.pi * r)
        motor_turns = spool_turns * float(cal.gear_ratio[i])
        out.append(float(cal.sign[i]) * motor_turns)
    return out


def coerce_vec6_to_mm(msg, field_name: str):
    vec = msg.get(field_name, [])
    units = (msg.get("units") or "mm").lower()
    if not isinstance(vec, (list, tuple)) or len(vec) != 6:
        raise ValueError(f"{field_name} must be length-6 list")
    vec = [float(x) for x in vec]

    if units == "mm":
        return vec
    if units == "turns":
        return turns_to_mm(vec)
    raise ValueError(f"Unknown units '{units}' (expected 'mm' or 'turns')")
