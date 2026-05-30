"""Periodic axis-state logging helpers."""

from __future__ import annotations

import logging
import time


logger = logging.getLogger("robot")


def axes_state_logger(state):
    while True:
        try:
            pos = state.get_pos_fbk()
            vel = state.get_vel_fbk()
            bus = state.get_bus_voltage()
            busi = state.get_bus_current()
            temp_f = state.get_temp_fet()
            temp_m = state.get_temp_motor()
            st = state.get_state()

            fmt_pos = ", ".join("---" if x is None else f"{x:.3f}" for x in pos)
            fmt_vel = ", ".join("---" if v is None else f"{v:.3f}" for v in vel)
            fmt_bus = ", ".join("---" if b is None else f"{b:.2f}" for b in bus)
            fmt_busi = ", ".join("---" if i is None else f"{i:.2f}" for i in busi)
            fmt_tf = ", ".join("---" if x is None else f"{x:.1f}" for x in temp_f)
            fmt_tm = ", ".join("---" if x is None else f"{x:.1f}" for x in temp_m)

            logger.info(
                f"[LOG] State={st} "
                f"Pos=[{fmt_pos}] "
                f"Vel=[{fmt_vel}] "
                f"BusV=[{fmt_bus}]"
                f"BusI=[{fmt_busi}]"
                f"TempFET=[{fmt_tf}] "
                f"TempMotor=[{fmt_tm}]"
            )
        except Exception as e:
            logger.error(f"[LOG] Error: {e}")
        time.sleep(1.0)
