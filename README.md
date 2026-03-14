# Cable-Jugglebot-Robot
Robot controller

## Offline Trajectory Planning

The repository now includes a packaged planning library under `jugglebot.planning`.

- Generate a built-in throw profile:
  - `python -m jugglebot.apps.plantraj --profile simple_throw --out-dir .`
- Generate from a YAML endpoint/segment profile at a defined command rate:
  - `python -m jugglebot.apps.plantraj --profile-file src/jugglebot/profiles/simple_throw.yaml --command-rate-hz 500 --plot --out-dir .`
- Output files:
  - `pose_cmd.csv`
  - `pose_cmd_full.csv`
  - `trajectory_plot.png` (when `--plot` is used)

## Run Simple Throw In Simulation

1. Generate the throw trajectory:
   - `python -m jugglebot.apps.plantraj --profile simple_throw --command-rate-hz 500 --out-dir .`
2. Start the simulation daemon:
   - `python -m jugglebot.apps.simd --viewer --auto-enable`
3. In a second terminal, upload and play the generated trajectory:
   - `python -m jugglebot.apps.playtraj --csv pose_cmd.csv --auto-enable`
4. To include trajectory velocity/acceleration feedforward terms in control:
   - `python -m jugglebot.apps.playtraj --csv pose_cmd_full.csv --full-csv --auto-enable`
   - (`--full-csv` is optional when the CSV header includes `vx_mps/.../az_mps2`; auto-detected)
   - If sim runs slower than real-time, increase connection hold time so playback is not cut short:
   - `python -m jugglebot.apps.playtraj --csv pose_cmd_full.csv --auto-enable --wait-scale 3.0`

## Diagnostic Logging And Review

- While `simd`/`robotd` is running, `ControlBridge` now writes a structured diagnostic CSV:
  - `Logs/control_diag_YYYYMMDD_HHMMSS.csv`
- Logged signals include:
  - hand platform command and response (position/orientation and response rates)
  - platform wrench command and response (Fx/Fy/Fz, Tx/Ty)
  - spool command and response (mm and mm/s)
  - spool torque/tension commands
  - spool tension response (when provided by the driver, including MuJoCo sim)
  - bus, current, and temperature channels
- Review the newest log interactively with matplotlib:
  - `python -m jugglebot.apps.reviewlog`
  - When available, plots use `sim_time_s` on the x-axis (not wall time).
- Or review a specific file:
  - `python -m jugglebot.apps.reviewlog --log Logs/control_diag_20260314_120000.csv`

## Network Control Interface (Prototype)

- Run the prototype control UI as a packaged app:
  - `python -m jugglebot.apps.controlui --host <robot-host-or-ip> --tcp-port 5555 --udp-port 5556`
- Example on the same LAN:
  - `python -m jugglebot.apps.controlui --host 192.168.1.42`
- Environment variable alternatives:
  - `JUGGLEBOT_HOST`, `JUGGLEBOT_TCP_PORT`, `JUGGLEBOT_UDP_PORT`
