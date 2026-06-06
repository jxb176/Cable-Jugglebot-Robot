# Cable-Jugglebot-Robot
Robot controller

## Offline Trajectory Planning

The repository now includes a packaged planning library under `jugglebot.planning`.

- Generate a built-in throw profile:
  - `python -m jugglebot.apps.plantraj --profile simple_throw --out-dir .`
- Generate from a YAML endpoint/segment profile at a defined command rate:
  - `python -m jugglebot.apps.plantraj --profile-file src/jugglebot/profiles/simple_throw.yaml --command-rate-hz 500 --plot --out-dir .`
- Generate directly from a pattern project YAML by sampling one hand trajectory:
  - `python -m jugglebot.apps.plantraj --pattern-file src/jugglebot/profiles/one_ball_one_hand.yaml --hand right --command-rate-hz 500 --plot --out-dir .`
- Output files:
  - `pose_cmd.csv`
  - `pose_cmd_full.csv`
  - `trajectory_plot.png` (when `--plot` is used)

## Run Simple Throw In Simulation

1. Generate the throw trajectory:
   - `python -m jugglebot.apps.plantraj --profile simple_throw --command-rate-hz 500 --out-dir .`
2. Start the simulation daemon:
   - `python -m jugglebot.apps.simd --viewer --auto-enable`
   - Uses `src/jugglebot/config/sim.yaml` by default
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

- Install GUI dependencies:
  - `pip install -e .[gui]`
- Runtime configs:
  - hardware daemon default: `src/jugglebot/config/robot.yaml`
  - simulation daemon default: `src/jugglebot/config/sim.yaml`
- Run the prototype control UI as a packaged app:
  - `python -m jugglebot.apps.controlui --host <robot-host-or-ip> --tcp-port 5555 --udp-port 5556`
- Example on the same LAN:
  - `python -m jugglebot.apps.controlui --host 192.168.1.42`
- Environment variable alternatives:
  - `JUGGLEBOT_HOST`, `JUGGLEBOT_TCP_PORT`, `JUGGLEBOT_UDP_PORT`

## Pattern Studio

- Launch the unconstrained juggling pattern editor:
  - `python -m jugglebot.apps.patternui`
- Launch the standalone B-spline sandbox:
  - `python -m jugglebot.apps.bsplineui`
- Open an existing pattern YAML:
  - `python -m jugglebot.apps.patternui --file path/to/pattern.yaml`
- Checked-in sample pattern:
  - `src/jugglebot/patterns/examples/three_ball_cascade.yaml`
- Additional sample patterns:
  - `src/jugglebot/patterns/examples/one_ball_one_hand.yaml`
  - `src/jugglebot/patterns/examples/two_balls_one_hand.yaml`
- Current scope:
  - define throw and catch points per event
  - author unconstrained hand trajectory keyframes in a `hands` YAML section with explicit waypoint velocities
  - choose `cubic`, `quintic`, or `bspline` interpolation for authored hand segments
  - for `bspline` hand segments, configure cubic or quintic spline degree, control-point count, tangent direction, and a separate scalar `path_speed`
  - changing `path_speed` on a `bspline` segment changes traversal timing along the curve without changing the curve geometry
  - enforce throw hand position/velocity to match the ball throw state with zero acceleration at the throw point
  - enforce catch hand position with velocity set by `catch_velocity_scale * ball_velocity` and zero acceleration at the catch point
  - edit throw/catch timing and positions directly
  - adjust the selected event with live sliders and update the hand trajectory in real time
  - preview ball flights, left/right hand trajectories, authored hand waypoints, and throw/catch anchors in `x / y`, `x / z`, `y / z`, and isometric views
  - click authored hand waypoints in the `x / z` and `y / z` views to select them and drag them live
  - drag authored waypoint velocity arrows in the `x / z` and `y / z` views with immediate trajectory feedback; for `bspline` waypoints the arrow steers tangent direction while `path_speed` remains a separate scalar control
  - inspect the generated control polygon for the selected outgoing `bspline` segment in the preview
  - highlight the selected waypoint and its connected hand spline segment in the preview
  - inspect stacked position / velocity / acceleration plots for the selected hand
  - switch between looped playback and single-run playback
  - multi-select events in the throw list and delete them as a batch
  - save/load YAML pattern projects
- The editor is intentionally unconstrained for now:
  - no robot kinematic limits
  - no path optimization
  - no cable/Stewart feasibility checks

## B-Spline Sandbox

- A lightweight 2D sandbox for B-spline geometry experiments, separate from the juggling pattern editor.
- Supports:
  - changing spline degree and control-point count
  - dragging endpoints independently
  - dragging start/end tangent vectors independently
  - dragging interior control points and the end-adjacent control point
  - panning and zooming the canvas while inspecting the sampled curve, control polygon, knot vector, and arc length
