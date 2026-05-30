#!/usr/bin/env python3
"""Launch the unconstrained juggling pattern editor."""

from __future__ import annotations

import argparse
from pathlib import Path

from jugglebot.patterns import build_three_ball_cascade_pattern, load_pattern_project
from jugglebot.patterns.tk_app import launch_pattern_studio


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive juggling pattern editor")
    parser.add_argument("--file", type=str, default=None, help="Optional YAML pattern project to open")
    args = parser.parse_args()

    if args.file:
        project = load_pattern_project(Path(args.file))
        launch_pattern_studio(project=project, initial_path=args.file)
        return

    launch_pattern_studio(project=build_three_ball_cascade_pattern())


if __name__ == "__main__":
    main()
