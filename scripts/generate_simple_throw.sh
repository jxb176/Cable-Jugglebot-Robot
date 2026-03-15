#!/usr/bin/env bash
set -euo pipefail
python -m jugglebot.apps.plantraj --profile simple_throw --sample-hz 100 --out-dir . "$@"
