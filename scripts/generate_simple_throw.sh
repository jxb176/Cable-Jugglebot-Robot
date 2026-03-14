#!/usr/bin/env bash
set -euo pipefail
python -m jugglebot.apps.plantraj --profile simple_throw --sample-hz 500 --out-dir . "$@"
