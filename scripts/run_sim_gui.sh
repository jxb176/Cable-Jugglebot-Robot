#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-mesa}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-radeonsi}"
export LIBGL_DRI3_DISABLE="${LIBGL_DRI3_DISABLE:-1}"
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-xcb_glx}"
export JUGGLEBOT_QT_OPENGL="${JUGGLEBOT_QT_OPENGL:-desktop}"

exec "${PYTHON_BIN}" -m jugglebot.apps.controlui \
  --host "${JUGGLEBOT_HOST:-127.0.0.1}" \
  --tcp-port "${JUGGLEBOT_TCP_PORT:-5555}" \
  --udp-port "${JUGGLEBOT_UDP_PORT:-5556}" \
  "$@"
