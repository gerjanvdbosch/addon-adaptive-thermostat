#!/usr/bin/env bash
set -euo pipefail

# Require and read options via bashio (Supervisor add-on environment)
# source bashio helpers (available in Supervisor add-on base images)
if [ -f /usr/lib/bashio/helpers.sh ]; then
  # shellcheck disable=SC1091
  source /usr/lib/bashio/helpers.sh
fi

# Ensure model_dir option exists
bashio::config.require 'model_dir'
MODEL_DIR="$(bashio::config 'model_dir')"

# Fallback safety
MODEL_DIR="${MODEL_DIR:-/data}"
export MODEL_DIR
mkdir -p "$MODEL_DIR"

# startup marker for verification/backups (non-fatal)
printf '%s run.sh: starting (pid %s, MODEL_DIR=%s)\n' "$(date --iso-8601=seconds)" "$$" "$MODEL_DIR"

# Start WSGI entry (api_service exposes app via run_proc.py)
exec python3 -u /opt/adaptive_thermostat/addon/run_proc.py
