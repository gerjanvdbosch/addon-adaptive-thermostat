#!/usr/bin/with-contenv bashio

set -euo pipefail

printf 'Preparing to start...\n'

# If Supervisor runtime provides with-contenv and bashio, use them to source helpers
if [ -x /command/with-contenv ] && [ -f /usr/lib/bashio/helpers.sh ]; then
  exec /command/with-contenv bashio "$0" "$@"
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
exec python3 -u /app/run_proc.py
