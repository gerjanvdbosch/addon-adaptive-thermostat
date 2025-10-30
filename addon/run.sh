#!/usr/bin/env bash
set -eo pipefail

# Config
MODEL_DIR="${MODEL_DIR:-/data}"

printf 'Preparing to start...\n'

export MODEL_DIR
mkdir -p "$MODEL_DIR"

# startup marker for verification/backups (non-fatal)
printf '%s run.sh: starting (pid %s, MODEL_DIR=%s)\n' "$(date --iso-8601=seconds)" "$$" "$MODEL_DIR"

# Start WSGI entry (api_service exposes app via run_proc.py)
exec python3 -u /app/run_proc.py
