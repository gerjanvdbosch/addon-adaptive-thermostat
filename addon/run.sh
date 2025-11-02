#!/command/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

bashio::log.info "Preparing to start..."

SUPERVISOR_TOKEN="$(bashio::addon.supervisor.token)"
export SUPERVISOR_TOKEN

python3 -u ./src/run.py
