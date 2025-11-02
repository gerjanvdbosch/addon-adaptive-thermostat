#!/command/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

bashio::log.info "Preparing to start..."

export HA_TOKEN="$(bashio::addon.supervisor.token)"

python3 -u ./src/run.py
