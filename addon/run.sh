#!/command/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

bashio::log.info "Preparing to start..."

MODEL_DIR="$(bashio::config 'model_dir' || echo '/homeassistant/data/adaptive_thermostat')"

mkdir -p "${MODEL_DIR}"

python3 -u ./run_proc.py
