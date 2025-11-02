#!/command/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

bashio::log.info "Preparing to start..."
bashio::config.require 'data_path'

DATA_PATH=$(bashio::config 'data_path')
mkdir -p "${DATA_PATH}"

python3 -u ./src/run_proc.py
