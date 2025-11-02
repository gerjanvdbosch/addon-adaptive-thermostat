#!/command/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

bashio::log.info "Preparing to start..."

bashio::config.require 'data_path'

DATA_PATH=$(bashio::config 'data_path')

mkdir -p "${DATA_PATH}"
if [ ! -d "${DATA_PATH}" ] || [ ! -w "${DATA_PATH}" ]; then
  bashio::log.fatal "DATA_PATH ${DATA_PATH} is not writable or cannot be created"
  exit 1
fi

python3 -u ./src/run_proc.py
