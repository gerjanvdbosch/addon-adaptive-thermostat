#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Preparing to start..."

mkdir -p /config/models
mkdir -p /config/db
mkdir -p /config/logs

python3 -u ./src/main.py
