#!/usr/bin/with-contenv bashio

bashio::log.info "Preparing to start..."

python3 -u ./src/run.py
