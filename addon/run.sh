#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Preparing to start..."

mkdir -p /config/models
mkdir -p /config/db

export CLIMATE_ENTITY="$(bashio::config 'climate_entity')"
export SAMPLE_INTERVAL_SECONDS="$(bashio::config 'sample_interval_seconds')"
export PARTIAL_FIT_INTERVAL_SECONDS="$(bashio::config 'partial_fit_interval_seconds')"
export FULL_RETRAIN_TIME="$(bashio::config 'full_retrain_time')"
export MIN_SETPOINT="$(bashio::config 'min_setpoint')"
export MAX_SETPOINT="$(bashio::config 'max_setpoint')"
export MIN_CHANGE_THRESHOLD="$(bashio::config 'min_change_threshold')"
export BUFFER_DAYS="$(bashio::config 'buffer_days')"
export SENSORS="$(bashio::config 'sensors' || echo '{}')"

python3 -u ./src/main.py
