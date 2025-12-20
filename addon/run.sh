#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Preparing to start..."

export MODEL_DIR="/config/models"
export DB_DIR="/config/db"

mkdir -p "${MODEL_DIR}"
mkdir -p "${DB_DIR}"

export DB_PATH="${DB_DIR}/database.sqlite"
export THERMOSTAT_MODEL_PATH="${MODEL_DIR}/thermostat_model.joblib"
export SOLAR_MODEL_PATH="${MODEL_DIR}/solar_model.joblib"
export PRESENCE_MODEL_PATH="${MODEL_DIR}/presence_model.joblib"
export THERMAL_MODEL_PATH="${MODEL_DIR}/thermal_model.joblib"

export THERMOSTAT_ENTITY="$(bashio::config 'thermostat_entity')"
export SOLAR_ENTITY="$(bashio::config 'solar_entity')"
export PRESENCE_ENTITY="$(bashio::config 'presence_entity')"
export THERMAL_ENTITY="$(bashio::config 'thermal_entity')"
export THERMOSTAT_INTERVAL_SECONDS="$(bashio::config 'thermostat_interval_seconds')"
export SOLAR_INTERVAL_SECONDS="$(bashio::config 'solar_interval_seconds')"
export COOLDOWN_HOURS="$(bashio::config 'cooldown_hours')"
export STABILITY_HOURS="$(bashio::config 'stability_hours')"
export FULL_RETRAIN_TIME="$(bashio::config 'full_retrain_time')"
export MIN_SETPOINT="$(bashio::config 'min_setpoint')"
export MAX_SETPOINT="$(bashio::config 'max_setpoint')"
export MIN_CHANGE_THRESHOLD="$(bashio::config 'min_change_threshold')"
export BUFFER_DAYS="$(bashio::config 'buffer_days')"
export LOG_LEVEL="$(bashio::config 'log_level')"
export SENSORS="$(bashio::config 'sensors' || echo '{}')"

python3 -u ./src/main.py
