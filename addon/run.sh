#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Preparing to start..."

export MODEL_DIR="/config/models"
export DB_DIR="/config/db"

mkdir -p "${MODEL_DIR}"
mkdir -p "${DB_DIR}"

export DB_PATH="${DB_DIR}/setpoints.sqlite"
export MODEL_PATH="${MODEL_DIR}/delta_model.joblib"
export MODEL_PATH_FULL="${MODEL_DIR}/full_model.joblib"

if bashio::config.true 'shadow_mode'; then
  export SHADOW_MODE=1
fi

export CLIMATE_ENTITY="$(bashio::config 'climate_entity')"
export SHADOW_SETPOINT="$(bashio::config 'shadow_setpoint')"
export INFERENCER_INTERVAL_SECONDS="$(bashio::config 'inferencer_interval_seconds')"
export SAMPLE_INTERVAL_SECONDS="$(bashio::config 'sample_interval_seconds')"
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
