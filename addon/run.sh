#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Preparing to start..."

export MODEL_DIR="/config/models"
export DB_DIR="/config/db"

mkdir -p "${MODEL_DIR}"
mkdir -p "${DB_DIR}"

export DB_PATH="${DB_DIR}/samples.sqlite"
export MODEL_PATH_PARTIAL="${MODEL_DIR}/partial_model.joblib"
export MODEL_PATH_FULL="${MODEL_DIR}/full_model.joblib"

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
