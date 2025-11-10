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

if bashio::config.true 'shadow_mode'; then
  export SHADOW_MODE=1
fi

if bashio::config.true 'use_unlabeled'; then
  export USE_UNLABELED=1
fi

export CLIMATE_ENTITY="$(bashio::config 'climate_entity')"
export SHADOW_SETPOINT="$(bashio::config 'shadow_setpoint')"
export INFERENCER_INTERVAL_SECONDS="$(bashio::config 'inferencer_interval_seconds')"
export SAMPLE_INTERVAL_SECONDS="$(bashio::config 'sample_interval_seconds')"
export PARTIAL_FIT_INTERVAL_SECONDS="$(bashio::config 'partial_fit_interval_seconds')"
export FULL_RETRAIN_TIME="$(bashio::config 'full_retrain_time')"
export MIN_SETPOINT="$(bashio::config 'min_setpoint')"
export MAX_SETPOINT="$(bashio::config 'max_setpoint')"
export MIN_CHANGE_THRESHOLD="$(bashio::config 'min_change_threshold')"
export BUFFER_DAYS="$(bashio::config 'buffer_days')"
export PARTIAL_LEARNING_RATE="$(bashio::config 'partial_learning_rate')"
export PARTIAL_ETA0="$(bashio::config 'partial_eta0')"
export PARTIAL_ALPHA="$(bashio::config 'partial_alpha')"        
export PSEUDO_LIMIT="$(bashio::config 'pseudo_limit')"
export WEIGHT_LABEL="$(bashio::config 'weight_label')"
export WEIGHT_PSEUDO="$(bashio::config 'weight_pseudo')"
export LOG_LEVEL="$(bashio::config 'log_level')"
export SENSORS="$(bashio::config 'sensors' || echo '{}')"

python3 -u ./src/main.py
