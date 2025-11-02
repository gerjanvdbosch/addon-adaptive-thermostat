#!/command/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

bashio::log.info "Preparing to start..."

# Read required/optional config from add-on options; use internal defaults if option absent
MODEL_DIR="$(bashio::config 'model_dir' || echo '/data/adaptive_thermostat')"

# MAIN_PY and PYTHON_BIN are fixed defaults (not read from bashio)
MAIN_PY="/app/run_proc.py"
PYTHON_BIN="python3"

# Ensure model dir exists and is writable
mkdir -p "${MODEL_DIR}"
if [ ! -d "${MODEL_DIR}" ] || [ ! -w "${MODEL_DIR}" ]; then
  bashio::log.fatal "MODEL_DIR ${MODEL_DIR} is not writable or cannot be created"
  exit 1
fi

# Log directory under MODEL_DIR
LOG_DIR="${LOG_DIR:-${MODEL_DIR}/logs}"
mkdir -p "${LOG_DIR}"
if [ ! -d "${LOG_DIR}" ] || [ ! -w "${LOG_DIR}" ]; then
  bashio::log.fatal "LOG_DIR ${LOG_DIR} is not writable or cannot be created"
  exit 1
fi

STDOUT_LOG="${LOG_DIR}/service.out"
STDERR_LOG="${LOG_DIR}/service.err"

bashio::log.info "Using MODEL_DIR=${MODEL_DIR}"
bashio::log.info "Using LOG_DIR=${LOG_DIR}"
bashio::log.info "Starting ${PYTHON_BIN} ${MAIN_PY}"

# helper to also append timestamped lines to stdout log
_log_to_file() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "${STDOUT_LOG}"
}

_child_pid=0

# Forward termination signals to child and attempt graceful shutdown
_term() {
  bashio::log.info "Received termination signal, stopping child ${_child_pid}"
  _log_to_file "Received termination signal, stopping child ${_child_pid}"

  if [ "${_child_pid}" -ne 0 ]; then
    kill -TERM "${_child_pid}" 2>/dev/null || true

    # wait up to 10s for graceful shutdown
    for i in $(seq 1 10); do
      if kill -0 "${_child_pid}" 2>/dev/null; then
        sleep 1
      else
        break
      fi
    done

    if kill -0 "${_child_pid}" 2>/dev/null; then
      bashio::log.warning "Child did not exit, sending KILL"
      _log_to_file "Child did not exit, sending KILL"
      kill -KILL "${_child_pid}" 2>/dev/null || true
    fi
  fi

  bashio::log.info "Exiting run.sh"
  _log_to_file "Exiting run.sh"
  exit 0
}

trap _term SIGTERM SIGINT

# Start the python process unbuffered and capture output
"${PYTHON_BIN}" -u "${MAIN_PY}" >> "${STDOUT_LOG}" 2>> "${STDERR_LOG}" &
_child_pid=$!

# Monitor child in background so we can log its exit code promptly
(
  wait "${_child_pid}"
  rc=$?
  if [ "${rc}" -eq 0 ]; then
    bashio::log.info "Child exited with code ${rc}"
    _log_to_file "Child exited with code ${rc}"
  else
    bashio::log.error "Child exited with code ${rc}"
    _log_to_file "Child exited with code ${rc}"
  fi
) &

# Block main script until child finishes; trap will handle signals
wait "${_child_pid}"
rc=$?

# Ensure final log (in case monitor subshell raced)
if [ "${rc}" -eq 0 ]; then
  bashio::log.info "Child exited with code ${rc}"
  _log_to_file "Child exited with code ${rc}"
else
  bashio::log.error "Child exited with code ${rc}"
  _log_to_file "Child exited with code ${rc}"
fi

exit "${rc}"
