#!/command/with-contenv bashio
# shellcheck shell=bash
set -euo pipefail

bashio::log.info "Preparing to start..."

MODEL_DIR="$(bashio::config 'model_dir' || echo '/data/adaptive_thermostat')"
MAIN_PY="/app/run_proc.py"
PYTHON_BIN="python3"

mkdir -p "${MODEL_DIR}"
if [ ! -d "${MODEL_DIR}" ] || [ ! -w "${MODEL_DIR}" ]; then
  bashio::log.fatal "MODEL_DIR ${MODEL_DIR} is not writable or cannot be created"
  exit 1
fi

LOG_DIR="${LOG_DIR:-${MODEL_DIR}/logs}"
mkdir -p "${LOG_DIR}"
if [ ! -d "${LOG_DIR}" ] || [ ! -w "${LOG_DIR}" ]; then
  bashio::log.fatal "LOG_DIR ${LOG_DIR} is not writable or cannot be created"
  exit 1
fi

STDOUT_LOG="${LOG_DIR}/service.out"
STDERR_LOG="${LOG_DIR}/service.err"

_log_to_file() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >> "${STDOUT_LOG}"; }

bashio::log.info "Starting ${PYTHON_BIN} ${MAIN_PY}"
_log_to_file "Starting ${PYTHON_BIN} ${MAIN_PY}"

_child_pid=0

# Trap: forward signals to child and try graceful shutdown
_term() {
  bashio::log.info "Received termination signal, stopping child ${_child_pid}"
  _log_to_file "Received termination signal, stopping child ${_child_pid}"

  if [ "${_child_pid}" -ne 0 ]; then
    kill -TERM "${_child_pid}" 2>/dev/null || true

    # wait up to 10s for child exit
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

# Start child in the current shell (background) — no subshell wrapper
"${PYTHON_BIN}" -u "${MAIN_PY}" >> "${STDOUT_LOG}" 2>> "${STDERR_LOG}" &
_child_pid=$!

# Directly wait on the child PID in this shell (no monitor subshell)
wait "${_child_pid}"
rc=$?

# Log exit
if [ "${rc}" -eq 0 ]; then
  bashio::log.info "Child exited with code ${rc}"
  _log_to_file "Child exited with code ${rc}"
else
  bashio::log.error "Child exited with code ${rc}"
  _log_to_file "Child exited with code ${rc}"
fi

exit "${rc}"
