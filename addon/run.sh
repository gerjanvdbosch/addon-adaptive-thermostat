#!/command/with-contenv bashio
# shellcheck shell=bash

bashio::log.info "Preparing to start..."

bashio::config.require 'model_dir'

# Configurable paths (omgeving kan deze overschrijven)
# MODEL_DIR="${MODEL_DIR:-/data}"
# LOG_DIR="${MODEL_DIR:-/tmp}/logs"
# MAIN_PY="${MAIN_PY:-/app/run_proc.py}"   # pas aan naar jouw entrypoint (bijv. /app/app.py)
# PYTHON_BIN="${PYTHON_BIN:-python3}"

# mkdir -p "${LOG_DIR}"

# STDOUT_LOG="${LOG_DIR}/service.out"
# STDERR_LOG="${LOG_DIR}/service.err"

# # log function with timestamp
# log() {
#   printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" >> "${STDOUT_LOG}"
# }

# # Start child process and forward signals
# _child_pid=0
# _term() {
#   log "Received termination signal, stopping child ${_child_pid}"
#   if [ "${_child_pid}" -ne 0 ]; then
#     kill -TERM "${_child_pid}" 2>/dev/null || true
#     # wait up to 10s for graceful shutdown, then force
#     for i in $(seq 1 10); do
#       if kill -0 "${_child_pid}" 2>/dev/null; then
#         sleep 1
#       else
#         break
#       fi
#     done
#     if kill -0 "${_child_pid}" 2>/dev/null; then
#       log "Child did not exit, sending KILL"
#       kill -KILL "${_child_pid}" 2>/dev/null || true
#     fi
#   fi
#   log "Exiting run.sh"
#   exit 0
# }

# trap _term SIGTERM SIGINT

# log "Starting service: ${PYTHON_BIN} ${MAIN_PY}"
# # run the python process, redirect logs to files, keep it in foreground
# # NOTE: use -u for unbuffered output
# "${PYTHON_BIN}" -u "${MAIN_PY}" >> "${STDOUT_LOG}" 2>> "${STDERR_LOG}" &
# _child_pid=$!

# # wait for the child process to finish
# wait "${_child_pid}"
# rc=$?
# log "Child exited with code ${rc}"
# exit "${rc}"
