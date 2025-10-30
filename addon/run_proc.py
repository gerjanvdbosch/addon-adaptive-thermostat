#!/usr/bin/env python3
# run_proc.py
import os
import sys
import signal
import time
import logging
import yaml

# Ensure project dir is the addon/app directory
HERE = os.path.dirname(__file__)
os.chdir(HERE)

# Load defaults
CFG_PATH = os.path.join(HERE, "config_default.yaml")
cfg_default = {}
try:
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r") as f:
            cfg_default = yaml.safe_load(f) or {}
except Exception:
    cfg_default = {}

# Environment / model dir
MODEL_DIR = os.environ.get("MODEL_DIR", cfg_default.get("model_dir", "/data"))
os.makedirs(MODEL_DIR, exist_ok=True)

# Simple logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("adaptive_thermostat")

# Import the Flask app (api_service exposes `app`)
try:
    from api_service import app as flask_app
except Exception as e:
    logger.exception("Failed to import api_service.app: %s", e)
    raise

# Graceful shutdown handling for Flask built-in server
shutdown_flag = False


def _handle_signal(signum, frame):
    global shutdown_flag
    logger.info("Received signal %s, shutting down.", signum)
    shutdown_flag = True
    # If using Flask's built-in server, calling sys.exit will terminate main thread
    # We set flag and allow the server to stop naturally after request handling.


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

def run():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")

    logger.info("Starting Flask app on %s:%d (debug=%s). MODEL_DIR=%s", host, port, debug, MODEL_DIR)

    # Use Flask's built-in server in threaded mode; for production you can swap to gunicorn/waitress
    try:
        flask_app.run(host=host, port=port, threaded=True, debug=debug, use_reloader=False)
    except Exception as e:
        logger.exception("Flask app terminated with exception: %s", e)
        raise

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting.")
    except SystemExit:
        logger.info("SystemExit caught, exiting.")
    except Exception:
        logger.exception("Unhandled exception, exiting.")
    finally:
        logger.info("Process exiting.")
