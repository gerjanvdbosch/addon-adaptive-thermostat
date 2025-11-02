#!/usr/bin/env python3
import os
import sys
import signal
import logging

# import centralized config
from config import cfg_default  # CFG_PATH is available in config if needed

# logging from config only
LOG_LEVEL = str(cfg_default.get("log_level", "INFO")).upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("adaptive_thermostat")

# import the Flask app (api_service.py expected in this directory)
try:
    from api_service import app as flask_app
except Exception as e:
    logger.exception("Failed to import app.api_service.app: %s", e)
    logger.error("sys.path=%s", sys.path)
    logger.error("cwd=%s", os.getcwd())
    try:
        logger.error("files=%s", os.listdir(_here))
    except Exception:
        pass
    raise

def _handle_signal(signum, frame):
    logger.info("Received signal %s, shutting down.", signum)
    try:
        sys.exit(0)
    except SystemExit:
        pass

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

def run():
    host = cfg_default.get("host", "0.0.0.0")
    port = int(cfg_default.get("port", 5189))
    debug = bool(cfg_default.get("flask_debug", False))
    logger.info("Starting Flask app on %s:%d (debug=%s).", host, port, debug)
    flask_app.run(host=host, port=port, threaded=True, debug=debug, use_reloader=False)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting.")
    except Exception:
        logger.exception("Unhandled exception, exiting.")
