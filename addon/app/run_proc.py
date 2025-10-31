#!/usr/bin/env python3
import os
import sys
import signal
import logging
import yaml

# Zorg dat de addon root op sys.path staat (de map waarin deze run_proc.py staat)
HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Werkdirectory = addon root
os.chdir(HERE)

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("adaptive_thermostat")

# Probeer config te laden (optioneel)
cfg_default = {}
CFG_PATH = os.path.join(HERE, "addon", "app", "config_default.yaml")
if not os.path.exists(CFG_PATH):
    CFG_PATH = os.path.join(HERE, "app", "config_default.yaml")
if os.path.exists(CFG_PATH):
    try:
        with open(CFG_PATH, "r") as f:
            cfg_default = yaml.safe_load(f) or {}
    except Exception:
        cfg_default = {}

# Zorg dat package 'app' importeerbaar is en importeer de Flask app
try:
    from app.api_service import app as flask_app
except Exception as e:
    logger.exception("Failed to import app.api_service.app: %s", e)
    # dump debug info to help locate import path issues
    logger.error("sys.path=%s", sys.path)
    logger.error("cwd=%s", os.getcwd())
    logger.error("files=%s", os.listdir('.'))
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
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", str(cfg_default.get("port", 5000))))
    debug = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    logger.info("Starting Flask app on %s:%d (debug=%s).", host, port, debug)
    flask_app.run(host=host, port=port, threaded=True, debug=debug, use_reloader=False)

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, exiting.")
    except Exception:
        logger.exception("Unhandled exception, exiting.")
