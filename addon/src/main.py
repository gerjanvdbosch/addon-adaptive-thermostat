import os
import threading
import time
import json
import logging
import uvicorn

from apscheduler.schedulers.background import BackgroundScheduler
from ha_client import HAClient
from collector import Collector
from coordinator import ClimateCoordinator
from webapi import set_coordinator

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("apscheduler").setLevel(logging.WARNING)


def _start_api(host: str, port: int):
    uvicorn.run("webapi:app", host=host, port=port, log_level="warning")


def _load_options():
    try:
        sensors = json.loads(os.getenv("SENSORS", None))
    except Exception:
        sensors = None

    return {
        "thermostat_entity": os.getenv("THERMOSTAT_ENTITY"),
        "solar_entity": os.getenv("SOLAR_ENTITY"),
        "presence_entity": os.getenv("PRESENCE_ENTITY"),
        "thermal_entity": os.getenv("THERMAL_ENTITY"),
        "thermostat_interval_seconds": int(
            os.getenv("THERMOSTAT_INTERVAL_SECONDS", 60)
        ),
        "dhw_entity": os.getenv("DHW_ENTITY"),
        "solar_interval_seconds": int(os.getenv("SOLAR_INTERVAL_SECONDS", 15)),
        "cooldown_hours": float(os.getenv("COOLDOWN_HOURS", 2)),
        "full_retrain_time": os.getenv("FULL_RETRAIN_TIME", "03:00"),
        "stability_hours": float(os.getenv("STABILITY_HOURS", 8)),
        "min_setpoint": float(os.getenv("MIN_SETPOINT", 15.0)),
        "max_setpoint": float(os.getenv("MAX_SETPOINT", 24.0)),
        "min_change_threshold": float(os.getenv("MIN_CHANGE_THRESHOLD", 0.5)),
        "buffer_days": int(os.getenv("BUFFER_DAYS", 730)),
        "webapi_host": os.getenv("WEBAPI_HOST", "0.0.0.0"),
        "webapi_port": int(os.getenv("WEBAPI_PORT", 8000)),
        "thermostat_model_path": os.getenv("THERMOSTAT_MODEL_PATH"),
        "solar_model_path": os.getenv("SOLAR_MODEL_PATH"),
        "presence_model_path": os.getenv("PRESENCE_MODEL_PATH"),
        "thermal_model_path": os.getenv("THERMAL_MODEL_PATH"),
        "sensors": sensors,
    }


def main():
    logger.info("System: Starting...")

    opts = _load_options()

    ha = HAClient(opts)
    collector = Collector(ha, opts)

    logger.info("System: Initializing...")
    coordinator = ClimateCoordinator(ha, collector, opts)

    set_coordinator(coordinator)

    api_thread = threading.Thread(
        target=_start_api,
        args=(opts.get("webapi_host", "0.0.0.0"), opts.get("webapi_port", 8000)),
        daemon=True,
    )
    api_thread.start()
    logger.info(f"System: Web API started on port {opts.get('webapi_port', 8000)}")

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        coordinator.tick,
        "interval",
        seconds=int(opts.get("thermostat_interval_seconds", 60)),
        id="coordinator_tick",
    )
    scheduler.add_job(
        coordinator.solar_tick,
        "interval",
        seconds=int(opts.get("solar_interval_seconds", 15)),
        id="solar_tick",
    )

    train_time = opts.get("full_retrain_time", "03:00")
    hh, mm = map(int, train_time.split(":"))

    scheduler.add_job(
        coordinator.perform_nightly_training,
        "cron",
        hour=hh,
        minute=mm,
        id="nightly_training",
    )

    scheduler.start()
    logger.info("System: Scheduler started. Engine running.")

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("System: Stopping scheduler and exiting...")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
