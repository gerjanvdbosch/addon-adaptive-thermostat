import os
import threading
import time
import logging
import uvicorn

from apscheduler.schedulers.background import BackgroundScheduler
from config import load_options
from ha_client import HAClient
from collector import Collector
from climate_coordinator import ClimateCoordinator

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("apscheduler").setLevel(logging.WARNING)


def start_api(host: str, port: int):
    uvicorn.run("webapi:app", host=host, port=port, log_level="warning")


def main():
    logger.info("System: Starting...")

    opts = load_options()

    ha = HAClient(opts)
    collector = Collector(ha, opts)

    logger.info("System: Initializing...")
    coordinator = ClimateCoordinator(ha, collector, opts)

    api_thread = threading.Thread(
        target=start_api,
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
