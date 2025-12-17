import os
import threading
import time
import logging
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from collector import Collector
from trainer import Trainer
from trainer_delta import TrainerDelta

from solar import SolarController
from inferencer import Inferencer
from inferencer_delta import InferencerDelta
from ha_client import HAClient
from config import load_options

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("apscheduler").setLevel(logging.WARNING)


def start_api(host: str, port: int):
    uvicorn.run("webapi:app", host=host, port=port, log_level="info")


def main():
    opts = load_options()
    ha = HAClient(opts)
    collector = Collector(ha, opts)
    trainer = Trainer(ha, opts)
    trainer_delta = TrainerDelta(ha, opts)
    inferencer = Inferencer(ha, collector, opts)
    inferencer_delta = InferencerDelta(ha, collector, opts)

    # --- SOLAR CONTROLLER INITIALISEREN ---
    logger.info("Initializing Solar Brain...")
    solar_ctrl = SolarController(ha, opts)

    # Probeer direct data op te halen (voorkomt wachten op eerste tick)
    try:
        solar_ctrl.update_solcast()
    except Exception as e:
        logger.warning(f"Initial Solar update failed: {e}")
    # --------------------------------------

    api_thread = threading.Thread(
        target=start_api, args=(opts["webapi_host"], opts["webapi_port"]), daemon=True
    )
    api_thread.start()
    logger.info(
        "Started internal web API on %s:%s", opts["webapi_host"], opts["webapi_port"]
    )

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        collector.sample_and_store,
        "interval",
        seconds=opts["sample_interval_seconds"],
        id="collector",
    )

    hh, mm = map(int, opts["full_retrain_time"].split(":"))

    scheduler.add_job(trainer.train_job, "cron", hour=hh, minute=mm, id="full_retrain")
    scheduler.add_job(
        trainer_delta.train_job, "cron", hour=hh, minute=mm, id="full_retrain_delta"
    )
    scheduler.add_job(
        inferencer.inference_job,
        "interval",
        seconds=opts["inferencer_interval_seconds"],
        id="inference",
    )
    scheduler.add_job(
        inferencer_delta.run_cycle,
        "interval",
        seconds=opts["inferencer_interval_seconds"],
        id="inference_delta",
    )

    solar_interval = opts.get("solar_interval_seconds", 15)

    # 1. Main Tick: Gebruikt nu dezelfde interval als de inferencer
    scheduler.add_job(
        solar_ctrl.tick, "interval", seconds=solar_interval, id="solar_tick"
    )

    # 2. Retrain: Dagelijks
    scheduler.add_job(
        solar_ctrl.train_model, "cron", hour=hh, minute=mm, id="solar_retrain"
    )

    scheduler.start()
    logger.info("Adaptive Thermostat add-on started")
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping scheduler and exiting")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
