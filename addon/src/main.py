import os
import threading
import time
import logging
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from collector import Collector
from trainer import Trainer
from trainer_delta import TrainerDelta

from inferencer import Inferencer
from inferencer_delta import InferencerDelta
from ha_client import HAClient
from config import load_options

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
