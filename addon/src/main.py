import os
import threading
import time
import logging
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from collector import Collector
from trainer import Trainer
from trainer2 import Trainer2
from inferencer import Inferencer
from inferencer2 import Inferencer2
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
    trainer2 = Trainer2(ha, opts)
    inferencer = Inferencer(ha, collector, opts)
    inferencer2 = Inferencer2(ha, collector, opts)

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
    scheduler.add_job(
        trainer.partial_fit_job,
        "interval",
        seconds=opts["partial_fit_interval_seconds"],
        id="partial_fit",
    )

    hh, mm = map(int, opts["full_retrain_time"].split(":"))

    scheduler.add_job(
        trainer.full_retrain_job, "cron", hour=hh, minute=mm, id="full_retrain"
    )
    scheduler.add_job(
        trainer2.train_job, "cron", hour=hh, minute=mm, id="full_retrain2"
    )
    scheduler.add_job(
        inferencer.inference_job,
        "interval",
        seconds=opts["inferencer_interval_seconds"],
        id="inference",
    )
    scheduler.add_job(
        inferencer2.inference_job,
        "interval",
        seconds=opts["inferencer_interval_seconds"],
        id="inference2",
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
