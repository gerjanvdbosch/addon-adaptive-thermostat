import os
import threading
import time
import logging
import uvicorn

from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import Config
from context import Context
from collector import Collector

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("apscheduler").setLevel(logging.WARNING)


class Coordinator:
    def __init__(self, scheduler):
        self.config = Config.load()
        self.context = Context(self.config)

        collector = Collector(self.context)

        scheduler.add_job(collector.tick, "interval", seconds=60)
        # Voeg een kleine vertraging toe t.o.v. de sensor update
        scheduler.add_job(
            self.tick,
            trigger=IntervalTrigger(
                seconds=60, start_date=f"{datetime.now().date()} 00:00:05"
            ),
        )
        scheduler.start()

        api_thread = threading.Thread(
            target=self.start_api,
            args=(self.config.webapi, self.config.webapi_port),
            daemon=True,
        )
        api_thread.start()

        logger.info("System: Engine running.")

    def tick(self):
        self.context.now = datetime.now()

        plan = self.strategy.create_plan(self.ctx)

        self.dhw_machine.process(plan)
        self.climate_machine.process(plan)

    def start_api(host: str, port: int):
        uvicorn.run("webapi:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    logger.info("System: Starting...")

    scheduler = BackgroundScheduler()

    try:
        coordinator = Coordinator(scheduler)
        coordinator.tick()

        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("System: Stopping and exiting...")
        scheduler.shutdown()
