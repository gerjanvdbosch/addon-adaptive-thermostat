import os
import threading
import logging
import uvicorn

from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

from config import Config
from context import Context
from collector import Collector
from client import HAClient
from forecaster import SolarForecaster
from planner import Planner
from dhw import DhwMachine
from climate import ClimateMachine

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("apscheduler").setLevel(logging.WARNING)


class Coordinator:
    def __init__(self):
        self.client = HAClient()
        self.config = Config.load(self.client)
        self.context = Context(now=datetime.now())

        self.dhw_machine = DhwMachine(self.context)
        self.climate_machine = ClimateMachine(self.context)

        self.forecaster = SolarForecaster(self.config, self.context)
        self.planner = Planner(self.forecaster, self.context)

        self.collector = Collector(self.client, self.context, self.config)

    def tick(self):
        self.context.now = datetime.now()

        plan = self.planner.create_plan()

        self.dhw_machine.process(plan)
        self.climate_machine.process(plan)

    def start_api(self):
        uvicorn.run(
            "api:api",
            host=self.config.webapi_host,
            port=self.config.webapi_port,
            log_level="warning",
        )


if __name__ == "__main__":
    logger.info("System: Starting...")

    scheduler = BlockingScheduler()

    try:
        coordinator = Coordinator()

        webapi = threading.Thread(target=coordinator.start_api, daemon=True)
        webapi.start()

        scheduler.add_job(coordinator.collector.tick, "interval", seconds=60)

        # Coordinator tick job: elke 60s, kleine startvertraging
        scheduler.add_job(
            coordinator.tick,
            "interval",
            seconds=60,
            start_date=f"{datetime.now().date()} 00:00:02",
        )

        logger.info("System: Engine running.")
        scheduler.start()

    except (KeyboardInterrupt, SystemExit):
        logger.info("System: Stopping and exiting...")
        scheduler.shutdown()
