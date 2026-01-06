import os
import threading
import logging
import uvicorn

from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.blocking import BlockingScheduler

from config import Config
from context import Context
from collector import Collector
from client import HAClient
from planner import Planner
from database import Database
from dhw import DhwMachine
from climate import ClimateMachine
from webapi import api

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

logging.getLogger("apscheduler").setLevel(logging.WARNING)


class Coordinator:
    def __init__(self, context: Context, config: Config, collector: Collector):
        self.planner = Planner(context, config)
        self.dhw_machine = DhwMachine(context)
        self.climate_machine = ClimateMachine(context)
        self.context = context
        self.config = config
        self.collector = collector

    def tick(self):
        self.collector.update_sensors()

        self.context.now = datetime.now(timezone.utc)

        plan = self.planner.create_plan()

        self.dhw_machine.process(plan)
        self.climate_machine.process(plan)

    def train(self):
        cutoff_date = self.context.now - timedelta(days=730)
        history = self.collector.database.get_forecast_history(cutoff_date)
        self.planner.forecaster.model.train(history, system_max=self.config.pv_max_kw)

    def start_api(self):
        api.state.coordinator = self
        uvicorn.run(
            api,
            host=self.config.webapi_host,
            port=self.config.webapi_port,
            log_level="warning",
        )


if __name__ == "__main__":
    logger.info("[System] Starting...")

    scheduler = BlockingScheduler()

    try:
        client = HAClient()
        config = Config.load(client)
        context = Context(now=datetime.now(timezone.utc))
        database = Database(config)
        collector = Collector(client, database, context, config)
        coordinator = Coordinator(context, config, collector)

        webapi = threading.Thread(target=coordinator.start_api, daemon=True)
        webapi.start()

        logger.info("[System] API server started.")

        scheduler.add_job(collector.update_forecast, "interval", minutes=15)
        scheduler.add_job(collector.update_pv, "interval", seconds=15)

        scheduler.add_job(coordinator.tick, "interval", minutes=1)
        scheduler.add_job(coordinator.train, "cron", hour=21, minute=46)

        logger.info("[System] Engine running.")

        collector.update_forecast()
        coordinator.tick()

        scheduler.start()

    except (KeyboardInterrupt, SystemExit):
        logger.info("[System] Stopping and exiting...")
        scheduler.shutdown()
