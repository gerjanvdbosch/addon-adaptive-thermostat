import logging

from dataclasses import dataclass
from forecaster import SolarForecaster
from config import Config
from context import Context

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str = None
    dhw_start_time: str = None
    heating_start_time: str = None


class Planner:
    def __init__(self, config: Config, context: Context):
        self.forecaster = SolarForecaster(config, context)
        self.context = context

    def create_plan(self):
        now = self.context.now
        forecast = self.forecaster.analyze(now, self.context.stable_load)

        if forecast is not None:
            logger.info(f"Planner: Action {forecast.action}")
            logger.info(f"Planner: Reason {forecast.reason}")
            logger.info(f"Planner: Load now {forecast.load_now}kW")
            logger.info(f"Planner: Energy now {forecast.energy_now}kW")
            logger.info(f"Planner: Energy best {forecast.energy_best}kW")
            logger.info(f"Planner: Opportunity cost {forecast.oppurtunity_cost}")
            logger.info(f"Planner: Confidence {forecast.confidence}")
            logger.info(f"Planner: Bias {forecast.bias}")
            logger.info(f"Planner: Planned start {forecast.planned_start}")
        else:
            logger.info("Planner: No forecast available")

        # Compressor freq gebruiken voor load / power inschatting

        # Planner wanneer de verwarming aan moet als het x tijd warm moet zijn

        # In zomer, beste DHW moment plannen
        # In winter, beste verwarmingsmoment plannen en evt in piek/teruglevering DHW (verwarmen stoppen)
        # DHW zo laat mogelijk voor deadline plannen

        # Als het warm blijft tot x uur, geen verwarming nodig en lager zetten
