import logging

from dataclasses import dataclass
from forecaster import SolarForecaster
from context import Context

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str = None
    dhw_start_time: str = None
    heating_start_time: str = None


class Planner:
    def __init__(self, forecaster: SolarForecaster, context: Context):
        self.forecaster = forecaster
        self.context = context

    def create_plan(self):
        now = self.context.now
        strategy = self.forecaster.analyze(now, self.context.stable_load)

        logger.info(f"Planner: Strategy {strategy}")

        # Compressor freq gebruiken voor load / power inschatting

        # Planner wanneer de verwarming aan moet als het x tijd warm moet zijn

        # In zomer, beste DHW moment plannen
        # In winter, beste verwarmingsmoment plannen en evt in piek/teruglevering DHW (verwarmen stoppen)
        # DHW zo laat mogelijk voor deadline plannen

        # Als het warm blijft tot x uur, geen verwarming nodig en lager zetten
