from dataclasses import dataclass
from forecaster import SolarForecaster
from context import Context


@dataclass
class Plan:
    action: str  # "START_DHW", "START_HEATING", "IDLE"
    dhw_start_time: str = None
    heating_start_time: str = None


class Planner:
    def __init__(self, forecaster: SolarForecaster, context: Context):
        self.forecast = forecaster
        self.context = context

    def create_plan(self):
        now = self.context.now
        strategy = self.forecaster.analyze(now, self.context.stable_load)

        pass
