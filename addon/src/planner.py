import logging

from dataclasses import dataclass
from context import Context
from config import Config
from forecaster import SolarForecaster
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    action: str = None
    dhw_start_time: str = None
    heating_start_time: str = None


class Planner:
    def __init__(self, context: Context, config: Config):
        self.forecaster = SolarForecaster(config, context)
        self.context = context

    def create_plan(self):
        now = self.context.now
        status, forecast = self.forecaster.analyze(now, self.context.stable_load)

        self.context.forecast = forecast
        logger.info(f"[Planner] Status {status}")

        if forecast is not None:
            if forecast.planned_start is None or forecast.planned_start < now:
                forecast.planned_start = datetime.now(timezone.utc).replace(
                    hour=16, minute=0, second=0, microsecond=0
                )


#             logger.info(f"[Planner] Reason {forecast.reason}")
#             logger.info(f"[Planner] PV now {forecast.actual_pv}kW")
#             logger.info(f"[Planner] Load now {forecast.load_now}kW")
#             logger.info(f"[Planner] Forecast now {forecast.energy_now}kW")
#             logger.info(f"[Planner] Forecast best {forecast.energy_best}kW")
#             logger.info(f"[Planner] Opportunity cost {forecast.opportunity_cost}")
#             logger.info(f"[Planner] Confidence {forecast.confidence}")
#             logger.info(f"[Planner] Bias {forecast.current_bias}")
#             logger.info(f"[Planner] Planned start {forecast.planned_start}")

# Compressor freq gebruiken voor load / power inschatting

# Planner wanneer de verwarming aan moet als het x tijd warm moet zijn

# In zomer, beste DHW moment plannen
# In winter, beste verwarmingsmoment plannen en evt in piek/teruglevering DHW (verwarmen stoppen)
# DHW zo laat mogelijk voor deadline plannen

# Als het warm blijft tot x uur, geen verwarming nodig en lager zetten
