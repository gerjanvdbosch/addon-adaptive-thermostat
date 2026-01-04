import numpy as np
import logging

from context import Context
from client import HAClient
from config import Config
from collections import deque

logger = logging.getLogger(__name__)


class Collector:
    def __init__(self, client: HAClient, context: Context, config: Config):
        self.client = client
        self.context = context
        self.config = config

    def update_forecast(self):
        logger.info("Collector: Forecast update started")

    def update_sensors(self):
        self.context.current_pv = self.client.get_pv_power(self.config.sensor_pv)
        self.context.current_load = self.client.get_load_power(self.config.sensor_load)

        self.context.stable_pv = self._update_buffer(
            self.context.pv_buffer, self.context.current_pv
        )
        self.context.stable_load = self._update_buffer(
            self.context.load_buffer, self.context.current_load
        )

        self.context.hvac_mode = self.client.get_hvac_mode(self.config.sensor_hvac)

    def _update_buffer(self, buffer: deque, value: float) -> float:
        buffer.append(value)
        return float(np.median(buffer))

    def log_snapshot(self):
        pass

    def update_pv_status(self):
        now = self.context.now
        aggregation_minutes = 15
        slot_minute = (now.minute // aggregation_minutes) * aggregation_minutes
        slot_start = now.replace(minute=slot_minute, second=0, microsecond=0)

        # Als dit de allereerste sample is
        if self.context.current_slot_start is None:
            self.context.current_slot_start = slot_start

        # Als we een nieuw kwartier zijn binnengegaan
        if slot_start > self.context.current_slot_start:
            if self.context.slot_samples:
                avg_pv = float(np.mean(self.context.slot_samples))
                # Sla het gemiddelde op voor het AFGELOPEN kwartier
                # upsert_solar_record(self.current_slot_start, actual_pv_yield=avg_pv)
                logger.info(
                    f"Collector: Actual yield opgeslagen voor {self.context.current_slot_start.strftime('%H:%M')}: {avg_pv:.2f}kW"
                )

            self.context.slot_samples = []
            self.context.current_slot_start = slot_start

        self.context.slot_samples.append(self.context.current_pv)
