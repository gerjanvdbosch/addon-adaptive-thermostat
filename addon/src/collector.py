import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from context import Context
from client import HAClient
from config import Config
from collections import deque
from weather import WeatherClient
from database import Database

logger = logging.getLogger(__name__)


class Collector:
    def __init__(
        self, client: HAClient, database: Database, context: Context, config: Config
    ):
        self.weather = WeatherClient(config)
        self.client = client
        self.database = database
        self.context = context
        self.config = config

    def update_forecast(self):
        solcast = self.client.get_forecast(self.config.sensor_solcast)

        now_local = pd.Timestamp.now(tz=datetime.now().astimezone().tzinfo)
        start_filter = now_local.replace(
            hour=0, minute=0, second=0, microsecond=0
        ).tz_convert("UTC")
        end_filter = start_filter + timedelta(days=1)
        full_index = pd.date_range(
            start=start_filter, periods=97, freq="15min", name="timestamp"
        )

        df = pd.DataFrame(solcast)
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)

        df_sol = (
            df.set_index("timestamp")
            .apply(pd.to_numeric, errors="coerce")
            .infer_objects(copy=False)
            .reindex(full_index)
            .interpolate(method="linear")
            .fillna(0)
            .reset_index()
        )

        df_om = self.weather.get_forecast()
        df_merged = pd.merge(df_sol, df_om, on="timestamp", how="left")

        df_today = (
            df_merged[
                (df_merged["timestamp"] >= start_filter)
                & (df_merged["timestamp"] <= end_filter)
            ]
            .copy()
            .sort_values("timestamp")
        )

        self.context.forecast_df = df_today
        self.database.save_forecast(df_today)

        logger.info("[Collector] Forecast updated")

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

        logger.info("[Collector] Sensors updated")

    def update_pv(self):
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
                self.database.update_pv_actual(
                    self.context.current_slot_start, yield_kw=avg_pv
                )
                logger.info(
                    f"[Collector] Actual yield opgeslagen voor {self.context.current_slot_start.strftime('%H:%M')}: {avg_pv:.2f}kW"
                )

            self.context.slot_samples = []
            self.context.current_slot_start = slot_start

        self.context.slot_samples.append(self.context.current_pv)

    def log_snapshot(self):
        pass

    def _update_buffer(self, buffer: deque, value: float) -> float:
        buffer.append(value)
        return float(np.median(buffer))
