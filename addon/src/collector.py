import logging
from datetime import datetime
import time
from typing import Optional, Dict, Any, List

from utils import (
    safe_float,
    round_half,
    cyclical_hour,
    cyclical_day,
    encode_wind,
    cyclical_month,
    month_to_season,
    day_or_night,
)
from db import insert_sample, insert_setpoint

logger = logging.getLogger(__name__)

FEATURE_ORDER: List[str] = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "season",
    "day_or_night",
    "current_setpoint",
    "current_temp",
    "temp_change",
    "min_temp_today",
    "max_temp_today",
    "min_temp_tomorrow",
    "max_temp_tomorrow",
    "solar_kwh_today",
    "solar_kwh_tomorrow",
    "solar_chance_today",
    "solar_chance_tomorrow",
    "wind_speed_today",
    "wind_speed_tomorrow",
    "wind_dir_today_sin",
    "wind_dir_today_cos",
    "wind_dir_tomorrow_sin",
    "wind_dir_tomorrow_cos",
    "outside_temp",
    "hvac_mode",
]


class Collector:
    def __init__(self, ha_client, opts: dict, impute_value: float = 0.0):
        self.ha = ha_client
        self.opts = opts or {}
        self.impute_value = impute_value
        # Require explicit sensor mapping in options; fail fast if not provided
        self.sensor_map = self.opts.get("sensors")
        if not isinstance(self.sensor_map, dict) or not self.sensor_map:
            raise RuntimeError(
                "Sensor mapping missing in add-on config (opts['sensors']). "
                "Please configure sensor entity IDs in the add-on options."
            )

    def read_sensors(self) -> Dict[str, Optional[float]]:
        """
        Read current setpoint/temp from HA client and then each mapped sensor.
        Returns a dict with raw numeric values or None.
        """
        current_setpoint, current_temp, hvac_mode = self.ha.get_setpoint()
        data: Dict[str, Optional[float]] = {
            "current_setpoint": current_setpoint,
            "current_temp": current_temp,
            "hvac_mode": hvac_mode,
        }
        time.sleep(0.01)
        for feature_key, entity_id in self.sensor_map.items():
            data[feature_key] = None
            st = self.ha.get_state(entity_id)
            if not st:
                time.sleep(0.01)
                continue
            val = st.get("state")
            data[feature_key] = val
            time.sleep(0.01)
        return data

    def features_from_raw(
        self, sensor_dict: Dict[str, Any], timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Convert raw sensor dict into feature dictionary following FEATURE_ORDER keys.
        Defensive: uses safe_float and encoding helpers.
        """
        ts = timestamp or datetime.now()
        hx, hy = cyclical_hour(ts)
        dx, dy = cyclical_day(ts)
        mx, my = cyclical_month(ts)
        season_idx = month_to_season(ts)

        # these keys are optional in sensor_dict; keep names consistent with what you write into DB
        wind_dir_today = sensor_dict.get("wind_direction_today")
        wind_dir_tomorrow = sensor_dict.get("wind_direction_tomorrow")

        wtd_sin, wtd_cos = encode_wind(wind_dir_today)
        wtm_sin, wtm_cos = encode_wind(wind_dir_tomorrow)

        hvac_mode = {"off": 0, "heat": 1, "cool": 2}.get(
            sensor_dict.get("hvac_mode"), 0
        )

        return {
            "hour_sin": hx,
            "hour_cos": hy,
            "day_sin": dx,
            "day_cos": dy,
            "month_sin": mx,
            "month_cos": my,
            "season": float(season_idx),
            "day_or_night": float(day_or_night(ts)),
            "current_setpoint": safe_float(sensor_dict.get("current_setpoint")),
            "current_temp": safe_float(sensor_dict.get("current_temp")),
            "temp_change": safe_float(sensor_dict.get("temp_change")),
            "min_temp_today": safe_float(sensor_dict.get("min_temp_today")),
            "max_temp_today": safe_float(sensor_dict.get("max_temp_today")),
            "min_temp_tomorrow": safe_float(sensor_dict.get("min_temp_tomorrow")),
            "max_temp_tomorrow": safe_float(sensor_dict.get("max_temp_tomorrow")),
            "solar_kwh_today": round(
                round_half(safe_float(sensor_dict.get("solar_kwh_today"))), 1
            ),
            "solar_kwh_tomorrow": round(
                round_half(safe_float(sensor_dict.get("solar_kwh_tomorrow"))), 1
            ),
            "solar_chance_today": safe_float(sensor_dict.get("solar_chance_today")),
            "solar_chance_tomorrow": safe_float(
                sensor_dict.get("solar_chance_tomorrow")
            ),
            "wind_speed_today": safe_float(sensor_dict.get("wind_speed_today")),
            "wind_speed_tomorrow": safe_float(sensor_dict.get("wind_speed_tomorrow")),
            "wind_dir_today_sin": wtd_sin,
            "wind_dir_today_cos": wtd_cos,
            "wind_dir_tomorrow_sin": wtm_sin,
            "wind_dir_tomorrow_cos": wtm_cos,
            "outside_temp": safe_float(sensor_dict.get("outside_temp")),
            "hvac_mode": safe_float(hvac_mode),
        }

    def get_features(self, ts: datetime):
        """
        Public method used by inferencer/trainer to obtain a features dict for timestamp ts.
        """
        try:
            sensors = self.read_sensors()
            features = self.features_from_raw(sensors, timestamp=ts)
            return features
        except Exception:
            logger.exception("Unexpected error while reading sensors")
            return None

    def sample_and_store(self):
        ts = datetime.now()
        try:
            sensors = self.read_sensors()
            features = self.features_from_raw(sensors, timestamp=ts)
            insert_sample(features)
            insert_setpoint(features)
            logger.info(
                "Sample stored: current_setpoint=%s current_temp=%s",
                sensors.get("current_setpoint"),
                sensors.get("current_temp"),
            )
        except Exception:
            logger.exception(
                "Unexpected error while reading sensors; skipping this sample"
            )
