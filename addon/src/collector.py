import logging
import datetime
import time
from typing import Optional, Dict, Any, List

import numpy as np

from utils import (
    safe_float,
    round_half,
    cyclical_hour,
    cyclical_day,
    encode_wind,
    encode_binary_onoff,
)
from db import insert_sample

logger = logging.getLogger(__name__)

FEATURE_ORDER: List[str] = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
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
    "thermostat_demand",
    "operational_status",
    "prohibit_heat",
]


class Collector:
    OP_STATUS_CATEGORIES = [
        "Uit",  # index 0 -> represents off/unknown
        "SWW",  # index 1
        "Legionellapreventie",  # index 2
        "Verwarmen",  # index 3
        "Koelen",  # index 4
        "Vorstbescherming",  # index 5
    ]
    OP_STATUS_MAP = {cat.lower(): idx for idx, cat in enumerate(OP_STATUS_CATEGORIES)}

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
        current_setpoint, current_temp = self.ha.get_setpoint()
        data: Dict[str, Optional[float]] = {
            "current_setpoint": current_setpoint,
            "current_temp": current_temp,
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

    def _encode_operational_status(self, status: Optional[str]) -> int:
        if not isinstance(status, str):
            return 0  # Uit
        s = status.strip().lower()
        return self.OP_STATUS_MAP.get(s, 0)

    def features_from_raw(
        self, sensor_dict: Dict[str, Any], timestamp: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Convert raw sensor dict into feature dictionary following FEATURE_ORDER keys.
        Defensive: uses safe_float and encoding helpers.
        """
        ts = timestamp or datetime.datetime.utcnow()
        hx, hy = cyclical_hour(ts)
        dx, dy = cyclical_day(ts)

        # these keys are optional in sensor_dict; keep names consistent with what you write into DB
        wind_dir_today = sensor_dict.get("wind_direction_today")
        wind_dir_tomorrow = sensor_dict.get("wind_direction_tomorrow")

        wtd_sin, wtd_cos = encode_wind(wind_dir_today)
        wtm_sin, wtm_cos = encode_wind(wind_dir_tomorrow)

        td_raw = sensor_dict.get("thermostat_demand")
        td = encode_binary_onoff(td_raw)

        op_raw = sensor_dict.get("operational_status")
        op_idx = self._encode_operational_status(op_raw)

        ph_raw = sensor_dict.get("prohibit_heat")
        ph = encode_binary_onoff(ph_raw)

        return {
            "hour_sin": hx,
            "hour_cos": hy,
            "day_sin": dx,
            "day_cos": dy,
            "current_setpoint": safe_float(sensor_dict.get("current_setpoint")),
            "current_temp": safe_float(sensor_dict.get("current_temp")),
            "temp_change": safe_float(sensor_dict.get("temp_change")),
            "min_temp_today": safe_float(sensor_dict.get("min_temp_today")),
            "max_temp_today": safe_float(sensor_dict.get("max_temp_today")),
            "min_temp_tomorrow": safe_float(sensor_dict.get("min_temp_tomorrow")),
            "max_temp_tomorrow": safe_float(sensor_dict.get("max_temp_tomorrow")),
            "solar_kwh_today": round(round_half(safe_float(sensor_dict.get("solar_kwh_today"))), 1),
            "solar_kwh_tomorrow": round(round_half(safe_float(sensor_dict.get("solar_kwh_tomorrow"))), 1),
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
            "thermostat_demand": td,
            "operational_status": float(op_idx),
            "prohibit_heat": ph,
        }

    def get_features(self, ts: datetime.datetime) -> Optional[Dict[str, Any]]:
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
        ts = datetime.datetime.utcnow()
        try:
            sensors = self.read_sensors()
            features = self.features_from_raw(sensors, timestamp=ts)
            insert_sample({"timestamp": ts.isoformat(), "features": features})
            logger.info(
                "Sample stored: current_setpoint=%s current_temp=%s",
                sensors.get("current_setpoint"),
                sensors.get("current_temp"),
            )
        except Exception:
            logger.exception(
                "Unexpected error while reading sensors; skipping this sample"
            )

    def get_vector(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert a feature dict into a numpy vector following FEATURE_ORDER.
        Missing values are imputed with self.impute_value.
        """
        vec = []
        for k in FEATURE_ORDER:
            v = feature_dict.get(k, None)
            vec.append(self.impute_value if v is None else v)
        return np.array(vec, dtype=float)
