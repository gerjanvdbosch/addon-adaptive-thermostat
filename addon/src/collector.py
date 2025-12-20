import logging
import time
import numpy as np
from datetime import datetime

from utils import (
    safe_float,
    safe_round,
    cyclical_hour,
    cyclical_day,
    cyclical_doy,
    encode_wind,
)

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "hour_sin",  # Ochtend/Avond
    "hour_cos",
    "day_sin",  # Werkdag/Weekend
    "day_cos",
    "doy_sin",  # Seizoen (Zomer/Winter)
    "doy_cos",
    # --- STATUS & CONTEXT ---
    "home_presence",  # <--- ESSENTIEEL: Ben je thuis?
    "hvac_mode",  # Verwarmen/Koelen
    "heat_demand",
    "current_temp",  # Huidige binnen temp
    "temp_change",  # Hoe snel warmt het op/koelt het af?
    # --- BASISLIJN ---
    "current_setpoint",  # Waar staat hij nu op? (Startpunt voor delta)
    # --- WEER VANDAAG (Invloed op muren/ramen NU) ---
    "outside_temp",  # Actuele buitentemperatuur (Cruciaal)
    "min_temp",  # <--- BEHOUDEN: Hoe koud was de nacht? (Koude muren)
    "max_temp",  # <--- BEHOUDEN: Hoe warm wordt de dag? (Algemeen beeld)
    "wind_speed",  # Windchill op de gevel
    "wind_dir_sin",
    "wind_dir_cos",
    "solar_kwh",  # Zonkracht op de ramen (gratis warmte)
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

    def __init__(self, ha_client, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}
        # Require explicit sensor mapping in options
        self.sensor_map = self.opts.get("sensors")
        if not isinstance(self.sensor_map, dict) or not self.sensor_map:
            raise RuntimeError(
                "Sensor mapping missing in add-on config (opts['sensors']). "
                "Please configure sensor entity IDs."
            )

    def _encode_operational_status(self, status):
        if not isinstance(status, str):
            return 0  # Uit
        s = status.strip().lower()
        return self.OP_STATUS_MAP.get(s, 0)

    def read_sensors(self):
        """
        Read current setpoint/temp from HA client and then each mapped sensor.
        Returns a dict with raw numeric values or None.
        """
        current_setpoint, current_temp, hvac_mode = self.ha.get_setpoint()
        data = {
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

    def features_from_raw(self, sensor_dict, timestamp=None, override_setpoint=None):
        """
        Convert raw sensor dict into feature dictionary following FEATURE_ORDER keys.
        Defensive: uses safe_float and encoding helpers.
        """
        ts = timestamp or datetime.now()

        # Tijd features
        hx, hy = cyclical_hour(ts)
        dx, dy = cyclical_day(ts)  # Day of Week
        doy_x, doy_y = cyclical_doy(ts)  # Day of Year (Seizoen)

        # these keys are optional in sensor_dict; keep names consistent with what you write into DB
        wind_dir_today = sensor_dict.get("wind_direction_today")
        wind_dir_tomorrow = sensor_dict.get("wind_direction_tomorrow")

        wtd_sin, wtd_cos = encode_wind(wind_dir_today)
        wtm_sin, wtm_cos = encode_wind(wind_dir_tomorrow)

        hvac_mode = {"off": 0, "heat": 1, "cool": 2}.get(
            sensor_dict.get("hvac_mode"), 0
        )

        if override_setpoint is not None:
            raw_sp = override_setpoint
        else:
            raw_sp = sensor_dict.get("current_setpoint")

        return {
            "hour_sin": hx,
            "hour_cos": hy,
            "day_sin": dx,
            "day_cos": dy,
            "doy_sin": doy_x,  # Jaarritme (Zomer vs Winter)
            "doy_cos": doy_y,
            "current_setpoint": safe_float(raw_sp),
            "current_temp": safe_float(sensor_dict.get("current_temp")),
            "temp_change": safe_float(sensor_dict.get("temp_change")),
            "min_temp_today": safe_float(sensor_dict.get("min_temp_today")),
            "max_temp_today": safe_float(sensor_dict.get("max_temp_today")),
            "min_temp_tomorrow": safe_float(sensor_dict.get("min_temp_tomorrow")),
            "max_temp_tomorrow": safe_float(sensor_dict.get("max_temp_tomorrow")),
            "solar_kwh_today": safe_round(sensor_dict.get("solar_kwh_today")),
            "solar_kwh_tomorrow": safe_round(sensor_dict.get("solar_kwh_tomorrow")),
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

    def features_to_vector(self, features: dict):
        """Converts feature dict to list in correct order for model."""
        vec = []
        for k in FEATURE_ORDER:
            # We halen de waarde op. Als de key niet bestaat, is het None.
            val = features.get(k)

            # Feature specifieke checks (optioneel), maar safe_float kan None returnen
            if val is None:
                vec.append(np.nan)
            else:
                try:
                    vec.append(float(val))
                except (ValueError, TypeError):
                    vec.append(np.nan)

        return np.array([vec], dtype=float)  # Return as 2D numpy array [1, n_features]
