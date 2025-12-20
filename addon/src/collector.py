import logging
import time
from datetime import datetime

from utils import (
    safe_float,
    safe_round,
    cyclical_hour,
    cyclical_day,
    cyclical_doy,
    encode_wind,
    encode_binary_onoff,
)

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "hour_sin",  # Ochtend/Avond
    "hour_cos",
    "day_sin",  # Werkdag/Weekend
    "day_cos",
    "doy_sin",  # Seizoen (Zomer/Winter)
    "doy_cos",
    "home_presence",  # <--- ESSENTIEEL: Ben je thuis?
    "hvac_mode",  # Verwarmen/Koelen
    "heat_demand",
    "current_temp",  # Huidige binnen temp
    "current_setpoint",  # Waar staat hij nu op? (Startpunt voor delta)
    "temp_change",  # Hoe snel warmt het op/koelt het af?
    "outside_temp",  # Actuele buitentemperatuur (Cruciaal)
    "min_temp",  # <--- BEHOUDEN: Hoe koud was de nacht? (Koude muren)
    "max_temp",  # <--- BEHOUDEN: Hoe warm wordt de dag? (Algemeen beeld)
    "wind_speed",  # Windchill op de gevel
    "wind_dir_sin",
    "wind_dir_cos",
    "solar_kwh",  # Zonkracht op de ramen (gratis warmte)
]


class Collector:
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

    def read_sensors(self):
        data = {}
        for feature_key, entity_id in self.sensor_map.items():
            data[feature_key] = None
            st = self.ha.get_state(entity_id)
            time.sleep(0.05)
            if not st:
                continue
            val = st
            data[feature_key] = val
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

        wind_dir = sensor_dict.get("wind_dir")
        wtd_sin, wtd_cos = encode_wind(wind_dir)

        hvac_mode = {
            "Uit": 0,
            "Verwarmen": 1,
            "SWW": 2,
            "Koelen": 3,
            "Legionellapreventie": 4,
            "Vorstbescherming": 5,
        }.get(sensor_dict.get("hvac_mode"), 0)

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
            "home_presence": encode_binary_onoff(sensor_dict.get("home_presence")),
            "hvac_mode": safe_float(hvac_mode),
            "heat_demand": encode_binary_onoff(sensor_dict.get("heat_demand")),
            "current_temp": safe_float(sensor_dict.get("current_temp")),
            "current_setpoint": safe_float(raw_sp),
            "temp_change": safe_float(sensor_dict.get("temp_change")),
            "outside_temp": safe_float(sensor_dict.get("outside_temp")),
            "min_temp": safe_float(sensor_dict.get("min_temp")),
            "max_temp": safe_float(sensor_dict.get("max_temp")),
            "wind_speed": safe_float(sensor_dict.get("wind_speed")),
            "wind_dir_sin": wtd_sin,
            "wind_dir_cos": wtd_cos,
            "solar_kwh": safe_round(sensor_dict.get("solar_kwh")),
        }
