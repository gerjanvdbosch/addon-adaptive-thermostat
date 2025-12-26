import logging
import time
from datetime import datetime

from utils import (
    safe_float,
    cyclical_hour,
    cyclical_day,
    cyclical_doy,
    safe_bool_to_float,
    encode_wind,
)

logger = logging.getLogger(__name__)


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
        dx, dy = cyclical_day(ts)
        doy_x, doy_y = cyclical_doy(ts)
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
            raw_sp = self.ha.get_setpoint()

        return {
            "hour_sin": hx,
            "hour_cos": hy,
            "day_sin": dx,
            "day_cos": dy,
            "doy_sin": doy_x,
            "doy_cos": doy_y,
            "home_presence": safe_bool_to_float(sensor_dict.get("home_presence")),
            "hvac_mode": safe_float(hvac_mode),
            "heat_demand": safe_bool_to_float(sensor_dict.get("heat_demand")),
            "current_temp": safe_float(sensor_dict.get("current_temp")),
            "current_setpoint": safe_float(raw_sp),
            "temp_change": safe_float(sensor_dict.get("temp_change")),
            "outside_temp": safe_float(sensor_dict.get("outside_temp")),
            "min_temp": safe_float(sensor_dict.get("min_temp")),
            "max_temp": safe_float(sensor_dict.get("max_temp")),
            "solar_kwh": safe_float(sensor_dict.get("solar_kwh")),
            "wind_speed": safe_float(sensor_dict.get("wind_speed")),
            "wind_dir_sin": wtd_sin,
            "wind_dir_cos": wtd_cos,
            "pv_power": safe_float(sensor_dict.get("pv_power")),
            "supply_temp": safe_float(sensor_dict.get("supply_temp")),
            "dhw_temp": safe_float(sensor_dict.get("dhw_temp")),
            "dhw_temp2": safe_float(sensor_dict.get("dhw_temp2")),
        }
