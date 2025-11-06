import datetime
import math
import numpy as np

FEATURE_ORDER = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "current_setpoint", "current_temp", "temp_change",
    "min_temp_today", "max_temp_today",
    "min_temp_tomorrow", "max_temp_tomorrow",
    "solar_kwh_today", "solar_kwh_tomorrow",
    "solar_chance_today", "solar_chance_tomorrow",
    "wind_speed_today", "wind_speed_tomorrow",
    "wind_dir_today_sin", "wind_dir_today_cos",
    "wind_dir_tomorrow_sin", "wind_dir_tomorrow_cos",
    "outside_temp",
    "thermostat_demand", "operational_status", "prohibit_heat"
]


class FeatureExtractor:
    OP_STATUS_CATEGORIES = [
        "Uit",                  # index 0 -> represents off/unknown
        "SWW",                  # index 1
        "Legionellapreventie",  # index 2
        "Verwarmen",            # index 3
        "Koelen",               # index 4
        "Vorstbescherming"      # index 5
    ]
    OP_STATUS_MAP = {cat.lower(): idx for idx, cat in enumerate(OP_STATUS_CATEGORIES)}
    
    def __init__(self, impute_value=0.0):
        self.impute_value = impute_value

    def _cyclical_hour(self, ts):
        h = ts.hour + ts.minute / 60.0
        return math.sin(2 * math.pi * h / 24.0), math.cos(2 * math.pi * h / 24.0)

    def _cyclical_day(self, ts):
        d = ts.weekday()
        return math.sin(2 * math.pi * d / 7.0), math.cos(2 * math.pi * d / 7.0)

    def _encode_wind(self, degrees):
        if degrees is None:
            return 0.0, 0.0
        try:
            rad = math.radians(float(degrees) % 360)
            return math.sin(rad), math.cos(rad)
        except Exception:
            return 0.0, 0.0

    def _safe_float(self, x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    def _encode_binary_onoff(self, val):
        if val is None:
            return None
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        try:
            f = float(val)
            return 1.0 if f != 0.0 else 0.0
        except Exception:
            s = str(val).strip().lower()
            if s in ("on", "aan", "true", "1", "yes", "y"):
                return 1.0
            if s in ("off", "uit", "false", "0", "no", "n"):
                return 0.0
        return None

    def _encode_operational_status(self, status):
        if not isinstance(status, str):
            return 0  # Uit
        s = status.strip().lower()
        return self.OP_STATUS_MAP.get(s, 0)
    
    def features_from_raw(self, sensor_dict: dict, timestamp: datetime.datetime = None) -> dict:
        ts = timestamp or datetime.datetime.utcnow()
        hx, hy = self._cyclical_hour(ts)
        dx, dy = self._cyclical_day(ts)

        wind_dir_today = sensor_dict.get("wind_direction_today")
        wind_dir_tomorrow = sensor_dict.get("wind_direction_tomorrow")

        wtd_sin, wtd_cos = self._encode_wind(wind_dir_today)
        wtm_sin, wtm_cos = self._encode_wind(wind_dir_tomorrow)

        td_raw = sensor_dict.get("thermostat_demand")
        td = self._encode_binary_onoff(td_raw)

        op_raw = sensor_dict.get("operational_status")
        op_idx = self._encode_operational_status(op_raw)

        ph_raw = sensor_dict.get("prohibit_heat")
        ph = self._encode_binary_onoff(ph_raw)

        return {
            "hour_sin": hx,
            "hour_cos": hy,
            "day_sin": dx,
            "day_cos": dy,
            "current_setpoint": self._safe_float(sensor_dict.get("current_setpoint")),
            "current_temp": self._safe_float(sensor_dict.get("current_temp")),
            "temp_change": self._safe_float(sensor_dict.get("temp_change")),
            "min_temp_today": self._safe_float(sensor_dict.get("min_temp_today")),
            "max_temp_today": self._safe_float(sensor_dict.get("max_temp_today")),
            "min_temp_tomorrow": self._safe_float(sensor_dict.get("min_temp_tomorrow")),
            "max_temp_tomorrow": self._safe_float(sensor_dict.get("max_temp_tomorrow")),
            "solar_kwh_today": self._safe_float(sensor_dict.get("solar_kwh_today")),
            "solar_kwh_tomorrow": self._safe_float(sensor_dict.get("solar_kwh_tomorrow")),
            "solar_chance_today": self._safe_float(sensor_dict.get("solar_chance_today")),
            "solar_chance_tomorrow": self._safe_float(sensor_dict.get("solar_chance_tomorrow")),
            "wind_speed_today": self._safe_float(sensor_dict.get("wind_speed_today")),
            "wind_speed_tomorrow": self._safe_float(sensor_dict.get("wind_speed_tomorrow")),
            "wind_dir_today_sin": wtd_sin,
            "wind_dir_today_cos": wtd_cos,
            "wind_dir_tomorrow_sin": wtm_sin,
            "wind_dir_tomorrow_cos": wtm_cos,
            "outside_temp": self._safe_float(sensor_dict.get("outside_temp")),
            "thermostat_demand": td,
            "operational_status": float(op_idx),
            "prohibit_heat": ph
        }

    def get_vector(self, feature_dict: dict):
        vec = []
        for k in FEATURE_ORDER:
            v = feature_dict.get(k, None)
            vec.append(self.impute_value if v is None else v)
        return np.array(vec, dtype=float)
