import datetime
import math
import numpy as np

FEATURE_ORDER = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "current_temp", "temp_change",
    "min_temp_today", "max_temp_today",
    "min_temp_tomorrow", "max_temp_tomorrow",
    "solar_kwh_today", "solar_chance_today",
    "current_setpoint",
    "wind_speed_today", "wind_speed_tomorrow",
    "wind_sin", "wind_cos", "outside_temp"
]


class FeatureExtractor:
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
        try:
            return float(x)
        except Exception:
            return None

    def features_from_raw(self, sensor_dict: dict, timestamp: datetime.datetime = None) -> dict:
        ts = timestamp or datetime.datetime.utcnow()
        hx, hy = self._cyclical_hour(ts)
        dx, dy = self._cyclical_day(ts)
        wind_dir = sensor_dict.get("wind_direction_today")
        w_sin, w_cos = self._encode_wind(wind_dir)

        return {
            "hour_sin": hx,
            "hour_cos": hy,
            "day_sin": dx,
            "day_cos": dy,
            "current_temp": self._safe_float(sensor_dict.get("current_temp")),
            "temp_change": self._safe_float(sensor_dict.get("temp_change")),
            "min_temp_today": self._safe_float(sensor_dict.get("min_temp_today")),
            "max_temp_today": self._safe_float(sensor_dict.get("max_temp_today")),
            "min_temp_tomorrow": self._safe_float(sensor_dict.get("min_temp_tomorrow")),
            "max_temp_tomorrow": self._safe_float(sensor_dict.get("max_temp_tomorrow")),
            "solar_kwh_today": self._safe_float(sensor_dict.get("solar_kwh_today")),
            "solar_chance_today": self._safe_float(sensor_dict.get("solar_chance_today")),
            "current_setpoint": self._safe_float(sensor_dict.get("current_setpoint")),
            "wind_speed_today": self._safe_float(sensor_dict.get("wind_speed_today")),
            "wind_speed_tomorrow": self._safe_float(sensor_dict.get("wind_speed_tomorrow")),
            "wind_sin": w_sin,
            "wind_cos": w_cos,
            "outside_temp": self._safe_float(sensor_dict.get("outside_temp")),
        }

    def get_vector(self, feature_dict: dict):
        vec = []
        for k in FEATURE_ORDER:
            v = feature_dict.get(k, None)
            vec.append(self.impute_value if v is None else v)
        return np.array(vec, dtype=float)
