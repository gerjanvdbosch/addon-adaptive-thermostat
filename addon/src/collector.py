import logging
import datetime
from db import insert_sample
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

DEFAULT_SENSORS = {
    "current_temp": "sensor.huidige_temp",
    "temp_change": "sensor.temp_verandering",
    "min_temp_today": "sensor.min_temp_vandaag",
    "max_temp_today": "sensor.max_temp_vandaag",
    "min_temp_tomorrow": "sensor.min_temp_morgen",
    "max_temp_tomorrow": "sensor.max_temp_morgen",
    "solar_kwh_today": "sensor.zon_kwh_vandaag",
    "solar_chance_today": "sensor.zon_kans_vandaag",
    "current_setpoint": "sensor.huidige_setpoint",
    "wind_speed_today": "sensor.windkracht_vandaag",
    "wind_speed_tomorrow": "sensor.windkracht_morgen",
    "wind_direction_today": "sensor.windrichting_vandaag",
    "outside_temp": "sensor.buiten_temp"
}


class Collector:
    def __init__(self, ha_client, opts):
        self.ha = ha_client
        self.opts = opts or {}
        self.sensor_map = self.opts.get("sensors") or DEFAULT_SENSORS
        self.fe = FeatureExtractor()

    def read_sensors(self):
        data = {}
        for feature_key, entity_id in self.sensor_map.items():
            data[feature_key] = None
            st = self.ha.get_state(entity_id)
            if not st:
                continue
            val = st.get("state")
            attrs = st.get("attributes", {})
            numeric = attrs.get("value") if isinstance(attrs.get("value"), (int, float)) else None
            try:
                data[feature_key] = float(val)
            except Exception:
                try:
                    data[feature_key] = float(numeric) if numeric is not None else None
                except Exception:
                    data[feature_key] = None
        return data

    def sample_and_store(self):
        ts = datetime.datetime.utcnow()
        sensors = self.read_sensors()
        features = self.fe.features_from_raw(sensors, timestamp=ts)
        insert_sample({"timestamp": ts.isoformat(), "sensors": sensors, "features": features})
        logger.info("Sample stored at %s", ts.isoformat())
