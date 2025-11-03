import logging
import datetime
from db import insert_sample
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class Collector:
    def __init__(self, ha_client, opts):
        self.ha = ha_client
        self.opts = opts or {}

        # Log the raw opts and sensors mapping for debugging
        logger.info("Collector sensors mapping raw value: %s", repr(self.opts.get("climate_entity")))
        logger.info("Collector sensors mapping raw value: %s", repr(self.opts.get("sensors")))
        
        # Require explicit sensor mapping in options; fail fast if not provided
        self.sensor_map = self.opts.get("sensors")
        if not isinstance(self.sensor_map, dict) or not self.sensor_map:
            raise RuntimeError("Sensor mapping missing in add-on config (opts['sensors']). Please configure sensor entity IDs in the add-on options.")
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
