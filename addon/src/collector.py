import logging
import datetime
import time
from db import insert_sample
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class Collector:
    def __init__(self, ha_client, opts):
        self.ha = ha_client
        self.opts = opts or {}
        # Require explicit sensor mapping in options; fail fast if not provided
        self.sensor_map = self.opts.get("sensors")
        if not isinstance(self.sensor_map, dict) or not self.sensor_map:
            raise RuntimeError("Sensor mapping missing in add-on config (opts['sensors']). Please configure sensor entity IDs in the add-on options.")
        self.fe = FeatureExtractor()

    def _safe_float(self, x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    def read_sensors(self):
        data = {}
        current_temp = None
        current_setpoint = None
        climate = self.ha.get_state(self.opts.get("climate_entity", "climate.woonkamer"))
        if climate:
            attrs = climate.get("attributes", {})
            current_temp = self._safe_float((attrs.get("current_temperature"))
            if self.opts.get("shadow_mode"):
                shadow = self.ha.get_state(self.opts.get("shadow_setpoint"))
                current_setpoint = self._safe_float((shadow.get("state"))
            else:
                current_setpoint = self._safe_float((attrs.get("temperature"))
        if current_temp is None:
            raise RuntimeError("Failed to read current_temp.")
        if current_setpoint is None:
            raise RuntimeError("Failed to read current_setpoint.")
        data["current_temp"] = current_temp
        data["current_setpoint"] = current_setpoint
        time.sleep(0.01)
        for feature_key, entity_id in self.sensor_map.items():
            data[feature_key] = None
            st = self.ha.get_state(entity_id)
            if not st:
                time.sleep(0.01)
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
            time.sleep(0.01)
        return data

    def sample_and_store(self):
        ts = datetime.datetime.utcnow()
        try:
            sensors = self.read_sensors()
            features = self.fe.features_from_raw(sensors, timestamp=ts)
            insert_sample({"timestamp": ts.isoformat(), "sensors": sensors, "features": features})
            logger.info("Sample stored: current_setpoint=%.1f current_temp=%.1f",
                        sensors.get("current_setpoint"), sensors.get("current_temp"))
        except Exception:
            logger.exception("Unexpected error while reading sensors; skipping this sample")
