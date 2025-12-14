import os
import requests
import logging
from utils import round_half, safe_float, safe_round

logger = logging.getLogger(__name__)


class HAClient:
    def __init__(self, opts, url=None, token=None):
        self.opts = opts or {}
        self.url = url or os.environ.get("SUPERVISOR_API", "http://supervisor/core/api")
        self.token = token or os.environ.get("SUPERVISOR_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_state(self, entity_id):
        try:
            r = requests.get(
                f"{self.url}/states/{entity_id}", headers=self.headers, timeout=10
            )
            r.raise_for_status()
            payload = r.json()
            logger.debug("State %s fetched: %s", entity_id, payload.get("state"))
            return payload
        except Exception as e:
            logger.exception("Error getting state %s: %s", entity_id, e)
            return None

    def call_service(self, domain, service, data):
        try:
            r = requests.post(
                f"{self.url}/services/{domain}/{service}",
                json=data,
                headers=self.headers,
                timeout=10,
            )
            r.raise_for_status()
            return r.json() if r.text else {}
        except Exception as e:
            logger.exception("Error calling service %s.%s: %s", domain, service, e)
            return None

    def list_states(self):
        try:
            r = requests.get(f"{self.url}/states", headers=self.headers, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.exception("Error listing states: %s", e)
            return []

    def get_setpoint(self):
        current_setpoint = None
        current_temp = None
        hvac_mode = None
        climate = self.get_state(self.opts.get("climate_entity"))
        if climate:
            attrs = climate.get("attributes", {})
            current_temp = safe_float(attrs.get("current_temperature"))
            current_setpoint = safe_float(attrs.get("temperature"))
            hvac_mode = climate.get("state").strip().lower()
        if current_setpoint is None:
            raise RuntimeError("Failed to read current setpoint")
        if current_temp is None:
            raise RuntimeError("Failed to read current temperature")
        if hvac_mode is None:
            raise RuntimeError("Failed to read HVAC mode")
        return current_setpoint, current_temp, hvac_mode

    def get_shadow_setpoint(self):
        return safe_float(self.get_state(self.opts.get("shadow_setpoint")))

    def set_setpoint(self, value):
        try:
            setpoint = safe_round(round_half(float(value)))
        except Exception:
            logger.exception("Invalid setpoint value provided: %s", value)
            return None
        if self.opts.get("shadow_mode"):
            try:
                shadow = self.opts.get("shadow_setpoint")
                service_data = {"entity_id": shadow, "value": setpoint}
                self.call_service("input_number", "set_value", service_data)
            except Exception:
                logger.exception("Failed to update shadow setpoint: %s", shadow)
                return None
        else:
            return None
            climate = self.opts.get("climate_entity")
            service_data = {"entity_id": climate, "temperature": setpoint}
            self.call_service("climate", "set_temperature", service_data)
        logger.debug("Applied setpoint: %.1f", setpoint)
