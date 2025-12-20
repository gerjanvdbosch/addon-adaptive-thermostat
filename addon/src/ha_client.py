import os
import requests
import logging
from utils import safe_float, safe_round

logger = logging.getLogger(__name__)


class HAClient:
    def __init__(self, opts):
        self.opts = opts or {}
        self.url = os.environ.get("SUPERVISOR_API", "http://supervisor/core/api")
        self.token = os.environ.get("SUPERVISOR_TOKEN")
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
            state = payload
            logger.debug("State %s fetched: %s", entity_id, state)
            return state
        except Exception as e:
            logger.exception("Error getting state %s: %s", entity_id, e)
            return None

    def _call_service(self, domain, service, data):
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

    def get_setpoint(self):
        return safe_float(self.get_state(self.opts.get("thermostat_entity")))

    def set_setpoint(self, value):
        try:
            setpoint = safe_round(float(value))
        except Exception:
            logger.exception("Invalid setpoint value provided: %s", value)
            return None
        try:
            entity = self.opts.get("thermostat_entity")
            service_data = {"entity_id": entity, "value": safe_round(float(value))}
            self._call_service("input_number", "set_value", service_data)
        except Exception:
            logger.exception("Failed to update setpoint: %s", entity)
            return None
        logger.info("Applied setpoint: %.1f", setpoint)
