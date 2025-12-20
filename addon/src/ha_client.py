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
            state = payload.get("state")
            logger.debug("State %s fetched: %s", entity_id, state)
            return state
        except Exception as e:
            logger.exception("Error getting state %s: %s", entity_id, e)
            return None

    def _set_state(self, entity_id, state, attributes=None):
        if attributes is None:
            attributes = {}

        url = f"{self.url}/states/{entity_id}"

        payload = {"state": state, "attributes": attributes}

        try:
            r = requests.post(url, json=payload, headers=self.headers)
            r.raise_for_status()
            logger.debug(
                f"State set for {entity_id}: {state} (attrs: {len(attributes)})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to set state for {entity_id}: {e}")
            return False

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
        setpoint = safe_round(float(value))
        entity = self.opts.get("thermostat_entity")
        service_data = {"entity_id": entity, "value": safe_round(float(value))}
        self._call_service("input_number", "set_value", service_data)
        logger.debug("Applied setpoint: %.1f", setpoint)

    def set_solar_prediction(self, value, attrs):
        entity = self.opts.get("solar_entity")
        service_data = {"entity_id": entity, "value": value}
        self._call_service("input_text", "set_value", service_data)
        logger.debug(f"Solar prediction updated: {value} with attrs {attrs}")
