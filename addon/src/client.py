import os
import requests
import logging

from utils import safe_float, to_kw
from context import HvacMode

logger = logging.getLogger(__name__)


class HAClient:
    def __init__(self):
        self.url = os.environ.get("SUPERVISOR_API", "http://supervisor/core/api")
        self.token = os.environ.get("SUPERVISOR_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_location(self, entity_id):
        attributes = self._get_attributes(entity_id)
        if not attributes:
            return None, None

        latitude = safe_float(attributes.get("latitude"))
        longitude = safe_float(attributes.get("longitude"))
        return latitude, longitude

    def get_pv_power(self, entity_id):
        return to_kw(self._get_state(entity_id))

    def get_load_power(self, entity_id):
        return to_kw(self._get_state(entity_id))

    def get_hvac_mode(self, entity_id):
        hvac = {
            "Uit": HvacMode.OFF,
            "SWW": HvacMode.DHW,
            "Verwarmen": HvacMode.HEATING,
            "Koelen": HvacMode.COOLING,
            "Legionellapreventie": HvacMode.LEGIONELLA_PREVENTION,
            "Vorstbescherming": HvacMode.FROST_PROTECTION,
        }.get(self._get_state(entity_id))
        return HvacMode(hvac)

    def _get_state(self, entity_id):
        return self._get_payload(entity_id).get("state")

    def _get_attributes(self, entity_id):
        return self._get_payload(entity_id).get("attributes", {})

    def _get_payload(self, entity_id):
        try:
            r = requests.get(
                f"{self.url}/states/{entity_id}", headers=self.headers, timeout=10
            )
            r.raise_for_status()
            payload = r.json()
            logger.debug("Payload %s fetched", entity_id)
            return payload
        except Exception as e:
            logger.exception("Error getting state %s: %s", entity_id, e)
            return None

    def _set_state(self, entity_id, state, attributes=None, friendly_name=None):
        if attributes is None:
            attributes = {}

        if friendly_name:
            attributes["friendly_name"] = friendly_name

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
            logger.error(
                f"Failed to set state for {entity_id} with payload {payload}: {e}"
            )
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
