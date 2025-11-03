import os
import requests
import logging

logger = logging.getLogger(__name__)

class HAClient:
    def __init__(self, url=None, token=None):
        self.url = url or os.environ.get("SUPERVISOR_API", "http://supervisor/core/api")
        self.token = token or os.environ.get("SUPERVISOR_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def get_state(self, entity_id):
        try:
            r = requests.get(f"{self.url}/states/{entity_id}", headers=self.headers, timeout=10)
            r.raise_for_status()
            payload = r.json()
            logger.debug("State %s fetched: %s", entity_id, payload.get("state"))
            return payload
        except Exception as e:
            logger.exception("Error getting state %s: %s", entity_id, e)
            return None

    def call_service(self, domain, service, data):
        try:
            r = requests.post(f"{self.url}/services/{domain}/{service}", json=data, headers=self.headers, timeout=10)
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
