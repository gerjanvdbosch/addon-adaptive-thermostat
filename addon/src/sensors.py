import os
import requests

SUPERVISOR_URL = "http://supervisor/core/api"
TOKEN = os.getenv("SUPERVISOR_TOKEN")

def states(entity_id):
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    url = f"{SUPERVISOR_URL}/states/{entity_id}"
    resp = requests.get(url, headers=headers, timeout=5)
    resp.raise_for_status()
    return resp.json()["state"]
