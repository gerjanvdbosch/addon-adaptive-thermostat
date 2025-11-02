import os
import requests

SUPERVISOR_API = "http://supervisor/core/api"
SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")

if not SUPERVISOR_TOKEN:
    raise RuntimeError("SUPERVISOR_TOKEN not found in environment.")

HEADERS = {
    "Authorization": f"Bearer {SUPERVISOR_TOKEN}",
    "Content-Type": "application/json",
}

def states(entity_id):
    url = f"{SUPERVISOR_API}/states/{entity_id}"
    resp = requests.get(url, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json()["state"]
