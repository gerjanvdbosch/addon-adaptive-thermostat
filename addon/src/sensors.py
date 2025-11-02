import os
import json
import requests

HA_URL = os.environ.get("HA_URL", "http://supervisor/core/api")
HA_TOKEN = os.environ.get("HA_TOKEN")
HEADERS = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

def states(id):
    resp = requests.get(f"{HA_URL}/states/{id}", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()
