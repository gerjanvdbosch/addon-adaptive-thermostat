import os
import json

from client import HAClient
from dataclasses import dataclass, fields


@dataclass
class Config:
    sensor_pv: str
    sensor_load: str
    sensor_hvac: str

    sensor_solcast_today: str
    sensor_solcast_tomorrow: str

    temp_night: float = 19.0
    temp_morning: float = 19.5
    temp_day: float = 19.5
    temp_evening: float = 20.0

    latitude: float = 52.0
    longitude: float = 5.0

    pv_azimuth: float = 148.0
    pv_tilt: float = 50.0
    pv_max_kw: float = 4.0

    dhw_duration_hours: float = 1.0

    min_kwh_threshold: float = 0.1
    avg_baseload_kw: float = 0.15
    max_compressor_freq: int = 70

    webapi_host: str = "0.0.0.0"
    webapi_port: int = 8000

    @staticmethod
    def load(client: HAClient):
        raw = os.environ.get("ADDON_CONFIG", "{}")
        options = json.loads(raw)

        field_names = {f.name for f in fields(Config)}
        filtered = {k: v for k, v in options.items() if k in field_names}

        config = Config(**filtered)

        location = client.get_location(options.get("sensor_home", "zone.home"))
        if location != (None, None):
            config.latitude, config.longitude = location

        return config
