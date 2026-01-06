from client import HAClient
from dataclasses import dataclass


@dataclass
class Config:
    temp_night: float = 19.0
    temp_morning: float = 19.5
    temp_day: float = 19.5
    temp_evening: float = 20.0

    latitude: float = 52.0
    longitude: float = 5.0

    pv_azimuth: float = 148.0
    pv_tilt: float = 50.0
    pv_max_kw: float = 2.0

    dhw_duration_hours: float = 1.0

    min_kwh_threshold: float = 0.3
    avg_baseload_kw: float = 0.15
    max_compressor_freq: int = 70

    sensor_pv: str = "sensor.pv_output"
    sensor_load: str = "sensor.stroomverbruik"
    sensor_hvac: str = "sensor.ecodan_heatpump_ca09ec_status_bedrijf"

    sensor_solcast: str = "sensor.solcast_pv_forecast_forecast_today"

    sensor_home: str = "zone.home"

    database_path: str = "/config/db/database.sqlite"
    solar_model_path: str = "/config/models/solar_model.joblib"

    webapi_host: str = "0.0.0.0"
    webapi_port: int = 8000

    @staticmethod
    def load(client: HAClient):
        config = Config()

        location = client.get_location(config.sensor_home)
        if location != (None, None):
            config.latitude, config.longitude = location

        return config
