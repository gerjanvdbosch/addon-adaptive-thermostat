import requests
import pandas as pd
import logging

from config import Config

logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, config: Config):
        self.lat = config.latitude
        self.lon = config.longitude
        self.tilt = config.pv_tilt
        self.azimuth = config.pv_azimuth
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    def get_forecast(self):
        sensors = {
            "temperature_2m",
            "cloud_cover",
            "wind_speed_10m",
            "precipitation",
            "shortwave_radiation_instant",
            "diffuse_radiation_instant",
            "global_tilted_irradiance_instant",
        }
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "tilt": self.tilt,
            "azimuth": self.azimuth,
            "minutely_15": ",".join(sensors),
            "timezone": "UTC",
            "forecast_days": 1,
            "past_days": 1,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            minutely = data.get("minutely_15", {})

            if not minutely:
                logger.error("[Weather] Geen 15-min data ontvangen")
                return pd.DataFrame()

            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(minutely["time"], utc=True),
                    "temp": minutely["temperature_2m"],
                    "cloud": minutely["cloud_cover"],
                    "wind": minutely["wind_speed_10m"],
                    "precipitation": minutely["precipitation"],
                    "radiation": minutely["shortwave_radiation_instant"],
                    "diffuse": minutely["diffuse_radiation_instant"],
                    "tilted": minutely["global_tilted_irradiance_instant"],
                }
            )

            logger.debug("[Weather] API-update succesvol")
            return df

        except Exception as e:
            logger.error(f"[Weather] Fout bij ophalen OpenMeteo: {e}")
            return pd.DataFrame()
