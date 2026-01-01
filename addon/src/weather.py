import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, opts):
        self.lat = float(opts.get("latitude", 52.0))
        self.lon = float(opts.get("longitude", 5.0))
        self.tilt = float(opts.get("solar_tilt", 50.0))
        self.azimuth = float(opts.get("solar_azimuth", 148.0))
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    def get_forecast(self):
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "tilt": self.tilt,
            "azimuth": self.azimuth,
            "minutely_15": "temperature_2m,cloud_cover,direct_radiation,diffuse_radiation,global_tilted_irradiance",
            "timezone": "UTC",
            "forecast_days": 1,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            minutely = data.get("minutely_15", {})

            if not minutely:
                logger.error("WeatherClient: Geen 15-min data ontvangen.")
                return pd.DataFrame()

            # 1. Direct DataFrame maken (is al per 15 min!)
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(minutely["time"], utc=True),
                    "temp": minutely["temperature_2m"],
                    "cloud": minutely["cloud_cover"],
                    "radiation": minutely["direct_radiation"],
                    "diffuse": minutely["diffuse_radiation"],
                    "irradiance": minutely["global_tilted_irradiance"],  # W/m2
                }
            )

            logger.info("WeatherClient: API-update succesvol.")
            return df.sort_values("timestamp")

        except Exception as e:
            logger.error(f"WeatherClient: Fout bij ophalen OpenMeteo: {e}")
            return pd.DataFrame()
