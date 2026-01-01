import logging
from datetime import datetime, timedelta
from thermal import ThermalPhysics
import pandas as pd

logger = logging.getLogger(__name__)


class ThermostatAI:
    def __init__(self, ha, collector, opts):
        self.ha = ha
        self.physics = ThermalPhysics()  # Laadt automatisch
        self.sleep_temp = float(opts.get("sleep_temp", 15.0))
        self.wake_time = opts.get("wake_time", "07:00")
        self.sleep_time = opts.get("sleep_time", "23:00")
        # ... (ML Model init hier) ...

    def check_solar_deferral(self, current_temp, current_sp, forecast_df):
        """Snooze functie: Wacht op zon als het niet te koud is."""
        if forecast_df is None or forecast_df.empty:
            return None
        if current_temp < (current_sp - 0.5):
            return None  # Te koud om te wachten

        # Kijk 90 min vooruit
        end = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=90)
        future = forecast_df[
            (forecast_df["timestamp"] > pd.Timestamp.now(tz="UTC"))
            & (forecast_df["timestamp"] < end)
        ]

        if not future.empty and future["power_corrected"].max() > 1.0:  # Drempel 1kW
            logger.info("Thermostaat: Snooze actief, zon komt eraan.")
            return max(current_sp - 0.5, 15.0)  # Zak 0.5 graad
        return None

    def get_recommendation(self, features, current_sp):
        """Hybride Logica: Fysica (Nacht) of ML (Dag)."""
        now = datetime.now()
        cur_temp = features.get("current_temp")

        # Tijden parsen
        t_wake = now.replace(
            hour=int(self.wake_time.split(":")[0]),
            minute=int(self.wake_time.split(":")[1]),
        )
        t_sleep = now.replace(
            hour=int(self.sleep_time.split(":")[0]),
            minute=int(self.sleep_time.split(":")[1]),
        )
        is_night = now >= t_sleep or now < t_wake

        # 1. Pre-Heat (Ochtend)
        if is_night and now.hour < 10:
            comfort = 20.0  # Target
            hours = self.physics.time_to_heat(cur_temp, comfort)
            if now >= (t_wake - timedelta(hours=hours)):
                return comfort, "Pre-heat"

        # 2. Smart Coasting (Avond)
        if not is_night and now.hour > 18:
            hours_cool = self.physics.time_to_cool(
                cur_temp, self.sleep_temp, features.get("outside_temp")
            )
            hours_sleep = (t_sleep - now).total_seconds() / 3600
            if hours_cool > hours_sleep:
                return self.sleep_temp, "Coasting"

        if is_night:
            return self.sleep_temp, "Nacht"

        # 3. Dag: ML Model (Fallback naar huidig setpoint voor dit voorbeeld)
        # return self.ml_model.predict(...)
        return current_sp, "Comfort (Manual/ML)"
