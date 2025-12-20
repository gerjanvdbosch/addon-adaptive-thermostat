import logging
import joblib
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor

# Project Imports
from db import fetch_heating_cycles, upsert_heating_cycle
from ha_client import HAClient
from utils import safe_float

logger = logging.getLogger(__name__)


class ThermalAI:
    """
    ThermalAI voor Vloerverwarming & Warmtepomp.
    Houdt rekening met grote thermische massa en trage responstijden.
    """

    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}

        # Config
        self.model_path = Path(
            self.opts.get("thermal_model_path", "/config/models/thermal_model.joblib")
        )
        self.entity_temp = self.opts.get(
            "sensor_temp", "sensor.thermostat_current_temperature"
        )
        self.entity_outside = self.opts.get(
            "sensor_outside", "sensor.outside_temperature"
        )
        self.entity_hvac_action = self.opts.get(
            "sensor_hvac_action", "sensor.thermostat_hvac_action"
        )

        # State tracking
        self.cycle_start_ts = None
        self.start_temp = None
        self.last_state = "idle"
        self.last_run_ts = None

        # Warmtepomp specifieke instellingen
        self.min_cycle_minutes = (
            45  # Vloerverwarming heeft tijd nodig om massa te activeren
        )
        self.dead_time_minutes = (
            20  # Tijd voordat de vloer effect heeft op de luchttemperatuur
        )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
                self.model = (
                    payload.get("model") if isinstance(payload, dict) else payload
                )
                self.is_fitted = True
                logger.info("ThermalAI: Vloerverwarming model geladen.")
            except Exception:
                logger.warning("ThermalAI: Laden model mislukt.")

    def _atomic_save(self, meta=None):
        if not self.model:
            return
        tmp_path = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump({"model": self.model, "meta": meta}, tmp_path)
            tmp_path.replace(self.model_path)
        except Exception:
            logger.exception("ThermalAI: Opslaan mislukt.")

    def run_cycle(self):
        """Detecteert lange verwarmingscycli van de warmtepomp."""
        now = datetime.now(timezone.utc)

        # Check elke minuut
        if self.last_run_ts and (now - self.last_run_ts).total_seconds() < 55:
            return
        self.last_run_ts = now

        try:
            state_obj = self.ha.get_state(self.entity_hvac_action)
            current_action = state_obj.get("state") if state_obj else "unknown"

            temp = safe_float(self.ha.get_state(self.entity_temp).get("state"))
            outside = safe_float(self.ha.get_state(self.entity_outside).get("state"))
        except Exception:
            return

        if temp is None:
            return

        # START: Warmtepomp begint met stoken
        if current_action == "heating" and self.last_state != "heating":
            self.cycle_start_ts = now
            self.start_temp = temp
            logger.info(f"ThermalAI: Warmtepomp cyclus gestart op {temp}°C")

        # EINDE: Warmtepomp stopt (Target bereikt of overshoot fase)
        elif current_action != "heating" and self.last_state == "heating":
            if self.cycle_start_ts and self.start_temp is not None:
                duration_min = (now - self.cycle_start_ts).total_seconds() / 60.0
                temp_delta = temp - self.start_temp

                # Validatie voor vloerverwarming:
                # - Moet lang genoeg gedraaid hebben (min_cycle_minutes)
                # - Temp stijging is vaak erg traag (0.1C per uur is normaal)
                if self.min_cycle_minutes < duration_min < 600 and temp_delta > 0.05:
                    logger.info(
                        f"ThermalAI: Cyclus voltooid: +{temp_delta:.2f}°C in {duration_min:.0f} min."
                    )
                    upsert_heating_cycle(
                        timestamp=self.cycle_start_ts,
                        start_temp=self.start_temp,
                        end_temp=temp,
                        outside_temp=outside or 10.0,
                        duration_minutes=duration_min,
                    )

            self.cycle_start_ts = None
            self.start_temp = None

        self.last_state = current_action

    def train(self):
        """Traint het model met focus op trage opwarming."""
        logger.info("ThermalAI: Training start...")
        df = fetch_heating_cycles(days=120)  # Vloerverwarming heeft meer historie nodig

        if len(df) < 10:
            logger.warning("ThermalAI: Onvoldoende cycli voor vloerverwarming.")
            return

        # Rate = Graden per minuut (bij vloerverwarming vaak 0.001 - 0.005)
        df["rate"] = (df["end_temp"] - df["start_temp"]) / df["duration_minutes"]

        # Filter voor vloerverwarming + WP:
        # Max 0.1°C per 10 min (0.01/min), Min 0.01°C per uur (0.00016/min)
        df = df[(df["rate"] > 0.0001) & (df["rate"] < 0.01)].dropna()

        if len(df) < 8:
            return

        X = df[["start_temp", "outside_temp"]]
        y = df["rate"]

        # Gebruik MAE loss omdat vloerverwarming data grillig kan zijn door zoninstraling
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.03,  # Iets voorzichtiger leren
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )

        try:
            model.fit(X, y)
            self.model = model
            self.is_fitted = True

            avg_rate_hour = df["rate"].mean() * 60
            logger.info(
                f"ThermalAI: Training klaar. Gemiddelde stijging: {avg_rate_hour:.2f}°C/uur."
            )
            self._atomic_save(meta={"avg_rate_hour": avg_rate_hour})
        except Exception:
            logger.exception("ThermalAI: Training gefaald.")

    def predict_heating_time(self, target_temp):
        """
        Voorspelt opwarmtijd inclusief de trage start van een warmtepomp.
        """
        if not self.is_fitted or not self.model:
            return None

        try:
            current_temp = safe_float(self.ha.get_state(self.entity_temp).get("state"))
            outside_temp = safe_float(
                self.ha.get_state(self.entity_outside).get("state")
            )
        except Exception:
            return None

        if current_temp is None or outside_temp is None:
            return None

        delta_needed = target_temp - current_temp

        # Als we binnen 0.1C van target zijn, rekenen we 0 min (overshoot doet de rest)
        if delta_needed <= 0.1:
            return 0.0

        X_pred = pd.DataFrame(
            [[current_temp, outside_temp]], columns=["start_temp", "outside_temp"]
        )

        try:
            # Voorspelde rate per minuut
            pred_rate = float(self.model.predict(X_pred)[0])
            pred_rate = max(pred_rate, 0.0005)  # Absolute ondergrens

            # Berekening: Transporttijd + (Delta / Snelheid)
            # We trekken een kleine marge van delta_needed af (bijv 0.2)
            # omdat de vloer doorwarmt na het uitschakelen (overshoot).
            adjusted_delta = max(0, delta_needed - 0.2)

            minutes_needed = self.dead_time_minutes + (adjusted_delta / pred_rate)

            # Sanity check voor WP: Maximaal 12 uur vooruit plannen
            return min(minutes_needed, 720.0)
        except Exception:
            return None
