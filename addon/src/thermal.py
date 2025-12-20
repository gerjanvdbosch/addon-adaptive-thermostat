import os
import logging
import joblib
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Project Imports
from db import fetch_heating_cycles, upsert_heating_cycle
from ha_client import HAClient
from utils import safe_float

logger = logging.getLogger(__name__)

class ThermalAI:
    """
    Leert de thermische inertie (traagheid) van het huis.
    Vraag: "Hoeveel minuten kost het om van X naar Y te gaan bij buitentemperatuur Z?"
    """

    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}

        # Config
        self.model_path = Path(self.opts.get("model_path_thermal", "/config/models/thermal_model.joblib"))
        self.entity_temp = self.opts.get("sensor_temp", "sensor.thermostat_current_temperature")
        self.entity_target = self.opts.get("sensor_setpoint", "climate.thermostat_target")
        self.entity_outside = self.opts.get("sensor_outside", "sensor.outside_temperature")
        self.entity_hvac_action = self.opts.get("sensor_hvac_action", "sensor.thermostat_hvac_action")
        # hvac_action statussen: 'heating', 'idle', 'off'

        # State tracking voor cyclus detectie
        self.cycle_start_ts = None
        self.start_temp = None
        self.last_state = "idle"

        # Model state
        self.model = None
        self.is_fitted = False

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_model()

    # ==============================================================================
    # 1. MODEL BEHEER
    # ==============================================================================

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
                logger.info("ThermalAI: Model loaded.")
            except Exception:
                logger.warning("ThermalAI: Failed to load model.")

    def _atomic_save(self):
        if not self.model: return
        tmp = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump(self.model, tmp)
            tmp.replace(self.model_path)
            logger.info("ThermalAI: Model saved.")
        except Exception:
            logger.exception("ThermalAI: Save failed.")

    # ==============================================================================
    # 2. DATA COLLECTIE (De "Observer")
    # ==============================================================================

    def track_cycles(self):
        """
        Moet elke minuut draaien.
        Detecteert wanneer de verwarming AAN gaat en weer UIT gaat.
        Logt: (StartTemp, BuitenTemp) -> (Duur, EindTemp)
        """
        now = datetime.now()

        # 1. Lees statussen
        try:
            state_obj = self.ha.get_state(self.entity_hvac_action)
            current_action = state_obj.get("state") if state_obj else "unknown"

            temp = safe_float(self.ha.get_state(self.entity_temp).get("state"))
            outside = safe_float(self.ha.get_state(self.entity_outside).get("state"))
        except:
            return # Sensor error

        if temp is None: return

        # 2. Detecteer START van verwarmen (Idle -> Heating)
        if current_action == "heating" and self.last_state != "heating":
            self.cycle_start_ts = now
            self.start_temp = temp
            logger.debug(f"ThermalAI: Heating cycle started at {temp}°C")

        # 3. Detecteer EINDE van verwarmen (Heating -> Idle)
        elif current_action != "heating" and self.last_state == "heating":
            if self.cycle_start_ts and self.start_temp is not None:
                duration_min = (now - self.cycle_start_ts).total_seconds() / 60.0
                temp_delta = temp - self.start_temp

                # Filter ruis: Alleen opslaan als we minstens 5 min gestookt hebben
                # én de temperatuur daadwerkelijk gestegen is (> 0.1).
                if duration_min > 5 and temp_delta > 0.1:
                    logger.info(f"ThermalAI: Cycle finished. +{temp_delta:.1f}°C in {duration_min:.1f} min.")

                    # Opslaan in DB
                    upsert_heating_cycle(
                        timestamp=self.cycle_start_ts,
                        start_temp=self.start_temp,
                        end_temp=temp,
                        outside_temp=outside if outside is not None else 10.0, # Fallback
                        duration_minutes=duration_min
                    )
                else:
                    logger.debug("ThermalAI: Cycle too short or no heat gain. Ignored.")

            # Reset
            self.cycle_start_ts = None
            self.start_temp = None

        self.last_state = current_action

    # ==============================================================================
    # 3. TRAINEN (Leren van de natuurkunde)
    # ==============================================================================

    def train(self):
        """Leer de relatie tussen Binnen/Buiten temp en Opwarmsnelheid."""
        logger.info("ThermalAI: Training...")
        df = fetch_heating_cycles(days=90) # Haal data op

        if len(df) < 20:
            logger.info("ThermalAI: Too few cycles to train.")
            return

        # We voorspellen: Rate (°C per minuut)
        # Target = (Eind - Start) / Minuten
        df["temp_gain"] = df["end_temp"] - df["start_temp"]
        df["rate"] = df["temp_gain"] / df["duration_minutes"]

        # Filter onrealistische rates (meetfouten)
        df = df[(df["rate"] > 0.001) & (df["rate"] < 0.5)] # Max 0.5 graad per minuut is wel realistisch voor CV

        # Features: Start Temp, Buiten Temp
        # (Hoe kouder het is, hoe trager het gaat)
        X = df[["start_temp", "outside_temp"]]
        y = df["rate"]

        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )

        try:
            model.fit(X, y)
            self.model = model
            self.is_fitted = True
            self._atomic_save()
            logger.info(f"ThermalAI: Training done. Samples={len(df)}")
        except Exception:
            logger.exception("ThermalAI: Training failed")

    # ==============================================================================
    # 4. VOORSPELLEN (Voor de Calculator)
    # ==============================================================================

    def predict_heating_time(self, target_temp):
        """
        Berekent hoeveel minuten het kost om de huidige target te bereiken.
        """
        if not self.is_fitted or not self.model:
            # Fallback: Aanname 1 graad per 20 min (0.05 graad/min)
            return None

        try:
            current_temp = safe_float(self.ha.get_state(self.entity_temp).get("state"))
            outside_temp = safe_float(self.ha.get_state(self.entity_outside).get("state"))
        except:
            return None

        if current_temp is None or outside_temp is None:
            return None

        delta_needed = target_temp - current_temp
        if delta_needed <= 0:
            return 0 # We zijn er al

        # 1. Voorspel snelheid (°C/min)
        # Feature vector moet matchen met training: [start_temp, outside_temp]
        X_pred = pd.DataFrame([[current_temp, outside_temp]], columns=["start_temp", "outside_temp"])
        pred_rate = float(self.model.predict(X_pred)[0])

        # Safety clamp: Minimaal 0.01 graad/min, anders delen we door nul of krijgen we oneindig
        pred_rate = max(pred_rate, 0.01)

        # 2. Bereken tijd
        minutes_needed = delta_needed / pred_rate

        # Voeg 10% buffer toe voor zekerheid
        minutes_needed *= 1.1

        logger.debug(f"ThermalAI: Need {minutes_needed:.1f} min to bridge {delta_needed:.1f}°C (Rate: {pred_rate:.3f}/min)")

        return minutes_needed