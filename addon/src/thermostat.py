import os
import logging
import joblib
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Project imports
from db import fetch_training_setpoints_df, insert_setpoint
from collector import Collector, FEATURE_ORDER
from ha_client import HAClient
from utils import safe_round, safe_float

logger = logging.getLogger(__name__)

class ThermostatAI:
    """
    Slimme Thermostaat AI: Voorspelt de gewenste aanpassing (Delta) van het setpoint.
    Gebruikt HistGradientBoostingRegressor voor robuuste, niet-lineaire regressie.
    """

    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}

        # Config
        self.model_path = Path(self.opts.get("model_path", "/config/models/delta_model.joblib"))
        self.random_state = int(self.opts.get("random_state", 42))
        self.feature_columns = FEATURE_ORDER

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Runtime State
        self.model = None
        self.is_fitted = False
        self.last_known_setpoint = None
        self.last_ai_prediction = None
        self.last_ai_action_ts = None
        self.stability_start_ts = None
        self.last_run_ts = None

        self._load_model()
        self._init_runtime_state()

    # ==============================================================================
    # 1. HELPERS & LOADING
    # ==============================================================================

    def _init_runtime_state(self):
        sp = self.ha.get_shadow_setpoint()
        if sp is not None:
            self.last_known_setpoint = safe_round(sp)

    def _load_model(self):
        if self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
                self.model = payload.get("model")
                self.is_fitted = True
                logger.info("ThermostatAI: Model succesvol geladen.")
            except Exception:
                logger.exception("ThermostatAI: Laden van model mislukt.")

    def _atomic_save(self, model, meta):
        tmp_path = self.model_path.with_suffix(".tmp")
        try:
            payload = {"model": model, "meta": meta}
            joblib.dump(payload, tmp_path)
            tmp_path.replace(self.model_path)
            logger.info("ThermostatAI: Model en meta-data opgeslagen.")
        except Exception:
            logger.exception("ThermostatAI: Opslaan mislukt.")

    # ==============================================================================
    # 2. TRAINING
    # ==============================================================================

    def train(self, force=False):
        """Traint het AI model op basis van de kolom-gebaseerde database data."""
        logger.info("ThermostatAI: Start training...")
        start_time = time.time()

        # Haal data direct op als DataFrame (veel sneller dan loops door JSON)
        df = fetch_training_setpoints_df(days=int(self.opts.get("buffer_days", 30)))

        if df is None or len(df) < 20:
            logger.warning("ThermostatAI: Te weinig data voor training.")
            return

        # Bereken Delta (Vectorized)
        df["delta"] = df["setpoint"] - df["observed_current_setpoint"]

        # Filter uitschieters en ruim op
        df = df[df["delta"].abs() < 10].dropna(subset=["delta"])

        # Zorg dat alle features aanwezig zijn en numeriek zijn
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors='coerce')

        X = df[self.feature_columns]
        y = df["delta"]

        if len(X) < 20:
            return

        # Model configuratie (Optimale parameters voor tabular data)
        new_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=2000,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=self.random_state
        )

        try:
            new_model.fit(X, y)

            # Score berekenen voor logging
            y_pred = new_model.predict(X)
            mae = mean_absolute_error(y, y_pred)

            meta = {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "mae": float(mae),
                "samples": len(X),
                "feature_names": self.feature_columns
            }

            self.model = new_model
            self.is_fitted = True
            self._atomic_save(new_model, meta)

            duration = time.time() - start_time
            logger.info(f"ThermostatAI: Training gereed in {duration:.2f}s. MAE={mae:.3f}")

        except Exception:
            logger.exception("ThermostatAI: Training gecrasht.")

    # ==============================================================================
    # 3. INFERENCE
    # ==============================================================================

    def _predict_delta(self, features: dict):
        """Vraagt het model om een voorspelling middels een 1-rij DataFrame."""
        if self.model is None or not self.is_fitted:
            return 0.0

        # Maak DataFrame met 1 rij om kolom-consistentie te garanderen
        df_input = pd.DataFrame([features])

        # Zorg voor feature alignment
        for col in self.feature_columns:
            if col not in df_input.columns:
                df_input[col] = np.nan

        df_input = df_input[self.feature_columns].apply(pd.to_numeric, errors='coerce')

        try:
            prediction = self.model.predict(df_input)
            return float(prediction[0])
        except Exception:
            logger.exception("ThermostatAI: Prediction error.")
            return 0.0

    # ==============================================================================
    # 4. RUN CYCLE
    # ==============================================================================

    def run_cycle(self):
        """Main loop voor de thermostaat. Logica is onveranderd."""
        ts = datetime.now()

        if self.last_run_ts and (ts - self.last_run_ts).total_seconds() < 5:
            return
        self.last_run_ts = ts

        try:
            raw = self.collector.read_sensors()
            raw["current_setpoint"] = self.ha.get_shadow_setpoint()
        except Exception as e:
            logger.error(f"ThermostatAI: Sensor fout: {e}")
            return

        curr_sp = safe_float(raw.get("current_setpoint"))
        curr_temp = safe_float(raw.get("current_temp"))
        if curr_sp is None: return
        curr_sp_rounded = safe_round(curr_sp)

        # 1. DETECTEER HANDMATIGE AANPASSING (USER OVERRIDE)
        if self.last_known_setpoint is not None and curr_sp_rounded != self.last_known_setpoint:
            cooldown_period = float(self.opts.get("cooldown_hours", 1)) * 3600

            is_recent_ai = self.last_ai_action_ts and (ts - self.last_ai_action_ts).total_seconds() < cooldown_period
            is_ai_val = self.last_ai_prediction is not None and self.last_ai_prediction == curr_sp_rounded

            if not (is_ai_val and is_recent_ai):
                # Menselijke actie: Opslaan en trainen
                prev_sp = self.last_known_setpoint
                logger.info(f"User Override Gedetecteerd: {prev_sp} -> {curr_sp_rounded}. Retraining...")

                # Features ophalen van de collector
                feats = self.collector.features_from_raw(raw, timestamp=ts, override_setpoint=prev_sp)

                # Opslaan in de nieuwe database structuur
                insert_setpoint(feature_dict=feats, setpoint=curr_sp_rounded, observed_current=prev_sp)

                self.last_ai_action_ts = ts
                self.train(force=True)

            self.last_known_setpoint = curr_sp_rounded
            self.stability_start_ts = None
            return

        if self.last_known_setpoint is None:
            self.last_known_setpoint = curr_sp_rounded

        # 2. STABILITEIT LOGGEN
        feats = self.collector.features_from_raw(raw, timestamp=ts)
        is_stable = curr_temp is not None and curr_temp >= curr_sp

        if is_stable:
            if self.stability_start_ts is None:
                self.stability_start_ts = ts
            else:
                stable_hours = (ts - self.stability_start_ts).total_seconds() / 3600
                if stable_hours > float(self.opts.get("stability_hours", 8.0)):
                    logger.info("Stabiliteit bereikt: Datapunt opslaan.")
                    insert_setpoint(feature_dict=feats, setpoint=curr_sp_rounded, observed_current=curr_sp_rounded)
                    self.stability_start_ts = ts
        else:
            self.stability_start_ts = None

        # 3. AI VOORSPELLING
        if not self.is_fitted: return

        cooldown_seconds = float(self.opts.get("cooldown_hours", 1)) * 3600
        if self.last_ai_action_ts and (ts - self.last_ai_action_ts).total_seconds() < cooldown_seconds:
            return

        pred_delta = self._predict_delta(feats)
        new_target = curr_sp + pred_delta

        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 25.0))
        new_target = max(min(new_target, max_sp), min_sp)

        threshold = float(self.opts.get("min_change_threshold", 0.25))
        if abs(new_target - curr_sp) >= threshold:
            logger.info(f"AI Advies: {curr_sp:.1f} -> {new_target:.2f} (Delta: {pred_delta:.2f})")
            self.ha.set_setpoint(new_target)

            self.last_ai_prediction = safe_round(new_target)
            self.last_known_setpoint = safe_round(new_target)
            self.last_ai_action_ts = ts
            self.stability_start_ts = None