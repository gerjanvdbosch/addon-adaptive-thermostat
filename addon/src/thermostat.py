import logging
import joblib
import time
import numpy as np
import pandas as pd
from datetime import datetime
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
    Slimme Thermostaat AI:
    - Detecteert patronen en gebruikersinteracties.
    - Voorspelt de ideale Delta (aanpassing).
    - Bevat Cooldown logica om "zenuwachtig" gedrag te voorkomen.
    """

    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}

        # Config
        self.model_path = Path(
            self.opts.get(
                "thermostat_model_path", "/config/models/thermostat_model.joblib"
            )
        )
        self.feature_columns = FEATURE_ORDER
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Runtime State
        self.model = None
        self.is_fitted = False
        self.last_known_setpoint = None
        self.stability_start_ts = None
        self.last_ai_action_ts = None

        # Initialisatie
        self._load_model()
        sp = self.ha.get_setpoint()
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
    # NIEUW: Feature Engineering Methodes
    # ==============================================================================

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Voegt cyclische tijd-features toe aan de DataFrame op basis van de 'timestamp' kolom.
        Wordt gebruikt tijdens TRAINING.
        """
        if "timestamp" not in df.columns:
            return df

        # Zorg dat timestamp datetime is
        dt = pd.to_datetime(df["timestamp"])
        df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7.0)
        df["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 366.0)
        df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 366.0)

        return df

    # ==============================================================================
    # TRAINING LOGICA
    # ==============================================================================

    def train(self):
        """Traint het AI model op basis van de database data."""
        logger.info("ThermostatAI: Start training...")
        start_time = time.time()

        df = fetch_training_setpoints_df(days=int(self.opts.get("buffer_days", 30)))

        if df is None or len(df) < 20:
            logger.warning("ThermostatAI: Te weinig data voor training.")
            return

        # 1. Bereken de Delta
        df["delta"] = df["setpoint"] - df["current_setpoint"]
        df = df[df["delta"].abs() < 10].dropna(subset=["delta"])

        # 2. NIEUW: Voeg tijd-features toe op basis van timestamp
        df = self._add_time_features(df)

        # 3. Vul ontbrekende kolommen met NaN en converteer naar numeric
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")

        X = df[self.feature_columns]
        y = df["delta"]

        if len(X) < 20:
            return

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
            random_state=42,
        )

        try:
            new_model.fit(X, y)
            mae = mean_absolute_error(y, new_model.predict(X))
            meta = {
                "trained_at": datetime.now().isoformat(),
                "mae": float(mae),
                "samples": len(X),
            }
            self.model = new_model
            self.is_fitted = True
            self._atomic_save(new_model, meta)
            logger.info(
                f"ThermostatAI: Training gereed in {time.time()-start_time:.2f}s. MAE={mae:.3f}"
            )
        except Exception:
            logger.exception("ThermostatAI: Training gecrasht.")

    # ==============================================================================
    # INTERFACE METHODEN VOOR COORDINATOR
    # ==============================================================================

    def notify_system_change(self, new_setpoint):
        """
        Wordt aangeroepen door de Coordinator wanneer de AI/Systeem de setpoint verandert.
        """
        self.last_known_setpoint = safe_round(new_setpoint)
        self.stability_start_ts = None
        self.last_ai_action_ts = datetime.now()

    def update_learning_state(self, raw_data, current_sp):
        """
        Checkt op gebruikersinteractie en stabiliteit.
        """
        ts = datetime.now()
        curr_sp_rounded = safe_round(current_sp)

        if self.last_known_setpoint is None:
            self.last_known_setpoint = curr_sp_rounded
            return False

        updated = False

        # 1. DETECTEER HANDMATIGE AANPASSING (USER OVERRIDE)
        if curr_sp_rounded != self.last_known_setpoint:
            # Check of dit niet stiekem onze eigen actie was (race condition protection)
            is_recent_ai = (
                self.last_ai_action_ts
                and (ts - self.last_ai_action_ts).total_seconds() < 60
            )

            # Als het > 60 sec na onze eigen actie is, en het setpoint is anders -> User action
            if not is_recent_ai:
                prev_sp = self.last_known_setpoint
                logger.info(
                    f"User Override Gedetecteerd: {prev_sp} -> {curr_sp_rounded}. Retraining..."
                )

                feats = self.collector.features_from_raw(
                    raw_data, timestamp=ts, override_setpoint=prev_sp
                )
                insert_setpoint(
                    feature_dict=feats,
                    setpoint=curr_sp_rounded,
                    observed_current=prev_sp,
                )

                #self.train()
                updated = True

            self.last_known_setpoint = curr_sp_rounded
            self.stability_start_ts = None

        # 2. STABILITEIT LOGGEN
        else:
            curr_temp = safe_float(raw_data.get("current_temp"))
            is_stable = curr_temp is not None and curr_temp >= current_sp

            if is_stable:
                if self.stability_start_ts is None:
                    self.stability_start_ts = ts
                else:
                    stable_hours = (ts - self.stability_start_ts).total_seconds() / 3600
                    if stable_hours > float(self.opts.get("stability_hours", 8.0)):
                        logger.info("Stabiliteit bereikt: Datapunt opslaan.")
                        feats = self.collector.features_from_raw(raw_data, timestamp=ts)
                        insert_setpoint(
                            feature_dict=feats,
                            setpoint=curr_sp_rounded,
                            observed_current=curr_sp_rounded,
                        )
                        self.stability_start_ts = ts
            else:
                self.stability_start_ts = None

        return updated

    def get_recommended_setpoint(self, features, current_sp):
        """
        Geeft de aanbevolen setpoint terug.
        Houdt nu ook rekening met de COOLDOWN.
        """
        # 1. Check Cooldown: Als we recent iets veranderd hebben, houden we ons even stil.
        cooldown_seconds = float(self.opts.get("cooldown_hours", 1)) * 3600
        if self.last_ai_action_ts:
            elapsed = (datetime.now() - self.last_ai_action_ts).total_seconds()
            if elapsed < cooldown_seconds:
                # We zitten in de cooldown periode, dus we adviseren: "Doe niets (huidige setpoint)"
                return current_sp

        # 2. Voorspelling
        if not self.is_fitted or self.model is None:
            return current_sp

        df_input = pd.DataFrame([features])

        for col in self.feature_columns:
            if col not in df_input.columns:
                df_input[col] = np.nan

        try:
            prediction = self.model.predict(df_input[self.feature_columns])
            pred_delta = float(prediction[0])
        except Exception:
            return current_sp

        new_target = current_sp + pred_delta

        # 3. Bounds checken
        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 25.0))

        return max(min(new_target, max_sp), min_sp)
