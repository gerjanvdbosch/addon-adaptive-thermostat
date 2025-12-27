import logging
import joblib
import time
import pandas as pd
import numpy as np
import shap

from collections import deque
from datetime import datetime
from pathlib import Path

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Project imports
from db import fetch_training_setpoints_df, insert_setpoint
from collector import Collector
from ha_client import HAClient
from utils import safe_float, add_cyclic_time_features, round_half, safe_round

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

        # Deze lijst bepaalt de strikte volgorde van kolommen voor het model.
        self.feature_columns = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "doy_sin",
            "doy_cos",
            "home_presence",
            "hvac_mode",
            "heat_demand",
            "current_temp",
            "current_setpoint",
            "temp_change",
            "outside_temp",
            "min_temp",
            "max_temp",
            "solar_kwh",
            "wind_speed",
            "wind_dir_sin",
            "wind_dir_cos",
        ]

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Runtime State
        self.model = None
        self.is_fitted = False
        self.last_known_setpoint = None
        self.stability_start_ts = None
        self.last_ai_action_ts = None
        self.learning_blocked = False

        # Buffer voor de laatste 10 ticks
        self.prediction_buffer = deque(maxlen=5)

        # Initialisatie
        self._load_model()
        sp = self.ha.get_setpoint()
        if sp is not None:
            self.last_known_setpoint = round_half(sp)

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

        # 2. Voeg tijd-features toe op basis van timestamp
        df = add_cyclic_time_features(df, col_name="timestamp")

        X = df.reindex(columns=self.feature_columns)
        X = X.apply(pd.to_numeric, errors="coerce")
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

    def notify_system_change(self, new_setpoint, block_learning=False):
        """
        Wordt aangeroepen door de Coordinator.
        block_learning=True zorgt dat deze periode NIET wordt opgeslagen als 'normaal gedrag'.
        """
        self.last_known_setpoint = round_half(new_setpoint)
        self.stability_start_ts = None
        self.last_ai_action_ts = datetime.now()

        # Hier zetten we de 'blinddoek' op of af
        self.learning_blocked = block_learning

        if block_learning:
            logger.info(
                "ThermostatAI: Learning gepauzeerd (Solar Boost / Pre-heat actief)."
            )
        else:
            logger.info("ThermostatAI: Learning actief.")

    def update_learning_state(self, raw_data, current_sp):
        """
        Checkt op gebruikersinteractie en stabiliteit.
        """
        ts = datetime.now()
        curr_sp_rounded = round_half(current_sp)

        if self.last_known_setpoint is None:
            self.last_known_setpoint = curr_sp_rounded
            return False

        updated = False

        # 1. DETECTEER HANDMATIGE AANPASSING (USER OVERRIDE)
        # Dit mag ALTIJD doorgaan. Als jij tijdens een Boost aan de knop draait,
        # betekent het dat je het er niet mee eens bent. Dat is waardevolle data.
        if curr_sp_rounded != self.last_known_setpoint:
            is_recent_ai = (
                self.last_ai_action_ts
                and (ts - self.last_ai_action_ts).total_seconds() < 60
            )

            # Als het > 60 sec na onze eigen actie is, en het setpoint is anders -> User action
            if not is_recent_ai:
                prev_sp = self.last_known_setpoint
                logger.info(
                    f"ThermostatAI: User Override gedetecteerd: {prev_sp} -> {curr_sp_rounded}."
                )

                feats = self.collector.features_from_raw(
                    raw_data, timestamp=ts, override_setpoint=prev_sp
                )
                insert_setpoint(
                    feature_dict=feats,
                    setpoint=curr_sp_rounded,
                    observed_current=prev_sp,
                )

                self.last_ai_action_ts = ts
                updated = True

                # Als de gebruiker ingrijpt, heffen we de blokkade op
                self.learning_blocked = False

            self.last_known_setpoint = curr_sp_rounded
            self.stability_start_ts = None

        # 2. STABILITEIT LOGGEN (Alleen als learning NIET geblokkeerd is!)
        else:
            if self.learning_blocked:
                # We zitten in een Boost. We negeren stabiliteit.
                # Dit voorkomt dat '21 graden' als het nieuwe normaal wordt geleerd.
                self.stability_start_ts = None
                return False

            curr_temp = safe_float(raw_data.get("current_temp"))
            is_stable = curr_temp is not None and curr_temp >= current_sp

            if is_stable:
                if self.stability_start_ts is None:
                    self.stability_start_ts = ts
                    logger.info("ThermostatAI: Stabiliteit gestart.")
                else:
                    stable_hours = (ts - self.stability_start_ts).total_seconds() / 3600
                    if stable_hours > float(self.opts.get("stability_hours", 8)):
                        logger.info(
                            "ThermostatAI: Stabiliteit bereikt, loggen setpoint."
                        )
                        feats = self.collector.features_from_raw(raw_data, timestamp=ts)
                        insert_setpoint(
                            feature_dict=feats,
                            setpoint=curr_sp_rounded,
                            observed_current=curr_sp_rounded,
                        )
                        self.stability_start_ts = ts
                    else:
                        stable_hours_remaining = (
                            float(self.opts.get("stability_hours", 8)) - stable_hours
                        )
                        logger.info(
                            f"ThermostatAI: Stabiliteit nog niet bereikt (nog {stable_hours_remaining:.2f}h)."
                        )
            else:
                self.stability_start_ts = None
                logger.info("ThermostatAI: Niet stabiel, reset timer.")

        return updated

    def get_recommended_setpoint(self, features, current_sp):
        """
        Geeft de aanbevolen setpoint terug.
        Houdt nu ook rekening met de COOLDOWN.
        """

        # 1. Voorspelling
        if not self.is_fitted or self.model is None:
            return current_sp

        df_input = pd.DataFrame([features]).reindex(columns=self.feature_columns)
        df_input = df_input.apply(pd.to_numeric, errors="coerce")

        try:
            prediction = self.model.predict(df_input)
            pred_delta = float(prediction[0])
            raw_rec = current_sp + pred_delta

            rounded_rec = safe_round(raw_rec)
            self.prediction_buffer.append(rounded_rec)

            if (
                len(self.prediction_buffer) != 5
                or len(set(self.prediction_buffer)) != 1
            ):
                # Nog aan het twijfelen of buffer niet vol -> Doe niets.
                return current_sp

        except Exception:
            logger.exception("ThermostatAI: Fout bij voorspelling setpoint.")
            return current_sp

        # 3. Bounds checken
        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 25.0))

        return max(min(rounded_rec, max_sp), min_sp)

    def get_influence_factors(self, features, current_sp):
        """
        Berekent de invloed van elke feature op de huidige voorspelling.
        Returns een dictionary met {feature_naam: invloed_in_graden}.
        """
        if not self.is_fitted or self.model is None:
            return {}

        try:
            # 1. Bereid de data voor
            df_input = pd.DataFrame([features]).reindex(columns=self.feature_columns)
            df_input = df_input.apply(pd.to_numeric, errors="coerce")

            # 2. Gebruik SHAP TreeExplainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(df_input)

            # shap_values[0] is een numpy array.
            # We maken een dict en zorgen dat elke waarde een standaard python float is.
            influences = {
                col: float(val)
                for col, val in zip(self.feature_columns, shap_values[0])
            }

            # 3. Groepeer en converteer naar standaard floats
            # We gebruiken float() om numpy.float64 om te zetten naar een native python float
            readable_influences = {
                "Basiswaarde": float(
                    explainer.expected_value[0]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                ),
                "Tijd/dag": float(
                    influences.get("hour_sin", 0)
                    + influences.get("hour_cos", 0)
                    + influences.get("day_sin", 0)
                    + influences.get("day_cos", 0)
                ),
                "Aanwezigheid": float(influences.get("home_presence", 0)),
                "Buitentemperatuur": float(influences.get("outside_temp", 0)),
                "Minimale temperatuur": float(influences.get("min_temp", 0)),
                "Maximale temperatuur": float(influences.get("max_temp", 0)),
                "Zonkracht": float(influences.get("solar_kwh", 0)),
                "Huidige temperatuur": float(influences.get("current_temp", 0)),
                "Huidige setpoint": float(influences.get("current_setpoint", 0)),
                "Temperatuurverandering": float(influences.get("temp_change", 0)),
                "Thermostaatvraag": float(influences.get("heat_demand", 0)),
                "HVAC modus": float(influences.get("hvac_mode", 0)),
                "Windsnelheid": float(influences.get("wind_speed", 0)),
                "Windrichting": float(
                    influences.get("wind_dir_sin", 0)
                    + influences.get("wind_dir_cos", 0)
                ),
            }

            # Optioneel: Rond de waarden af voor een schonere API response
            return {k: round(v, 3) for k, v in readable_influences.items()}

        except Exception as e:
            logger.error(f"ThermostatAI: SHAP berekening mislukt: {e}")
            return {}
