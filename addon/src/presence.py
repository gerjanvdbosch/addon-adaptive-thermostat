import os
import logging
import joblib
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Machine Learning (Classifier want we voorspellen JA/NEE)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Project Imports
from db import fetch_presence_history, upsert_presence_record
from ha_client import HAClient

logger = logging.getLogger(__name__)

class PresenceAI:
    """
    Het 'Waarzegger' brein.
    Voorspelt of er iemand thuis ZAL zijn in de toekomst, zodat de verwarming
    alvast aan kan voordat de sleutel in het slot steekt.
    """

    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}

        # Config
        self.model_path = Path(self.opts.get("model_path_presence", "/config/models/presence_model.joblib"))
        self.presence_sensor = self.opts.get("sensor_presence", "zone.home") # Of group.family

        # Parameters
        self.preheat_minutes = int(self.opts.get("preheat_minutes", 60)) # Hoelang duurt opwarmen?
        self.confidence_threshold = float(self.opts.get("presence_threshold", 0.75)) # 75% zekerheid nodig

        # State
        self.model = None
        self.is_fitted = False
        self.last_train_ts = None
        self.last_log_ts = None

        # Zorg dat map bestaat
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
                logger.info("PresenceAI: Model loaded.")
            except Exception:
                logger.warning("PresenceAI: Could not load model.")

    def _atomic_save(self):
        if not self.model: return
        tmp = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump(self.model, tmp)
            tmp.replace(self.model_path)
            logger.info("PresenceAI: Model saved.")
        except Exception:
            logger.exception("PresenceAI: Save failed.")

    def _create_features(self, df: pd.DataFrame):
        """Maakt tijds-features van een timestamp."""
        df = df.copy()
        if not np.issubdtype(df["timestamp"], np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Lokale tijd is belangrijk voor menselijk ritme!
        # We converteren naar "lokale uren" door simpelweg de timezone offset mee te nemen of naive te werken
        # Voor ML is cyclisch het belangrijkst.

        dt = df["timestamp"].dt

        # Cyclische tijd
        df["hour_sin"] = np.sin(2 * np.pi * dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * dt.hour / 24)

        # Cyclische weekdag (Maandag vs Zondag)
        df["day_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df["day_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # Is het weekend?
        df["is_weekend"] = dt.dayofweek >= 5

        return df[["hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend"]]

    # ==============================================================================
    # 2. TRAINING (Leren van verleden)
    # ==============================================================================

    def log_current_state(self):
        """Wordt elke ~5/15 min aangeroepen om data te verzamelen."""
        now = datetime.now()

        # Rate limiter voor logging (niet vaker dan 1x per 10 min opslaan is genoeg voor patronen)
        if self.last_log_ts and (now - self.last_log_ts).total_seconds() < 600:
            return

        state = self.ha.get_state(self.presence_sensor)
        # HA 'zone' states zijn getallen (aantal personen) of 'home'/'not_home'
        is_home = False
        if state:
            val = state.get("state")
            # Logica: Als het een getal is > 0, of 'home', of 'on'
            if str(val).isdigit():
                is_home = int(val) > 0
            else:
                is_home = str(val) in ["home", "on", "true"]

        upsert_presence_record(now, is_home=is_home)
        self.last_log_ts = now

    def train(self):
        """Traint het model (bijv. 1x per week 's nachts)."""
        logger.info("PresenceAI: Start training...")

        # Haal historie op (bijv. laatste 60 dagen)
        df = fetch_presence_history(days=60)

        if len(df) < 100:
            logger.info("PresenceAI: Too few samples to train.")
            return

        X = self._create_features(df)
        y = df["is_home"].astype(int) # 0 of 1

        # We gebruiken een Classifier!
        # Deze geeft een 'kans' (probability) terug in plaats van een hard getal.
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=1000,
            max_leaf_nodes=31,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42
        )

        try:
            clf.fit(X, y)
            self.model = clf
            self.is_fitted = True
            self._atomic_save()

            score = clf.score(X, y) # Accuracy
            logger.info(f"PresenceAI: Training complete. Accuracy={score:.2f}")
        except Exception:
            logger.exception("PresenceAI: Training failed")

    # ==============================================================================
    # 3. VOORSPELLEN (Pre-Heating Check)
    # ==============================================================================

    def should_preheat(self):
        """
        Kijkt in de toekomst.
        Returns: (bool, probability) -> Moeten we stoken? En hoe zeker zijn we?
        """
        if not self.is_fitted or not self.model:
            return False, 0.0

        # We kijken X minuten vooruit
        future_ts = datetime.now() + timedelta(minutes=self.preheat_minutes)

        # Maak dataframe voor dit ene moment
        df_future = pd.DataFrame([{"timestamp": future_ts}])
        X_future = self._create_features(df_future)

        # Voorspel kans [Kans_Niet_Thuis, Kans_Wel_Thuis]
        probs = self.model.predict_proba(X_future)[0]
        prob_home = probs[1] # De kans op '1' (Wel thuis)

        # Check: Is er NU al iemand thuis? (Dan hoeven we niet te 'pre'-heaten, maar is het gewoon 'aan')
        # Dit laten we over aan de ThermostatAI.
        # Wij gaan puur over: "Er is niemand, maar KOMT er iemand?"

        triggered = prob_home >= self.confidence_threshold

        if triggered:
            logger.info(f"PresenceAI: Pre-heat trigger! Expecting arrival at {future_ts.strftime('%H:%M')} (Conf: {prob_home:.2f})")
        else:
            logger.debug(f"PresenceAI: No pre-heat. Prob home at {future_ts.strftime('%H:%M')} is {prob_home:.2f}")

        return triggered, prob_home