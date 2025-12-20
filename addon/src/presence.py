import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from utils import safe_bool

# Machine Learning
from sklearn.ensemble import HistGradientBoostingClassifier

# Project Imports
from db import fetch_presence_history, upsert_presence_record

logger = logging.getLogger(__name__)


class PresenceAI:
    """
    PresenceAI: Voorspelt aanwezigheid op basis van historische patronen.
    Geoptimaliseerd voor vloerverwarming (lange voorlooptijden).
    """

    def __init__(self, opts: dict):
        self.opts = opts or {}

        # Config
        self.model_path = Path(
            self.opts.get("presence_model_path", "/config/models/presence_model.joblib")
        )

        # Voor vloerverwarming: kijk 180-240 min vooruit
        self.preheat_minutes = int(self.opts.get("preheat_minutes", 180))
        self.confidence_threshold = float(self.opts.get("presence_threshold", 0.70))

        # State
        self.model = None
        self.is_fitted = False
        self.last_log_ts = None

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
                logger.info("PresenceAI: Model geladen.")
            except Exception:
                logger.warning("PresenceAI: Kon model niet laden.")

    def _atomic_save(self, meta=None):
        if not self.model:
            return
        tmp = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump({"model": self.model, "meta": meta}, tmp)
            tmp.replace(self.model_path)
        except Exception:
            logger.exception("PresenceAI: Opslaan mislukt.")

    def _create_features(self, df: pd.DataFrame):
        """Maakt features voor menselijke ritmes."""
        df = df.copy()
        if not np.issubdtype(df["timestamp"], np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Gebruik lokale tijd voor ritme-detectie
        local_dt = (
            df["timestamp"].dt.tz_convert("Europe/Amsterdam").dt
        )  # Pas aan naar jouw TZ
        hour_float = local_dt.hour + local_dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hour_float / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour_float / 24)
        df["day_sin"] = np.sin(2 * np.pi * local_dt.dayofweek / 7)
        df["day_cos"] = np.cos(2 * np.pi * local_dt.dayofweek / 7)
        df["doy_sin"] = np.sin(2 * np.pi * local_dt.dayofyear / 366.0)
        df["doy_cos"] = np.cos(2 * np.pi * local_dt.dayofyear / 366.0)

        return df[
            [
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
                "doy_sin",
                "doy_cos",
            ]
        ]

    def log_current_state(self, features):
        """Verzamelt aanwezigheidsdata van Home Assistant."""
        now = datetime.now()

        # Log elke 15 minuten (voldoende voor patronen)
        if self.last_log_ts and (now - self.last_log_ts).total_seconds() < 900:
            return

        is_home = safe_bool(features.get("home_presence", 0.0))

        upsert_presence_record(now, is_home=is_home)
        self.last_log_ts = now

    def train(self):
        """Traint de classifier om patronen te herkennen."""
        logger.info("PresenceAI: Training start...")
        df = fetch_presence_history(days=90)

        if len(df) < 200:
            logger.warning("PresenceAI: Te weinig data voor betrouwbare voorspelling.")
            return

        X = self._create_features(df)
        y = df["is_home"].astype(int)

        # Classifier: Gebruik class_weight='balanced' omdat mensen vaak
        # meer weg zijn dan thuis (of andersom), wat het model kan vertekenen.
        clf = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            class_weight="balanced",  # Belangrijk voor onregelmatige patronen
        )

        try:
            clf.fit(X, y)
            self.model = clf
            self.is_fitted = True

            score = clf.score(X, y)
            self._atomic_save(meta={"accuracy": score, "samples": len(df)})
            logger.info(f"PresenceAI: Training voltooid (Accuracy: {score:.2f})")
        except Exception:
            logger.exception("PresenceAI: Training gefaald.")

    def should_preheat(self, dynamic_minutes=None):
        """
        Kijkt in de toekomst met een dynamisch venster.
        dynamic_minutes: Het aantal minuten dat we vooruit moeten kijken (geleverd door ThermalAI).
        """
        if not self.is_fitted or not self.model:
            return False, 0.0

        # Gebruik de dynamische waarde, of val terug op de standaard uit de config
        lookahead = (
            int(dynamic_minutes)
            if dynamic_minutes is not None
            else self.preheat_minutes
        )

        # Begrens de lookahead (bijv. minimaal 30 min, maximaal 8 uur voor WP)
        lookahead = max(30, min(lookahead, 480))

        now = datetime.now()

        # Scan het venster (bijv. elke 30 minuten een check doen in de toekomst)
        check_steps = [now + timedelta(minutes=m) for m in range(30, lookahead + 1, 30)]

        # Voeg ook het exacte eindpunt toe
        check_steps.append(now + timedelta(minutes=lookahead))

        max_prob = 0.0
        for future_ts in check_steps:
            df_future = pd.DataFrame([{"timestamp": future_ts}])
            X_future = self._create_features(df_future)
            prob_home = self.model.predict_proba(X_future)[0][1]
            if prob_home > max_prob:
                max_prob = prob_home

        triggered = max_prob >= self.confidence_threshold

        if triggered:
            logger.info(
                f"PresenceAI: Dynamische Pre-heat trigger! Window: {lookahead} min. Kans: {max_prob:.2f}"
            )

        return triggered, max_prob
