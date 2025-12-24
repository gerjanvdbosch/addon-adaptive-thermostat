import logging
import joblib
import numpy as np
import pandas as pd
import shap

from datetime import datetime, timedelta
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier
from db import fetch_presence_history, upsert_presence_record
from utils import safe_bool, add_cyclic_time_features

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

        self.feature_columns = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "doy_sin",
            "doy_cos",
        ]

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
        df = add_cyclic_time_features(df, col_name="timestamp")

        df_out = df.reindex(columns=self.feature_columns)
        df_out = df_out.apply(pd.to_numeric, errors="coerce")

        return df_out

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

        if len(df) < 2000:
            logger.warning("PresenceAI: Te weinig data voor betrouwbare voorspelling.")
            return

        X = self._create_features(df)
        y = df["is_home"].astype(int)

        # Filter ongeldige targets (features mogen NaN zijn, targets niet)
        mask = np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(X) < 100:
            return

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

    def get_influence_factors(self, target_time):
        """
        Gebruikt SHAP om uit te leggen WAAROM het model denkt dat je thuis bent (of komt).
        Breekt de voorspelling op in: Tijdstip, Weekdag en Seizoen.
        """
        if not self.is_fitted or not self.model:
            return {"Status": "Model nog niet getraind"}

        try:
            # 1. Bereid data voor (voor het specifieke tijdstip)
            df = pd.DataFrame([{"timestamp": target_time}])
            X = self._create_features(df)

            # 2. SHAP Berekening
            # Voor classifiers geeft TreeExplainer de impact op de 'log-odds' (kansverhouding)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            # Bij binaire classificatie (Thuis/Niet Thuis) is de output soms een lijst
            # We hebben de waarden voor de positieve klasse (1 = Thuis) nodig.
            if isinstance(shap_values, list):
                vals = shap_values[1][0]  # [0] is klasse 0, [1] is klasse 1
            elif len(shap_values.shape) == 2:
                vals = shap_values[0]  # Soms direct de array
            else:
                vals = shap_values

            raw_influences = {
                col: float(val) for col, val in zip(self.feature_columns, vals)
            }

            # 3. Groeperen en vertalen
            influences = {}

            # Helper
            def format_impact(val):
                if abs(val) < 0.05:
                    return None
                # Positief = Verhoogt de kans op aanwezigheid
                # Negatief = Verlaagt de kans (dus voorspelt afwezigheid)
                return "Verhoogt Kans" if val > 0 else "Verlaagt Kans"

            # Tijdstip (Uur van de dag)
            # Dit pakt het dag-nacht ritme
            hour_impact = raw_influences.get("hour_sin", 0) + raw_influences.get(
                "hour_cos", 0
            )
            imp = format_impact(hour_impact)
            if imp:
                influences["Tijdstip"] = f"{imp} ({hour_impact:+.2f})"

            # Weekdag (Werkdag vs Weekend)
            # Dit pakt het weekritme (bijv. woensdagmiddag thuis, maandag weg)
            day_impact = raw_influences.get("day_sin", 0) + raw_influences.get(
                "day_cos", 0
            )
            imp = format_impact(day_impact)
            if imp:
                influences["Weekdag"] = f"{imp} ({day_impact:+.2f})"

            # Seizoen (Vakanties/Jaarritme)
            # Minder relevant voor dagelijkse patronen, maar kan vakanties oppikken
            doy_impact = raw_influences.get("doy_sin", 0) + raw_influences.get(
                "doy_cos", 0
            )
            if abs(doy_impact) > 0.1:  # Alleen melden als het significant is
                imp = format_impact(doy_impact)
                influences["Seizoen"] = f"{imp} ({doy_impact:+.2f})"

            return influences

        except Exception as e:
            logger.error(f"PresenceAI: SHAP berekening mislukt: {e}")
            return {}
