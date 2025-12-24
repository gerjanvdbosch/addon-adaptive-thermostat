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
from utils import safe_float, add_cyclic_time_features

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

        # Let op met de volgorde, cst moet ook overeenkomen
        self.feature_columns = [
            "start_temp",
            "end_temp",
            "avg_outside_temp",
            "avg_solar",
            "avg_supply_temp",
            "doy_sin",
            "doy_cos",
        ]

        # State tracking
        self.is_fitted = False
        self.model = None
        self.cycle_start_ts = None
        self.start_temp = None

        self.solar_samples = []
        self.supply_temps = []
        self.outside_samples = []

        self.last_state = "off"
        self.last_run_ts = None

        # Instellingen
        self.min_cycle_minutes = 45  # Vloerverwarming is traag
        self.dead_time_minutes = 20  # Transporttijd

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
                logger.info("ThermalAI: Model succesvol geladen.")
            except Exception:
                logger.warning("ThermalAI: Laden model mislukt.")
                self.model = None
                self.is_fitted = False

    def _atomic_save(self, meta=None):
        if not self.model:
            return
        tmp_path = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump({"model": self.model, "meta": meta}, tmp_path)
            tmp_path.replace(self.model_path)
            logger.info("ThermalAI: Model opgeslagen.")
        except Exception:
            logger.exception("ThermalAI: Opslaan mislukt.")

    def run_cycle(self, features, current_action):
        """
        Monitort de warmtepomp en slaat voltooide cycli op in de DB.
        """
        now = datetime.now(timezone.utc)

        # Check elke minuut
        if self.last_run_ts and (now - self.last_run_ts).total_seconds() < 55:
            return
        self.last_run_ts = now

        # Haal data op
        temp = safe_float(features.get("current_temp"))
        outside = safe_float(features.get("outside_temp"))
        solar = safe_float(features.get("pv_power", 0.0))
        supply = safe_float(features.get("supply_temp"))

        if temp is None:
            return

        # --- FASE 1: START CYCLUS ---
        if current_action == "heating" and self.last_state != "heating":
            self.cycle_start_ts = now
            self.start_temp = temp

            self.solar_samples = []
            self.supply_temps = []
            self.outside_samples = []

            if solar is not None:
                self.solar_samples.append(solar)
            if supply is not None:
                self.supply_temps.append(supply)
            if outside is not None:
                self.outside_samples.append(outside)

            logger.info(f"ThermalAI: Verwarmingscyclus gestart op {temp}°C")

        # --- FASE 2: TIJDENS CYCLUS ---
        elif current_action == "heating" and self.cycle_start_ts:
            if solar is not None:
                self.solar_samples.append(solar)
            if supply is not None:
                self.supply_temps.append(supply)

        # --- FASE 3: EINDE CYCLUS ---
        elif current_action != "heating" and self.last_state == "heating":
            if self.cycle_start_ts and self.start_temp is not None:
                duration_min = (now - self.cycle_start_ts).total_seconds() / 60.0
                temp_delta = temp - self.start_temp

                # Validatie:
                # 1. Minimaal 45 min (anders is het ruis/test)
                # 2. Maximaal 24 uur (lange run in de winter is normaal voor WP)
                # 3. Minimaal 0.05 graad stijging (anders meten we niks)
                if self.min_cycle_minutes < duration_min < 1440 and temp_delta > 0.05:
                    # Bereken de echte gemiddelden over de gehele periode
                    avg_supply = (
                        (sum(self.supply_temps) / len(self.supply_temps))
                        if self.supply_temps
                        else 30.0
                    )
                    avg_solar = (
                        (sum(self.solar_samples) / len(self.solar_samples))
                        if self.solar_samples
                        else 0.0
                    )
                    avg_outside = (
                        (sum(self.outside_samples) / len(self.outside_samples))
                        if self.outside_samples
                        else (outside or 10.0)
                    )

                    logger.info(
                        f"ThermalAI: Cyclus voltooid: +{temp_delta:.2f}°C in {duration_min:.0f} min. "
                        f"Gemiddelde zon: {avg_solar:.0f}W, Gemiddelde aanvoer: {avg_supply:.1f}°C, Gemiddelde buitentemp: {avg_outside:.1f}°C"
                    )

                    upsert_heating_cycle(
                        timestamp=self.cycle_start_ts,
                        start_temp=self.start_temp,
                        end_temp=temp,
                        duration_minutes=duration_min,
                        avg_outside_temp=avg_outside,
                        avg_solar=avg_solar,
                        avg_supply_temp=avg_supply,
                    )
                else:
                    logger.debug(
                        f"ThermalAI: Cyclus genegeerd. Duur: {duration_min:.0f}m, Delta: {temp_delta:.2f}C"
                    )

            # Reset
            self.cycle_start_ts = None
            self.start_temp = None
            self.solar_samples = []
            self.supply_temps = []
            self.outside_samples = []

        self.last_state = current_action

    def train(self):
        """
        Traint het model.
        Gebruikt Monotonic Constraints om fysische wetten af te dwingen.
        """
        logger.info("ThermalAI: Training start...")
        df = fetch_heating_cycles(days=120)

        if df is None or len(df) < 10:
            logger.warning("ThermalAI: Te weinig data voor training.")
            return

        # Rate = Graden per minuut
        # Omdat we delen door de totale duur (incl opstarten), zit de 'dode tijd' hier impliciet in.
        df["rate"] = (df["end_temp"] - df["start_temp"]) / df["duration_minutes"]

        # Filter ruis (vloerverwarming is traag: 0.0001 - 0.01 graad/minuut)
        df = df[(df["rate"] > 0.0001) & (df["rate"] < 0.01)].dropna()

        df = add_cyclic_time_features(df, col_name="timestamp")

        if len(df) < 8:
            logger.warning("ThermalAI: Te weinig 'clean' data na filtering.")
            return

        # Hierdoor garandeer je dat X altijd exact overeenkomt met de feature list
        X = df.reindex(columns=self.feature_columns)
        X = X.apply(pd.to_numeric, errors="coerce")

        y = df["rate"]

        # MONOTONIC CONSTRAINTS
        # We vertellen de AI hoe de wereld werkt:
        # -1: Negatieve invloed (hogere waarde = tragere verwarming)
        #  1: Positieve invloed (hogere waarde = snellere verwarming)
        #  0: Geen dwang (leer zelf de relatie)

        cst = [
            -1,  # start_temp: Hoe warmer binnen, hoe trager de stijging
            1,  # end_temp: Hoe hoger het doel, hoe harder de WP werkt
            1,  # outside_temp: Hoe warmer buiten, hoe minder verlies
            1,  # avg_solar: Hoe meer zon, hoe meer hulp
            1,  # avg_supply_temp: Hoe heter het water, hoe sneller warm
            0,  # doy_sin: Cyclisch
            0,  # doy_cos: Cyclisch
        ]

        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.03,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            monotonic_cst=cst,
        )

        try:
            model.fit(X, y)
            self.model = model
            self.is_fitted = True
            self._atomic_save()
            logger.info("ThermalAI: Training voltooid.")
        except Exception:
            logger.exception("ThermalAI: Training gefaald.")

    def predict_heating_time(self, target_temp, features):
        """
        Voorspelt opwarmtijd naar target_temp.
        """
        if not self.is_fitted or not self.model:
            return 180.0  # Fallback 3 uur

        # Data ophalen
        curr = safe_float(features.get("current_temp"))
        out = safe_float(features.get("outside_temp"))
        sol = safe_float(features.get("pv_power", 0.0))

        # Voor supply moeten we een aanname doen als hij nog niet draait
        # Een simpele stooklijn benadering is vaak goed genoeg voor een schatting
        sup = safe_float(features.get("supply_temp"))

        if not sup or sup < 20:
            sup = 25.0
        if curr is None:
            return None
        if out is None:
            out = 10.0

        delta_needed = target_temp - curr

        # Als we er bijna zijn, rekenen we 0 min (overshoot doet de rest)
        if delta_needed <= 0.1:
            return 0.0

        # Dictionary bouwen met de juiste keys
        input_data = {
            "timestamp": datetime.now(),
            "start_temp": curr,
            "end_temp": target_temp,
            "avg_outside_temp": out,
            "avg_solar": sol,
            "avg_supply_temp": sup,
        }

        df_pred = pd.DataFrame([input_data])
        df_pred = add_cyclic_time_features(df_pred)

        X_pred = df_pred.reindex(columns=self.feature_columns)
        X_pred = X_pred.apply(pd.to_numeric, errors="coerce")

        try:
            pred_rate = float(self.model.predict(X_pred)[0])
            pred_rate = max(pred_rate, 0.0005)

            # Overshoot correctie
            adjusted_delta = max(0, delta_needed - 0.2)
            minutes_needed = adjusted_delta / pred_rate

            return min(minutes_needed, 720.0)
        except Exception:
            logger.exception("ThermalAI: Fout bij predictie.")
            return 180.0
