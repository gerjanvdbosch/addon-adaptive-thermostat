import logging
import joblib
import pandas as pd

from ha_client import HAClient
from datetime import datetime, timedelta
from pathlib import Path

# Machine Learning
from sklearn.ensemble import HistGradientBoostingClassifier

# Project Imports
from db import fetch_sensor_history  # We moeten ruwe sensor data kunnen ophalen
from utils import add_cyclic_time_features, safe_float

logger = logging.getLogger(__name__)


class DhwAI:
    """
    Domestic Hot Water AI.
    - Leert tapgedrag (wanneer wordt er gedoucht?)
    - Voorspelt noodzaak voor verwarmen als de zon niet schijnt.
    """

    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}

        # Config
        self.model_path = Path(
            self.opts.get("dhw_model_path", "/config/models/dhw_model.joblib")
        )

        self.sww_top = self.opts.get(
            "sensor_top", "sensor.ecodan_heatpump_ca09ec_sww_2e_temp_sensor"
        )
        self.sww_btm = self.opts.get(
            "sensor_bottom", "sensor.ecodan_heatpump_ca09ec_sww_huidige_temp"
        )

        # Thermostat Entity voor SWW (Als je die apart kunt instellen)
        self.dhw_setpoint = self.opts.get("entity_dhw_setpoint", "climate.dhw_water")

        # Instellingen
        self.min_temp = float(
            self.opts.get("dhw_min_temp", 30.0)
        )  # Absolute ondergrens (altijd aan)
        self.target_temp = float(self.opts.get("dhw_target_temp", 50.0))  # Comfort doel
        self.boost_temp = float(
            self.opts.get("dhw_boost_temp", 50.0)
        )  # Solar boost doel

        # Hoe snel moet temp dalen om als 'douchen' geteld te worden? (Graden per minuut)
        self.usage_detection_threshold = 0.4

        # Hoeveel minuten van tevoren moet het water warm zijn?
        self.lookahead_minutes = int(self.opts.get("dhw_lookahead_minutes", 60))
        self.confidence_threshold = (
            0.65  # 65% zekerheid nodig om gas/stroom te verbruiken
        )

        self.feature_columns = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "doy_sin",
            "doy_cos",
        ]

        self.model = None
        self.is_fitted = False

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
                logger.info("DhwAI: Model geladen.")
            except Exception:
                logger.warning("DhwAI: Model laden mislukt.")

    def _atomic_save(self):
        tmp = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump(self.model, tmp)
            tmp.replace(self.model_path)
            logger.info("DhwAI: Model opgeslagen.")
        except Exception:
            logger.exception("DhwAI: Opslaan mislukt.")

    def train(self):
        """
        Analyseert historische temperatuurdata om douche-momenten te vinden
        en traint daarop een classifier.
        """
        logger.info("DhwAI: Training start (Analyseren tapgedrag)...")

        # 1. Haal ruwe data op (SWW Temp) van de afgelopen 60 dagen
        # We nemen aan dat je een helper hebt die raw sensor data als DataFrame geeft
        df = fetch_sensor_history(sensor="sensor.sww_temp", days=60)

        if df is None or len(df) < 1000:
            logger.warning("DhwAI: Te weinig data.")
            return

        # 2. Detecteer 'Events' (Snelle daling)
        # Bereken verschil met vorige meting (resample naar 5 min voor ruisonderdrukking)
        df = df.set_index("timestamp").resample("5min").mean().dropna()
        df["diff"] = df["value"].diff() * -1  # Positief maken bij daling

        # Als temp meer dan X graden zakt in 5 min, is het gebruik
        # We zetten 'is_usage' op 1.
        threshold_5min = self.usage_detection_threshold * 5
        df["is_usage"] = (df["diff"] > threshold_5min).astype(int)

        usage_count = df["is_usage"].sum()
        logger.info(f"DhwAI: {usage_count} tap-events gedetecteerd in data.")

        if usage_count < 10:
            logger.warning("DhwAI: Te weinig tap-events gevonden om te trainen.")
            return

        # 3. Features maken
        df = df.reset_index()
        df = add_cyclic_time_features(df, col_name="timestamp")

        X = df.reindex(columns=self.feature_columns)
        y = df["is_usage"]

        # 4. Train Model
        # We gebruiken class_weight='balanced' omdat douchen zeldzaam is tov 'niet douchen'
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, class_weight="balanced", random_state=42
        )

        try:
            clf.fit(X, y)
            self.model = clf
            self.is_fitted = True
            self._atomic_save()
            logger.info("DhwAI: Training voltooid.")
        except Exception:
            logger.exception("DhwAI: Training gefaald.")

    def get_recommendation(self, current_temp, solar_status_enum):
        """
        Geeft advies: Welke temperatuur moet de boiler hebben?
        """
        now = datetime.now()

        # 1. VEILIGHEID: Kritieke ondergrens (Lege boiler)
        if current_temp < self.min_temp:
            return {
                "action": "CRITICAL_HEAT",
                "target": self.target_temp,
                "reason": f"Temp te laag ({current_temp:.1f}C < {self.min_temp}C)",
            }

        # 2. SOLAR: Als SolarAI zegt 'START', gaan we boosten
        # (We luisteren hier naar de status van SolarAI die de coordinator doorgeeft)
        if str(solar_status_enum) == "START":
            # Check of we nog ruimte hebben (bottom sensor is hier handig voor, maar top werkt ook)
            if current_temp < self.boost_temp:
                return {
                    "action": "SOLAR_BOOST",
                    "target": self.boost_temp,
                    "reason": "Zonne-energie benutten",
                }
            else:
                return {
                    "action": "IDLE",
                    "target": self.min_temp,  # Terug naar min, buffer is vol
                    "reason": "Zonnebuffer vol",
                }

        # 3. VOORSPELLING: Alleen als solar 'DONE', 'NIGHT' of 'LOW_LIGHT' is
        # Of als we gewoon zeker willen zijn.
        if self.is_fitted:
            # Kijk vooruit in de tijd
            future_ts = now + timedelta(minutes=self.lookahead_minutes)

            # Maak features voor dat toekomstige moment
            df_fut = pd.DataFrame([{"timestamp": future_ts}])
            df_fut = add_cyclic_time_features(df_fut)
            X_fut = df_fut.reindex(columns=self.feature_columns)

            # Voorspel kans
            prob = self.model.predict_proba(X_fut)[0][1]  # Kans op klasse 1

            if prob > self.confidence_threshold:
                # Moeten we nog stoken?
                if current_temp < self.target_temp:
                    return {
                        "action": "PREDICTIVE_HEAT",
                        "target": self.target_temp,
                        "reason": f"Verwacht gebruik om {future_ts.strftime('%H:%M')} ({prob:.2f})",
                    }

        # 4. RUST: Niets aan de hand
        return {
            "action": "IDLE",
            "target": self.min_temp,  # Val terug op ondergrens
            "reason": "Geen vraag, geen zon",
        }

    def run_cycle(self, solar_advice):
        if self.sww_top is None:
            return

        # 2. Wat zegt SolarAI op dit moment?
        solar_status = solar_advice.get("action")  # Enum: START, WAIT, NIGHT, etc.

        # 3. Vraag DhwAI om advies
        advice = self.get_recommendation(self.sww_top, solar_status)

        action = advice["action"]
        target = advice["target"]
        reason = advice["reason"]

        # 4. Uitvoeren
        # Hier moet je weten hoe je jouw SWW aanstuurt.
        # Vaak is dat een setpoint zetten op de warmtepomp.

        current_dhw_sp = safe_float(
            self.ha.get_state(self.entity_dhw_setpoint, attribute="temperature")
        )

        # Deadband check (niet voor elke 0.1 graad sturen)
        if current_dhw_sp is not None and abs(current_dhw_sp - target) < 1.0:
            return

        logger.info(f"DhwAI: Actie [{action}] -> {target}C ({reason})")

        # Stuur commando naar HA
        self.ha.set_dhw_setpoint(target)  # Zelf implementeren in HAClient
