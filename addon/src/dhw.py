import logging
import joblib
import pandas as pd
import shap

from enum import Enum
from collections import deque
from ha_client import HAClient
from datetime import datetime, timedelta
from pathlib import Path

# Machine Learning
from sklearn.ensemble import HistGradientBoostingClassifier

# Project Imports
from db import fetch_dhw_history, upsert_dhw_sensor_data
from utils import add_cyclic_time_features, safe_float

logger = logging.getLogger(__name__)


class SensorPosition(Enum):
    TOP = 1
    BOTTOM = 2


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

        # Instellingen
        self.min_temp = float(
            self.opts.get("dhw_min_temp", 30.0)
        )  # Absolute ondergrens
        self.target_temp = float(self.opts.get("dhw_target_temp", 50.0))  # Comfort doel
        self.boost_temp = float(
            self.opts.get("dhw_boost_temp", 55.0)
        )  # Solar boost doel

        # 0.2 graden/minuut = 1.0 graad per 5 minuten.
        # Dit is gevoelig genoeg voor douche, maar filtert stilstand (0.001/min).
        self.usage_detection_threshold = 0.2

        # Hoeveel minuten van tevoren moet het water warm zijn?
        self.lookahead_minutes = int(self.opts.get("dhw_lookahead_minutes", 90))
        self.confidence_threshold = 0.65

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

        # --- Runtime State ---
        # Buffer voor live detectie (glijdende schaal over 5 min)
        self.temp_buffer = deque(maxlen=5)

        # Buffer voor output stabiliteit (voorkom pendelen WP)
        self.action_buffer = deque(maxlen=3)
        self.last_stable_setpoint = self.min_temp

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
        df = fetch_dhw_history(sensor=SensorPosition.TOP, days=60)

        if df is None or len(df) < 100:
            logger.warning("DhwAI: Te weinig data.")
            return

        # 2. FILTER: Negeer data tijdens SWW-run (Destratificatie/Mixing)
        # Als hvac_mode 'dhw' is, is een temperatuurdaling GEEN gebruik.
        # We filteren deze regels eruit.
        if "hvac_mode" in df.columns:
            # Behoud alleen rijen waar de WP NIET bezig is met warm water
            df = df[df["hvac_mode"] != 2]  # 2 = SWW modus

        # 3. Detecteer Events (Aangepast voor 0.5C steps)
        # We resamplen naar 5 minuten en kijken naar het verschil.
        df = df.set_index("timestamp").sort_index()

        # Interpoleren om gaten te vullen, daarna resamplen
        df_resampled = df["value"].resample("5min").mean().interpolate()

        # Bereken het verschil met 5 minuten geleden (shift 1 op 5min data)
        # diff is positief als temp daalt
        diff_series = df_resampled.diff(periods=1) * -1

        # Drempel: usage_threshold (0.2) * 5 minuten = 1.0 graad daling
        threshold_total_drop = self.usage_detection_threshold * 5.0

        # Labeling
        usage_labels = (diff_series > threshold_total_drop).astype(int)

        usage_count = usage_labels.sum()
        logger.info(f"DhwAI: {usage_count} tap-events gedetecteerd in data.")

        if usage_count < 10:
            logger.warning("DhwAI: Te weinig tap-events gevonden om te trainen.")
            return

        # 4. Features maken
        df_features = pd.DataFrame(index=df_resampled.index)
        df_features = add_cyclic_time_features(
            df_features, col_name=None
        )  # Index is datetime

        X = df_features.reindex(columns=self.feature_columns)
        y = usage_labels

        # 5. Train Model
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=500,
            class_weight="balanced",  # Belangrijk: douchen is zeldzaam
            random_state=42,
        )

        try:
            clf.fit(X, y)
            self.model = clf
            self.is_fitted = True
            self._atomic_save()
            logger.info("DhwAI: Training voltooid.")
        except Exception:
            logger.exception("DhwAI: Training gefaald.")

    # ==============================================================================
    # 2. RUNTIME DETECTIE
    # ==============================================================================

    def _detect_current_usage(self, current_temp):
        """
        Bepaalt of er NU gedoucht wordt, rekening houdend met 0.5C sensors en traagheid.
        """
        self.temp_buffer.append(current_temp)

        if len(self.temp_buffer) < 2:
            return False

        # Vergelijk oudste (5 min geleden) met nu
        # Bijv: buffer is [50.0, 50.0, 50.0, 49.5, 49.5] -> drop 0.5
        drop = self.temp_buffer[0] - self.temp_buffer[-1]

        # Drempel: 0.5 graad in 5 minuten is voor een 200L vat al een indicatie
        # Als je sensor stappen van 0.5 doet, is >= 0.5 de enige logische check.
        if drop >= 0.5:
            # Extra check: Is de temp echt aan het zakken? (voorkom ruis 50->49.5->50)
            return True

        return False

    # ==============================================================================
    # 3. BESLUITVORMING
    # ==============================================================================

    def get_recommendation(self, current_temp):
        """
        Geeft advies: Welke temperatuur moet de boiler hebben?
        """
        now = datetime.now()
        is_showering_now = self._detect_current_usage(current_temp)

        if is_showering_now:
            # Optioneel: Als er gedoucht wordt, zet setpoint tijdelijk laag
            # zodat WP niet gaat stoken TIJDENS douchen (comfort/flow issue).
            # Of juist hoog om buffer te houden. Afhankelijk van je systeem.
            pass

        # 1. VEILIGHEID: Kritieke ondergrens (Lege boiler)
        if current_temp < self.min_temp:
            logger.info(f"DhwAI: Temp kritiek laag {current_temp:.1f}C")
            return self.target_temp

        solar_status_enum = None  # Placeholder voor SolarAI status

        # 2. SOLAR: Als SolarAI zegt 'START', gaan we boosten
        # (We luisteren hier naar de status van SolarAI die de coordinator doorgeeft)
        if str(solar_status_enum) == "START":
            # Check of we nog ruimte hebben (bottom sensor is hier handig voor, maar top werkt ook)
            if current_temp < self.boost_temp:
                logger.info("DhwAI: Zonnebuffer actief.")
                return self.boost_temp
            else:
                logger.info("DhwAI: Zonnebuffer vol")
                return self.min_temp

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
                    logger.info(
                        f"DhwAI: Verwacht gebruik om {future_ts.strftime('%H:%M')} ({prob:.0%})"
                    )
                    return self.target_temp

        # 4. RUST
        return self.min_temp

    # ==============================================================================
    # 4. MAIN LOOP
    # ==============================================================================

    def run_cycle(self, features):
        temp = safe_float(features.get("dhw_temp"))

        if temp is None:
            logger.info("DhwAI: geen temp beschikbaar")
            return

        hvac_mode = features.get("hvac_mode")

        upsert_dhw_sensor_data(
            sensor_id=SensorPosition.TOP,
            value=temp,
            hvac_mode=hvac_mode,
        )

        # 2. Wat zegt SolarAI op dit moment?
        # solar_status = solar_advice.get("action")  # Enum: START, WAIT, NIGHT, etc.

        # 3. Vraag DhwAI om advies
        target = self.get_recommendation(temp)  # solar_status

        # 4. STABILITEIT (Debounce)
        # We voeren de actie pas uit als we 3x (3 minuten) hetzelfde setpoint willen
        self.action_buffer.append(target)

        if len(self.action_buffer) == 3:
            # Check of ze alle 3 gelijk zijn (stabiele wens)
            if len(set(self.action_buffer)) == 1:
                stable_target = self.action_buffer[0]

                # Voer alleen uit als het afwijkt van wat we als laatst deden
                current_dhw_sp = self.ha.get_dhw_setpoint()

                # Deadband van 1 graad
                if (
                    current_dhw_sp is not None
                    and abs(current_dhw_sp - stable_target) < 1.0
                ):
                    return  # Niks doen

                self.ha.set_dhw_setpoint(stable_target)
                self.last_stable_setpoint = stable_target
            else:
                # Flipperend advies (bijv net wel/niet solar), doe niets
                pass

    def get_influence_factors(self, target_time):
        """
        Gebruikt SHAP om uit te leggen WAAROM het model denkt dat er water nodig is.
        """
        if not self.is_fitted or not self.model:
            return {"Status": "Model nog niet getraind"}

        try:
            # 1. Data voorbereiden voor target_time
            df = pd.DataFrame([{"timestamp": target_time}])
            df = add_cyclic_time_features(df)
            X = df.reindex(columns=self.feature_columns).apply(
                pd.to_numeric, errors="coerce"
            )

            # 2. SHAP Berekening
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            # Binary classification handling (we willen de waarden voor klasse 1 = Usage)
            if isinstance(shap_values, list):
                vals = shap_values[1][0]
            elif len(shap_values.shape) == 2:
                vals = shap_values[0]  # Soms is output (1, features)
            else:
                vals = shap_values

            raw_influences = {
                col: float(val) for col, val in zip(self.feature_columns, vals)
            }

            influences = {}

            def format_impact(val):
                if abs(val) < 0.1:
                    return None
                return "Hoog" if val > 0 else "Laag"

            time_val = raw_influences.get("hour_sin", 0) + raw_influences.get(
                "hour_cos", 0
            )
            if abs(time_val) > 0.1:
                influences["Tijdstip"] = f"{format_impact(time_val)} ({time_val:+.2f})"

            day_val = raw_influences.get("day_sin", 0) + raw_influences.get(
                "day_cos", 0
            )
            if abs(day_val) > 0.1:
                influences["Weekdag"] = f"{format_impact(day_val)} ({day_val:+.2f})"

            return influences

        except Exception as e:
            logger.error(f"DhwAI SHAP mislukt: {e}")
            return {}
