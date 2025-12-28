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
from db import fetch_dhw_sessions, insert_dhw_session
from utils import add_cyclic_time_features, safe_float

logger = logging.getLogger(__name__)


class SensorPosition(Enum):
    TOP = 1
    BOTTOM = 2


class DhwAI:
    """
    Domestic Hot Water AI.
    - Event-based Logging: Slaat alleen complete douche-sessies op.
    - Session-based Training: Leert van de sessie-tabel.
    - Solar Aware.
    """

    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}

        # Config
        self.model_path = Path(
            self.opts.get("dhw_model_path", "/config/models/dhw_model.joblib")
        )

        # Instellingen
        self.min_temp = float(self.opts.get("dhw_min_temp", 30.0))
        self.target_temp = float(self.opts.get("dhw_target_temp", 50.0))
        self.boost_temp = float(self.opts.get("dhw_boost_temp", 55.0))

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

        # --- SESSION DETECTION STATE (In Memory) ---
        # We kijken 15 minuten terug om een start te detecteren
        self.temp_history = deque(maxlen=16)  # 15 min historie + 1 huidig

        self.in_session = False
        self.session_start_time = None
        self.session_start_temp = None
        self.lowest_temp_seen = None
        self.stable_counter = 0

        # --- Model State ---
        self.model = None
        self.is_fitted = False

        # Buffer voor output stabiliteit (voorkom pendelen WP)
        self.action_buffer = deque(maxlen=3)
        self.last_stable_setpoint = self.min_temp

        # Buffer voor snelle detectie (alleen voor 'is_showering_now' vlaggetje)
        self.temp_buffer = deque(maxlen=5)

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
        Traint het model op basis van de DHW Sessions tabel.
        Dit is veel schoner en sneller dan ruwe data.
        """
        logger.info("DhwAI: Training start (Session Table)...")

        # 1. Haal de sessies op (Events)
        df_sessions = fetch_dhw_sessions(days=60)

        if df_sessions is None or len(df_sessions) < 5:
            logger.warning("DhwAI: Te weinig sessies in DB om te trainen.")
            return

        logger.info(f"DhwAI: {len(df_sessions)} sessies opgehaald.")

        # 2. Maak een lege tijdlijn (Grid) voor de afgelopen 60 dagen (15 min steps)
        #    Gebruik lokale tijd, aangezien fetch_dhw_sessions dat ook doet.
        end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=60)

        grid_index = pd.date_range(start=start_date, end=end_date, freq="15min")
        df_grid = pd.DataFrame(index=grid_index)
        df_grid["is_showering"] = 0

        # 3. Map de sessies op de tijdlijn
        for _, row in df_sessions.iterrows():
            # Rond starttijd af op kwartier
            start = row["start_time"].round("15min")

            if start in df_grid.index:
                df_grid.at[start, "is_showering"] = 1

                # Optioneel: Als sessie > 15 min duurde, markeer ook het volgende blok
                if row["duration_minutes"] > 15:
                    nxt = start + timedelta(minutes=15)
                    if nxt in df_grid.index:
                        df_grid.at[nxt, "is_showering"] = 1

        # 4. Trainen
        df_features = add_cyclic_time_features(df_grid, col_name=None)
        X = df_features.reindex(columns=self.feature_columns)
        y = df_grid["is_showering"]

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
    # 2. SESSION DETECTOR (Event Loop)
    # ==============================================================================

    def run_cycle(self, features, current_hvac):
        """
        Draait elke minuut.
        1. Detecteert of er gedoucht wordt.
        2. Slaat sessie op in DB bij einde.
        3. Bepaalt boiler setpoint.
        """
        current_temp = safe_float(features.get("dhw_temp"))
        solar_action = features.get("solar_action")

        if current_temp is None:
            return

        now = datetime.now()

        # ----------------------------------------------------------------------
        # A. SESSION DETECTION LOGIC
        # ----------------------------------------------------------------------

        # Voeg huidige meting toe aan historie (bewaart tijd en temp)
        self.temp_history.append((now, current_temp))

        # Buffer voor realtime gebruik (korte termijn)
        self.temp_buffer.append(current_temp)

        if not self.in_session:
            # --- STATUS: RUST ---
            # We zoeken naar een 'Start Event'

            # We hebben minimaal 10 minuten historie nodig
            if len(self.temp_history) == self.temp_history.maxlen:
                # Kijk naar 15 min geleden (index 0) vs Nu (laatste)
                past_time, past_temp = self.temp_history[0]

                diff = past_temp - current_temp

                # TRIGGER: > 1.5 graad gedaald in 15 minuten EN WP staat niet op SWW
                # (Als WP op SWW staat, kan temp dippen door mixing, dat is geen douche)
                is_dhw_running = current_hvac == "dhw"

                if diff > 1.5 and not is_dhw_running:
                    logger.info(
                        f"DhwAI: Start sessie! (Gedaald van {past_temp} naar {current_temp})"
                    )
                    self.in_session = True
                    self.session_start_time = (
                        past_time  # Starttijd corrigeren naar begin daling
                    )
                    self.session_start_temp = past_temp
                    self.lowest_temp_seen = current_temp
                    self.stable_counter = 0

        else:
            # --- STATUS: SESSIE BEZIG ---

            # Update dieptepunt
            if current_temp < self.lowest_temp_seen:
                self.lowest_temp_seen = current_temp
                self.stable_counter = 0
            else:
                # Temp daalt niet meer (of stijgt). Tel hoe lang.
                self.stable_counter += 1

            # Check EINDE condities
            stop_reason = None

            # 1. Verwarming springt aan (Sessie kapot/voorbij)
            if current_hvac is not None and current_hvac != 0:
                stop_reason = "Heating started"

            # 2. Temp is al 5 minuten stabiel (of stijgend)
            elif self.stable_counter >= 5:
                stop_reason = "Temp stable"

            if stop_reason:
                # Opslaan!
                insert_dhw_session(
                    start_time=self.session_start_time,
                    end_time=now,
                    start_temp=self.session_start_temp,
                    end_temp=self.lowest_temp_seen,
                )

                duration = (now - self.session_start_time).total_seconds() / 60.0
                drop = self.session_start_tem - self.lowest_temp_seen
                logger.info(
                    f"DhwAI: Douche sessie opgeslagen! (-{drop:.1f}°C in {duration:.1f} min)"
                )

                # Reset
                self.in_session = False
                self.session_start_time = None
                self.session_start_temp = None
                self.lowest_temp_seen = None
                self.stable_counter = 0

        # ----------------------------------------------------------------------
        # B. BESLUITVORMING & ACTIE
        # ----------------------------------------------------------------------
        target = self.get_recommendation(current_temp, solar_action)

        self.action_buffer.append(target)
        if len(self.action_buffer) == 3 and len(set(self.action_buffer)) == 1:
            stable_target = self.action_buffer[0]
            current_dhw_sp = self.ha.get_dhw_setpoint()

            if current_dhw_sp is not None and abs(current_dhw_sp - stable_target) < 1.0:
                return

            self.ha.set_dhw_setpoint(stable_target)
            self.last_stable_setpoint = stable_target

    def get_recommendation(self, current_temp, solar_action=None):
        """
        Bepaalt setpoint op basis van:
        1. Live Gebruik (snel detecteren via buffer)
        2. Veiligheid
        3. Solar
        4. AI Voorspelling
        """
        now = datetime.now()

        # Korte termijn detectie (voor directe reactie tijdens douchen)
        is_showering_now = False
        if len(self.temp_buffer) >= 2:
            if (self.temp_buffer[0] - self.temp_buffer[-1]) >= 0.5:
                is_showering_now = True

        if is_showering_now:
            # Tijdens douchen passief blijven
            pass

        # 1. VEILIGHEID
        if current_temp < self.min_temp:
            logger.info(f"DhwAI: Temp kritiek laag {current_temp:.1f}°C")
            return self.target_temp

        # 2. SOLAR BOOST
        if str(solar_action) == "START":
            if current_temp < self.boost_temp:
                logger.info("DhwAI: Zonnebuffer actief.")
                return self.boost_temp
            else:
                logger.info("DhwAI: Zonnebuffer vol")
                return self.min_temp

        # 3. VOORSPELLING (AI)
        if self.is_fitted:
            future_ts = now + timedelta(minutes=self.lookahead_minutes)

            df_fut = pd.DataFrame([{"timestamp": future_ts}])
            df_fut = add_cyclic_time_features(df_fut)
            X_fut = df_fut.reindex(columns=self.feature_columns).apply(
                pd.to_numeric, errors="coerce"
            )

            prob = self.model.predict_proba(X_fut)[0][1]

            if prob > self.confidence_threshold:
                if current_temp < self.target_temp:
                    logger.info(
                        f"DhwAI: Verwacht gebruik om {future_ts.strftime('%H:%M')} ({prob:.0%})"
                    )
                    return self.target_temp

        return self.min_temp

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
