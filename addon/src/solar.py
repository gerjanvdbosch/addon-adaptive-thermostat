import logging
import joblib
import numpy as np
import pandas as pd
import shap
import math

from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
from enum import Enum
from utils import add_cyclic_time_features

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from pandas.api.types import is_datetime64_any_dtype

# Project Imports
from db import fetch_solar_training_data_orm, upsert_solar_record
from ha_client import HAClient

logger = logging.getLogger(__name__)


class SolarStatus(Enum):
    START = "START"
    WAIT = "WAIT"
    WAIT_STABLE = "WAIT_STABLE"
    NIGHT = "NIGHT"
    DONE = "DONE"  # Dag is voorbij, geen forecast meer
    LOW_LIGHT = "LOW_LIGHT"  # Wel dag, maar te weinig licht om iets te doen


class SolarAI:
    """
    SolarAI: Voorspelt het ideale moment voor energie-intensieve taken (zoals SWW).
    Maakt gebruik van Solcast data met een lokale Machine Learning bias-correctie.
    """

    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}

        # --- Config ---
        self.model_path = Path(
            self.opts.get("solar_model_path", "/config/models/solar_model.joblib")
        )

        # Feature definitie
        self.feature_columns = [
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "pv_estimate",
            "pv_estimate10",
            "pv_estimate90",
            "uncertainty",
        ]

        # Systeem instellingen
        self.system_max_kw = float(self.opts.get("system_max_kw", 2.0))
        self.duration_hours = float(self.opts.get("duration_hours", 1.0))
        self.aggregation_minutes = int(self.opts.get("aggregation_minutes", 30))

        # Logica drempels
        self.early_start_threshold = float(self.opts.get("early_start_threshold", 0.95))
        self.min_viable_kw = float(
            self.opts.get("min_viable_kw", 0.3)
        )  # Onder 300W = LOW_LIGHT
        self.min_noise_kw = float(
            self.opts.get("min_noise_kw", 0.01)
        )  # Onder 10W = NIGHT/Ruis

        self.interval = int(self.opts.get("solar_interval_seconds", 15))

        # Sensoren
        self.entity_pv = self.opts.get(
            "sensor_pv_power", "sensor.fuj7chn07b_pv_output_actual"
        )
        self.entity_solcast = self.opts.get(
            "sensor_solcast", "sensor.solcast_pv_forecast_forecast_today"
        )
        self.entity_solcast_poll = self.opts.get(
            "sensor_solcast_poll", "sensor.solcast_pv_forecast_api_last_polled"
        )

        # Zorg dat de map bestaat
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # --- State ---
        self.model = None
        self.is_fitted = False
        self.smoothed_bias = 1.0  # Startwaarde voor bias correctie

        # Solcast Cache
        self.cached_solcast_data = []
        self.last_solcast_poll_ts = None
        self.last_midnight_reset_date = datetime.now().date()

        # Stabiliteits Buffer
        self.history_len = int(600 / max(1, self.interval))
        self.pv_buffer = deque(maxlen=self.history_len)

        # Slot Aggregatie
        self.current_slot_start = None
        self.slot_samples = []

        self._load_model()

    # ==============================================================================
    # 1. MODEL MANAGEMENT & HELPERS
    # ==============================================================================

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
                logger.info("SolarAI: Model geladen van schijf.")
            except Exception:
                logger.exception("SolarAI: Laden van model mislukt.")

    def _atomic_save(self, model):
        tmp_path = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump(model, tmp_path)
            tmp_path.replace(self.model_path)
            logger.info("SolarAI: Model succesvol opgeslagen.")
        except Exception:
            logger.exception("SolarAI: Opslaan van model mislukt.")

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maakt features voor training en voorspelling."""
        df = df.copy()

        if not is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        df = add_cyclic_time_features(df, col_name="timestamp")

        # Onzekerheid (Verschil tussen Solcast P10 en P90)
        df["uncertainty"] = df["pv_estimate90"] - df["pv_estimate10"]

        df_out = df.reindex(columns=self.feature_columns)
        df_out = df_out.apply(pd.to_numeric, errors="coerce")

        return df_out

    # ==============================================================================
    # 2. TRAINING
    # ==============================================================================

    def train(self):
        """Traint het model op historische data."""
        logger.info("SolarAI: Training start...")

        df = fetch_solar_training_data_orm(days=180)

        if len(df) < 2000:
            logger.warning(
                f"SolarAI: Te weinig data ({len(df)} samples). Training overgeslagen."
            )
            return

        X = self._create_features(df)
        y = df["actual_pv_yield"].clip(0, self.system_max_kw)

        # Filter NaN's
        mask = np.isfinite(y)
        X, y = X[mask], y[mask]

        # Histogram Gradient Boosting (Robuust en snel)
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",  # TOP: Erg robuust tegen uitschieters
            learning_rate=0.05,  # TOP: Rustig leerproces
            max_iter=2000,  # Goed: Early stopping doet de rest
            max_leaf_nodes=31,  # Goed: Voorkomt te complexe bomen
            min_samples_leaf=20,  # Goed: Voorkomt regels op basis van te weinig data
            l2_regularization=1.0,  # TOP: Houdt de gewichten bescheiden
            early_stopping=True,  # Noodzakelijk bij 2000 iteraties
            validation_fraction=0.15,  # Laat Scikit-Learn ZELF 15% apart houden
            n_iter_no_change=20,  # Geduld voordat hij stopt
            random_state=42,  # Zorgt voor reproduceerbare resultaten
        )

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            self._atomic_save(self.model)
            mae = mean_absolute_error(y, self.model.predict(X))
            logger.info(f"SolarAI: Training voltooid MAE={mae:.3f}")
        except Exception:
            logger.exception("SolarAI: Training gefaald.")

    # ==============================================================================
    # 3. DATA VERWERKING (Runtime)
    # ==============================================================================

    def _check_midnight_reset(self):
        """Wist de cache zodra er een nieuwe dag begint."""
        today = datetime.now().date()
        if today != self.last_midnight_reset_date:
            logger.info("SolarAI: Nieuwe dag. Cache reset.")
            self.cached_solcast_data = []
            self.last_solcast_poll_ts = None
            self.last_midnight_reset_date = today

    def _update_solcast_cache(self):
        """Haalt nieuwe voorspellingen op uit HA en slaat ze op in de DB."""
        try:
            poll_state = self.ha.get_state(self.entity_solcast_poll)
            if not poll_state or poll_state in ["unknown", "unavailable"]:
                logger.warning("SolarAI: Solcast poll sensor onbereikbaar of onbekend.")
                return

            current_poll_ts = poll_state
            if current_poll_ts == self.last_solcast_poll_ts:
                logger.debug("SolarAI: Solcast data is up-to-date.")
                return

            payload = self.ha.get_payload(self.entity_solcast)
            if (
                payload
                and "attributes" in payload
                and "detailedForecast" in payload["attributes"]
            ):
                raw_data = payload["attributes"]["detailedForecast"]
                self.cached_solcast_data = raw_data

                # Opslaan in DB voor toekomstige training
                for item in raw_data:
                    ts = (
                        pd.to_datetime(item["period_start"])
                        .tz_convert("UTC")
                        .to_pydatetime()
                    )
                    upsert_solar_record(
                        ts,
                        pv_estimate=item.get("pv_estimate", 0.0),
                        pv_estimate10=item.get("pv_estimate10", 0.0),
                        pv_estimate90=item.get("pv_estimate90", 0.0),
                    )

                self.last_solcast_poll_ts = current_poll_ts
                logger.info("SolarAI: Solcast cache vernieuwd")
            else:
                logger.warning("SolarAI: Solcast sensor data onvolledig.")

        except Exception:
            logger.exception("SolarAI: Error tijdens Solcast update.")

    def _process_pv_sample(self, pv_kw):
        """Verwerkt de huidige meting en slaat periodiek op."""
        now = datetime.now(timezone.utc)
        self.pv_buffer.append(pv_kw)

        slot_minute = (
            now.minute // self.aggregation_minutes
        ) * self.aggregation_minutes
        slot_start = now.replace(minute=slot_minute, second=0, microsecond=0)

        if self.current_slot_start and slot_start > self.current_slot_start:
            if self.slot_samples:
                avg_pv = float(np.mean(self.slot_samples))
                upsert_solar_record(self.current_slot_start, actual_pv_yield=avg_pv)
            self.slot_samples = []

        self.current_slot_start = slot_start
        self.slot_samples.append(pv_kw)

    def _get_stability_stats(self):
        """
        Bepaalt de stabiliteit van het weer.
        Schaalt dynamisch mee met SYSTEM_MAX_KW.
        """
        # Situatie 1: Buffer is helemaal leeg (zou niet moeten kunnen na process_sample)
        if not self.pv_buffer:
            return 0.0, True

        # Situatie 2: Opstartfase (minder dan 5 metingen)
        # Gebruik DIRECT de laatste meting (Huidige waarde) in plaats van 0.0
        if len(self.pv_buffer) < 5:
            current_val = float(self.pv_buffer[-1])
            return current_val, True

        # Situatie 3: Normaal bedrijf (Genoeg data voor mediaan)
        arr = np.array(self.pv_buffer)
        median = float(np.median(arr))

        # IQR = Interkwartielafstand (verschil tussen 75% en 25% punt)
        # Dit geeft aan hoe hard de waarden heen en weer springen
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)

        high_power_limit = self.system_max_kw * 0.4
        if median > high_power_limit:
            threshold = self.system_max_kw * 0.20
        else:
            threshold = self.system_max_kw * 0.10

        threshold = max(threshold, 0.15)
        is_stable = iqr < threshold

        return median, is_stable

    # ==============================================================================
    # 4. INFERENCE
    # ==============================================================================

    def get_solar_recommendation(self):
        """
        Analyseert de forecast.
        Drempels schalen mee met de Piek (kW) én de Totale Opbrengst (kWh).
        Inclusief VEILIGE seizoenscorrectie.
        """
        if not self.cached_solcast_data:
            return {"action": SolarStatus.WAIT, "reason": "Geen Solcast data"}

        # 1. DATAFRAME & NACHT CHECK
        df = pd.DataFrame(self.cached_solcast_data)
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)
        df.sort_values("timestamp", inplace=True)

        df = df[df["pv_estimate"] > self.min_noise_kw]
        if df.empty:
            return {"action": SolarStatus.NIGHT, "reason": "Zon is onder"}

        now_utc = pd.Timestamp.now(tz="UTC")
        local_tz = datetime.now().astimezone().tzinfo

        # 2. AI PREDICTIE & BIAS
        if self.is_fitted and self.model:
            X_pred = self._create_features(df)
            pred_ai = self.model.predict(X_pred)
            # Blending: 60% AI, 40% Solcast
            df["ai_power_raw"] = (0.6 * pred_ai) + (0.4 * df["pv_estimate"])
            df["ai_power_raw"] = df["ai_power_raw"].clip(0, self.system_max_kw)
        else:
            df["ai_power_raw"] = df.get("pv_estimate", 0.0)

        median_pv, is_stable = self._get_stability_stats()

        # Update Bias logic (verkort weergegeven)
        df["time_diff"] = (df["timestamp"] - now_utc).abs()
        nearest_idx = df["time_diff"].idxmin()

        # Update Bias
        if df.loc[nearest_idx, "time_diff"] < pd.Timedelta(minutes=45):
            expected_now = df.loc[nearest_idx, "ai_power_raw"]
            if expected_now > self.min_noise_kw:
                new_bias = median_pv / expected_now
                # EWMA smoothing
                self.smoothed_bias = (0.8 * self.smoothed_bias) + (0.2 * new_bias)
                self.smoothed_bias = np.clip(self.smoothed_bias, 0.4, 1.6)

        df["ai_power"] = (df["ai_power_raw"] * self.smoothed_bias).clip(
            0, self.system_max_kw
        )

        # 3. ROLLING WINDOW
        window_steps = max(1, int(self.duration_hours * 2))
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_steps)

        # We gebruiken de biased power voor het plannen
        df["window_avg_power"] = df["ai_power"].rolling(window=indexer).mean()

        # Filter: Alleen toekomst
        future = df[df["timestamp"] >= (now_utc - timedelta(minutes=15))].copy()
        future.dropna(subset=["window_avg_power"], inplace=True)

        if future.empty:
            return {"action": SolarStatus.DONE, "reason": "Einde dag forecast"}

        # ----------------------------------------------------------------------
        # 4. DAG ANALYSE (PIEK & TOTAAL)
        # ----------------------------------------------------------------------

        today_start = now_utc.normalize()
        today_end = today_start + timedelta(days=1)
        df_today = df[(df["timestamp"] >= today_start) & (df["timestamp"] < today_end)]

        # A. Piek vermogen (kW)
        forecast_peak_day = (
            df_today["window_avg_power"].max() if not df_today.empty else 0.0
        )

        # SPLIT BIAS:
        # Historie mag niet kunstmatig verhoogd worden (max 1.0).
        # Toekomst mag wel optimistisch zijn voor Peak Hunting (max 1.15).
        history_bias = min(self.smoothed_bias, 1.0)
        future_bias = min(self.smoothed_bias, 1.15)

        # Toekomst Piek (Adjusted)
        future_max_raw = future["ai_power_raw"].rolling(window=indexer).mean().max()
        if pd.isna(future_max_raw):
            future_max_raw = 0.0
        adjusted_future_max = future_max_raw * future_bias

        # Dag Piek (Het hoogste van: Voorspelling(safe), Actueel, Toekomst)
        day_peak = max(forecast_peak_day * history_bias, median_pv, adjusted_future_max)

        # Totale energie (kWh) - Gebruik history_bias om overschatting te voorkomen
        daily_kwh = df_today["ai_power_raw"].sum() * 0.5 * history_bias

        # ----------------------------------------------------------------------
        # 5. DYNAMISCHE DREMPEL (MET SEIZOENSCORRECTIE)
        # ----------------------------------------------------------------------

        # Bereken Seizoensfactor (0.35 in winter, 1.0 in zomer)
        doy = now_utc.timetuple().tm_yday
        # Cosinus golf: Piek op dag 172 (21 juni), Dal op dag 355 (21 dec)
        # Formule: 0.75 - 0.25 (bereik 0.5 - 1.0) -> Zuiden
        season_factor = 0.75 - 0.25 * math.cos(2 * math.pi * (doy + 10) / 365)
        season_factor = max(0.50, min(season_factor, 1.0))

        # B. Wat is fysiek mogelijk VANDAAG?
        seasonal_max_kw = self.system_max_kw * season_factor

        # C. Bereken kwaliteit relatief aan het seizoen (Ratio 0.0 - 1.0)
        day_quality_ratio = day_peak / max(0.1, seasonal_max_kw)

        # Correctie: Flitsdagen (weinig totaal kWh) mogen geen top-ratio hebben
        full_load_hours = daily_kwh / max(1.0, self.system_max_kw)
        if full_load_hours < 2.0:
            # Op flitsdagen straffen we de ratio af, zodat we niet te streng worden
            day_quality_ratio = min(day_quality_ratio, 0.65)

        day_quality_ratio = min(day_quality_ratio, 1.0)

        # D. Interpolatie van de drempel (Percentage)
        # Basis: 55% tot 90%
        percentage = 0.55 + (day_quality_ratio * 0.35)

        # E. WINTER DEMPING (optioneel als de drempel te hoog ligt)
        # Als het winter is (season_factor < 0.6), cappen we het percentage op 80%.
        # Dit voorkomt dat we op een 'perfecte winterdag' (1.2kW) gaan wachten op
        # die laatste 100 Watt die misschien net niet komt.
        # if season_factor < 0.6:
        #     percentage = min(percentage, 0.80)

        day_quality_high = 0.75
        day_quality_average = 0.4

        # Label voor logging
        if day_quality_ratio > day_quality_high:
            day_type = "Sunny ☀️"
        elif day_quality_ratio > day_quality_average:
            day_type = "Average ⛅"
        else:
            day_type = "Gloomy ☁️"

        # Drempels
        future_threshold = adjusted_future_max * percentage
        day_floor_limit = day_peak * 0.25

        # Effectieve min viable (nooit hoger dan 90% van de dagpiek)
        effective_min_viable = min(self.min_viable_kw, day_peak * 0.90)
        effective_min_viable = max(effective_min_viable, self.min_noise_kw * 2)

        final_trigger_val = max(future_threshold, day_floor_limit, effective_min_viable)

        logger.info(
            f"SolarAI: Piek: {day_peak:.2f}kW (SeasonMax: {seasonal_max_kw:.2f}) | Totaal: {daily_kwh:.1f}kWh ({full_load_hours:.1f}h) | Ref-Future: {adjusted_future_max:.2f}kW | "
            f"Ratio: {day_quality_ratio:.2f} | Drempel: {final_trigger_val:.2f}kW | Actueel: {median_pv:.2f}kW"
        )

        # ----------------------------------------------------------------------
        # 6. BESLUITVORMING
        # ----------------------------------------------------------------------

        # Peak Hunting instellingen
        max_peak_power = future["window_avg_power"].max()
        threshold_planning = max(max_peak_power * self.early_start_threshold, 0.01)

        # Vind alle tijdstippen die aan de eis voldoen voor planning
        candidates = future[future["window_avg_power"] >= threshold_planning]

        # 95% Regeling: Pak het eerste moment dat bijna optimaal is
        if not candidates.empty:
            best_row = candidates.iloc[0]
        else:
            best_idx = future["window_avg_power"].idxmax()
            best_row = future.loc[best_idx]

        # Gebruik de 'raw' index alleen voor de wachttijd berekening (optioneel, maar hier gebruiken we de candidates)
        best_power = best_row["window_avg_power"]

        start_time_local = best_row["timestamp"].tz_convert(local_tz)
        wait_minutes = int((best_row["timestamp"] - now_utc).total_seconds() / 60)
        wait_hours = wait_minutes / 60.0

        # A. Low Light (onder dag-vloer)
        if adjusted_future_max < day_floor_limit and median_pv < effective_min_viable:
            return {
                "action": SolarStatus.LOW_LIGHT,
                "reason": f"[{day_type}] Verwachting ({adjusted_future_max:.2f}kW) te laag",
                "plan_start": start_time_local,
            }

        # --- PEAK HUNTING ---
        is_waiting_worth_it = False

        if day_quality_ratio > day_quality_average:
            # REGEL 1: Zijn we er al bijna? (85%)
            if median_pv >= (best_power * 0.85):
                is_waiting_worth_it = False
            else:
                # REGEL 2: Time Decay
                time_penalty_factor = max(0.5, 1.0 - (wait_hours * 0.10))
                discounted_future_power = best_power * time_penalty_factor

                if discounted_future_power > (median_pv * 1.10):
                    is_waiting_worth_it = True

        # B. Opportunisme
        current_slot_forecast_raw = df.loc[nearest_idx, "ai_power_raw"]
        is_sunny_surprise = median_pv > (current_slot_forecast_raw * 1.20)
        is_viable_now = median_pv >= effective_min_viable

        if is_sunny_surprise and is_viable_now and not is_waiting_worth_it:
            return {
                "action": SolarStatus.START,
                "reason": f"[{day_type}] Opportunisme: Feller dan forecast!",
                "plan_start": datetime.now(local_tz),
            }

        # C. Normale Drempel
        if median_pv >= final_trigger_val or (
            is_viable_now and adjusted_future_max < 0.1
        ):
            if is_waiting_worth_it:
                return {
                    "action": SolarStatus.WAIT,
                    "reason": f"[{day_type}] Wacht op piek ({best_power:.2f}kW)",
                    "plan_start": start_time_local,
                }

            if (
                day_quality_ratio > day_quality_average
                and not is_stable
                and median_pv < (best_power * 0.9)
            ):
                return {
                    "action": SolarStatus.WAIT_STABLE,
                    "reason": f"[{day_type}] Wachten op stabiel licht",
                    "plan_start": start_time_local,
                }

            return {
                "action": SolarStatus.START,
                "reason": f"[{day_type}] Drempel bereikt ({median_pv:.2f} >= {final_trigger_val:.2f})",
                "plan_start": datetime.now(local_tz),
            }

        # D. Wachten
        if wait_minutes > 0:
            wait_msg = f"over {wait_minutes} min"
        elif wait_minutes < 0:
            wait_msg = f"{abs(wait_minutes)} min geleden"
        else:
            wait_msg = "NU"

        return {
            "action": SolarStatus.WAIT,
            "reason": f"[{day_type}] Piek {wait_msg} ({best_power:.2f}kW)",
            "plan_start": start_time_local,
        }

    def run_cycle(self):
        """Wordt elke minuut aangeroepen vanuit de main loop."""
        self._check_midnight_reset()

        now = datetime.now(timezone.utc)

        # 1. Haal PV waarde op
        try:
            pv_val = self.ha.get_state(self.entity_pv)
            pv_kw = (
                float(pv_val) / 1000.0
                if pv_val not in ["unknown", "unavailable", None]
                else 0.0
            )
        except Exception:
            pv_kw = 0.0

        # 2. Update stats & cache
        self._process_pv_sample(pv_kw)
        self._update_solcast_cache()

        # 3. Krijg advies en log het
        advice = self.get_solar_recommendation()

        # Haal de string waarde uit de Enum
        res_enum = advice["action"]
        res_str = res_enum.value
        reason = advice["reason"]
        p_time = (
            advice.get("plan_start").strftime("%H:%M")
            if advice.get("plan_start")
            else "--:--"
        )

        logger.info(
            f"SolarAI: [{res_str}] {reason} | Plan: {p_time} | Bias: {self.smoothed_bias:.2f}"
        )

        self.ha.set_solar_prediction(
            f"{res_str}: {reason}",
            {
                "status": res_str,
                "reason": reason,
                "planned_start": (
                    advice.get("plan_start").isoformat()
                    if advice.get("plan_start")
                    else None
                ),
                "bias_factor": round(self.smoothed_bias, 2),
                "last_update": now.isoformat(),
            },
        )

    def get_influence_factors(self, df_row: pd.DataFrame) -> dict:
        """
        Gebruikt SHAP om de AI voorspelling te verklaren, en verrekent daarna
        de Blending (60/40) en de Bias-correctie voor het complete plaatje.
        """
        if not self.is_fitted or self.model is None:
            return {
                "Solcast": df_row.iloc[0].get("pv_estimate", 0),
                "Status": "Model nog niet getraind",
            }

        try:
            # 1. Data voorbereiden
            X = df_row.reindex(columns=self.feature_columns)
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            # 2. SHAP Berekening (Verklaart de 60% AI kant)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            # Base value = wat het model gemiddeld doet (zonder input)
            base_value = float(
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

            vals = shap_values[0]
            raw_influences = {
                col: float(val) for col, val in zip(self.feature_columns, vals)
            }

            # 3. Reconstructie van de waardes voor de Blending
            solcast_raw = df_row.iloc[0].get("pv_estimate", 0.0)

            # De AI telt maar voor 60% mee in het eindresultaat
            factor = 0.6

            # 4. Groeperen en Schalen

            # A. Solcast Invloed (Het totaal van Direct + Model)
            # - Direct: 40% van de ruwe voorspelling
            # - Model: 60% van wat het model vond van de 'pv_estimate' feature
            direct_impact = solcast_raw * 0.4
            model_impact = raw_influences.get("pv_estimate", 0) * factor

            # B. Onzekerheid (Cloud spread etc)
            uncertainty_impact = (
                raw_influences.get("pv_estimate10", 0)
                + raw_influences.get("pv_estimate90", 0)
                + raw_influences.get("uncertainty", 0)
            ) * factor

            # C. Tijd & Seizoen
            time_impact = (
                raw_influences.get("hour_sin", 0)
                + raw_influences.get("hour_cos", 0)
                + raw_influences.get("doy_sin", 0)
                + raw_influences.get("doy_cos", 0)
            ) * factor

            # 5. Resultaten samenstellen
            influences = {
                # Basislijn van het model (ook geschaald)
                "Model Basis": f"{base_value * factor:.2f}",
                # We tonen de Solcast invloed nu als twee delen om de 'Blending' te tonen
                "Solcast (Direct 40%)": f"{'+' if direct_impact > 0 else ''}{direct_impact:.2f}",
                "Solcast (Model 60%)": f"{'+' if model_impact > 0 else ''}{model_impact:.2f}",
            }

            # Drempel voor ruis (kleine getallen weglaten)
            threshold = 0.03

            if abs(uncertainty_impact) > threshold:
                influences["Onzekerheid Correctie"] = (
                    f"{'+' if uncertainty_impact > 0 else ''}{uncertainty_impact:.2f}"
                )

            if abs(time_impact) > threshold:
                influences["Tijd/Seizoen Correctie"] = (
                    f"{'+' if time_impact > 0 else ''}{time_impact:.2f}"
                )

            # 6. De Bias Correctie (Post-Processing)
            # Eerst berekenen we wat de totale voorspelling (Blended) is vóór bias
            # Reconstructie: (0.6 * (Base + SHAP_Sum)) + (0.4 * Raw)
            ai_part = base_value + sum(vals)
            ai_power_blended = (0.6 * ai_part) + (0.4 * solcast_raw)

            # Dan berekenen we het effect van de bias
            final_power = ai_power_blended * self.smoothed_bias
            bias_kw_effect = final_power - ai_power_blended

            if abs(bias_kw_effect) > 0.01:
                pct = (self.smoothed_bias - 1.0) * 100
                influences["Bias Effect"] = (
                    f"{'+' if bias_kw_effect > 0 else ''}{bias_kw_effect:.2f} ({pct:+.0f}%)"
                )

            return influences

        except Exception as e:
            logger.error(f"SolarAI: SHAP berekening mislukt: {e}")
            return {"Error": "Calculation failed"}
