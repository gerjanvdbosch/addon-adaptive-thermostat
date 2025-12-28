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
from dataclasses import dataclass

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
    DONE = "DONE"
    LOW_LIGHT = "LOW_LIGHT"


@dataclass
class SolarContext:
    day_peak_kw: float
    seasonal_max_kw: float
    remaining_day_ratio: float
    total_daily_kwh: float
    full_load_hours: float
    future_peak_kw: float
    day_quality_ratio: float
    trigger_threshold_kw: float
    current_pv_kw: float
    threshold_percentage: float
    bias_factor: float


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
        self.min_viable_kw = float(self.opts.get("min_viable_kw", 0.3))
        self.min_noise_kw = float(self.opts.get("min_noise_kw", 0.01))

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

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # --- State ---
        self.model = None
        self.is_fitted = False
        self.smoothed_bias = 1.0

        # Solcast Cache
        self.cached_solcast_data = []
        self.last_solcast_poll_ts = None
        self.solcast_just_updated = False
        self.last_midnight_reset_date = datetime.now().date()

        # Stabiliteits Buffer
        self.history_len = int(300 / max(1, self.interval))
        self.pv_buffer = deque(maxlen=self.history_len)
        self.state_len = 10
        self.state_buffer = deque(maxlen=self.state_len)

        self.last_stable_advice = {
            "action": SolarStatus.WAIT,
            "reason": "Systeem starten",
            "plan_start": None,
        }

        self.current_slot_start = None
        self.slot_samples = []

        self._load_model()

    # ==============================================================================
    # 1. MODEL & HELPERS
    # ==============================================================================

    def _load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
                logger.info("SolarAI: Model geladen.")
            except Exception:
                logger.exception("SolarAI: Laden mislukt.")

    def _atomic_save(self, model):
        tmp_path = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump(model, tmp_path)
            tmp_path.replace(self.model_path)
            logger.info("SolarAI: Model opgeslagen.")
        except Exception:
            logger.exception("SolarAI: Opslaan mislukt.")

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
    # 3. DATA & STATE
    # ==============================================================================

    def _check_midnight_reset(self):
        """Wist de cache zodra er een nieuwe dag begint."""
        today = datetime.now().date()
        if today != self.last_midnight_reset_date:
            logger.info("SolarAI: Nieuwe dag. Cache reset.")
            self.cached_solcast_data = []
            self.last_solcast_poll_ts = None
            self.solcast_just_updated = True
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
                self.solcast_just_updated = True
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
            return float(self.pv_buffer[-1]), True

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

    def _make_result(self, status, reason, plan_time=None, context=None):
        return {
            "action": status,
            "reason": reason,
            "plan_start": plan_time,
            "context": context,
        }

    # ==============================================================================
    # 4. INFERENCE
    # ==============================================================================

    def get_solar_recommendation(self):
        """Genereert advies op basis van 1-minuut geïnterpoleerde data."""

        if not self.cached_solcast_data:
            return {"action": SolarStatus.WAIT, "reason": "Geen Solcast data"}

        # 1. DataFrame Voorbereiden
        df = pd.DataFrame(self.cached_solcast_data)
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)
        df.sort_values("timestamp", inplace=True)

        # Nacht detectie obv data (ipv Astro)
        # We kijken of er in de HELE forecast nog iets > noise zit
        valid_power_df = df[df["pv_estimate"] > self.min_noise_kw]
        if valid_power_df.empty:
            return {"action": SolarStatus.NIGHT, "reason": "Zon onder"}

        # 2. AI Predictie op de ruwe blokken (30 min)
        # We voorspellen EERST, daarna interpoleren we.
        if self.is_fitted and self.model:
            X_pred = self._create_features(df)
            pred_ai = self.model.predict(X_pred)
            # Blending: 60% AI, 40% Solcast
            df["ai_power_raw"] = (0.6 * pred_ai) + (0.4 * df["pv_estimate"])
            df["ai_power_raw"] = df["ai_power_raw"].clip(0, self.system_max_kw)
        else:
            df["ai_power_raw"] = df.get("pv_estimate", 0.0)

        # FIX 3: INTERPOLATIE NAAR 1 MINUUT
        # We resamplen naar 1 minuut om bias-oscillatie te voorkomen.
        df = df.set_index("timestamp")

        # 1. Zorg dat Pandas begrijpt welke kolommen getallen zijn
        # 2. Selecteer alleen de numerieke kolommen om de waarschuwing te voorkomen
        df_numeric = df.select_dtypes(include=[np.number]).infer_objects(copy=False)

        # 3. Resample en Interpoleren
        df = df_numeric.resample("1min").interpolate(method="linear").reset_index()

        now_utc = pd.Timestamp.now(tz="UTC")
        local_tz = datetime.now().astimezone().tzinfo
        median_pv, is_stable = self._get_stability_stats()

        # 3. BIAS BEREKENING (Op exacte minuut)
        # Zoek de rij die overeenkomt met de huidige minuut
        now_floor = now_utc.replace(second=0, microsecond=0)
        current_row = df[df["timestamp"] == now_floor]

        expected_now = 0.0
        if not current_row.empty:
            expected_now = current_row.iloc[0]["ai_power_raw"]

            # Alleen updaten als er significant licht verwacht wordt
            if expected_now > self.min_noise_kw:
                new_bias = median_pv / expected_now
                # Als Solcast net update, vertrouwen we bias iets minder (0.8), anders traag (0.2)
                alpha = 0.8 if self.solcast_just_updated else 0.2

                # Extra demping om flipperen te voorkomen
                self.smoothed_bias = ((1.0 - alpha) * self.smoothed_bias) + (
                    alpha * new_bias
                )
                self.smoothed_bias = np.clip(self.smoothed_bias, 0.4, 1.6)
                self.solcast_just_updated = False

        # Pas bias toe op hele dataset
        df["ai_power"] = (df["ai_power_raw"] * self.smoothed_bias).clip(
            0, self.system_max_kw
        )

        # 4. ROLLING WINDOW (Aangepast voor 1-minuut data)
        # Omdat data nu per minuut is, is de window_size = uren * 60
        window_steps = max(1, int(self.duration_hours * 60))
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_steps)
        df["window_avg_power"] = df["ai_power"].rolling(window=indexer).mean()

        # Filter toekomst
        future = df[df["timestamp"] >= now_floor].copy()
        future.dropna(subset=["window_avg_power"], inplace=True)

        if future.empty:
            return self._make_result(SolarStatus.DONE, "Einde dag forecast", None, None)

        # 5. DAG ANALYSE
        today_start = now_utc.normalize()
        today_end = today_start + timedelta(days=1)
        df_today = df[(df["timestamp"] >= today_start) & (df["timestamp"] < today_end)]

        forecast_peak_day = (
            df_today["window_avg_power"].max() if not df_today.empty else 0.0
        )

        # Bias Logic
        history_bias = min(self.smoothed_bias, 1.0)
        future_bias_raw = self.smoothed_bias

        # FIX 1: DYNAMISCHE TIJDRESOLUTIE
        # Omdat we geresampled hebben naar 1 minuut, is elke rij 1/60e uur.

        # Totale energie: Som van kW / 60 minuten
        remaining_kwh_raw = future["ai_power_raw"].sum() / 60.0
        total_kwh_raw = df_today["ai_power_raw"].sum() / 60.0

        # Ratio berekening (Ruw / Ruw)
        remaining_ratio = 0.0
        if total_kwh_raw > 0.01:
            remaining_ratio = remaining_kwh_raw / total_kwh_raw
            remaining_ratio = max(0.0, min(remaining_ratio, 1.0))

        # Ochtend Optimisme voor bias
        if remaining_ratio > 0.50:
            future_bias_raw = (future_bias_raw * 0.70) + (0.85 * 0.30)

        future_bias = min(future_bias_raw, 1.15)

        # Toekomst Piek
        future_max_raw = future["ai_power_raw"].rolling(window=indexer).mean().max()
        if pd.isna(future_max_raw):
            future_max_raw = 0.0
        adjusted_future_max = future_max_raw * future_bias

        day_peak = max(forecast_peak_day * history_bias, median_pv, adjusted_future_max)

        # Realistische dagtotaal (voor logging)
        daily_kwh = total_kwh_raw * history_bias

        # 6. DREMPEL & SEIZOEN (Met Fix voor SeasonMax)
        doy = now_utc.timetuple().tm_yday
        # AANGEPAST: Winter = 0.70, Zomer = 1.00
        season_factor = 0.85 - 0.15 * math.cos(2 * math.pi * (doy + 10) / 365)
        season_factor = max(0.70, min(season_factor, 1.0))

        seasonal_max_kw = self.system_max_kw * season_factor

        # Kwaliteit
        day_quality_ratio = day_peak / max(0.1, seasonal_max_kw)

        full_load_hours = daily_kwh / max(1.0, self.system_max_kw)
        if full_load_hours < 2.0:
            day_quality_ratio = min(day_quality_ratio, 0.65)
        day_quality_ratio = min(day_quality_ratio, 1.0)

        # Formule (Combinatie Kwaliteit + Geduld)
        percentage = 0.60 + (day_quality_ratio * 0.20) + (remaining_ratio * 0.15)
        # Zomer Plafond
        percentage = max(0.65, min(percentage, 0.90))

        # Triggers
        future_threshold = adjusted_future_max * percentage
        day_floor_limit = day_peak * 0.25
        effective_min_viable = min(self.min_viable_kw, day_peak * 0.90)
        effective_min_viable = max(effective_min_viable, self.min_noise_kw * 2)

        final_trigger_val = max(future_threshold, day_floor_limit, effective_min_viable)

        context = SolarContext(
            day_peak_kw=day_peak,
            seasonal_max_kw=seasonal_max_kw,
            remaining_day_ratio=remaining_ratio,
            total_daily_kwh=daily_kwh,
            full_load_hours=full_load_hours,
            future_peak_kw=adjusted_future_max,
            day_quality_ratio=day_quality_ratio,
            trigger_threshold_kw=final_trigger_val,
            current_pv_kw=median_pv,
            threshold_percentage=percentage,
            bias_factor=self.smoothed_bias,
        )

        # Log labels & DEFINITIES
        day_quality_high = 0.75
        day_quality_average = 0.40

        if day_quality_ratio > day_quality_high:
            day_type = "Sunny ☀️"
        elif day_quality_ratio > day_quality_average:
            day_type = "Average ⛅"
        else:
            day_type = "Gloomy ☁️"

        logger.info(
            f"SolarAI: Piek: {day_peak:.2f}kW (Season-Max: {seasonal_max_kw:.2f}) | Rest: {remaining_ratio:.0%} | "
            f"Totaal: {daily_kwh:.1f}kWh ({full_load_hours:.1f}h) | Ref-Future: {adjusted_future_max:.2f}kW | "
            f"Day-Ratio: {day_quality_ratio:.2f} | Drempel: {final_trigger_val:.2f}kW | "
            f"Actueel: {median_pv:.2f}kW | Percentage: {percentage:.1%} | Bias: {self.smoothed_bias:.2f}"
        )

        # 7. BESLUITVORMING
        max_peak_power = future["window_avg_power"].max()
        threshold_planning = max(max_peak_power * percentage, 0.01)
        candidates = future[future["window_avg_power"] >= threshold_planning]

        if not candidates.empty:
            best_row = candidates.iloc[0]
        else:
            best_idx = future["window_avg_power"].idxmax()
            best_row = future.loc[best_idx]

        best_power = best_row["window_avg_power"]
        start_time_local = best_row["timestamp"].tz_convert(local_tz)
        wait_minutes = int((best_row["timestamp"] - now_utc).total_seconds() / 60)
        wait_hours = wait_minutes / 60.0

        # A. Low Light
        if adjusted_future_max < day_floor_limit and median_pv < day_floor_limit:
            return self._make_result(
                SolarStatus.LOW_LIGHT,
                f"[{day_type}] Te laag vermogen ({adjusted_future_max:.2f}kW)",
                start_time_local,
                context,
            )

        # --- PEAK HUNTING (FIX 4: Absolute Check) ---
        is_waiting_worth_it = False

        # Absolute winst in kW
        absolute_gain_kw = best_power - median_pv
        # Relatieve winst
        potential_gain_pct = (best_power - median_pv) / max(median_pv, 0.01)

        # We wachten alleen als:
        # 1. Er nog genoeg tijd is (>15% dag over)
        # 2. De winst substantieel is (>150 Watt Absoluut) OF (>20% Relatief)
        has_time_left = remaining_ratio > 0.15
        is_substantial_gain = (absolute_gain_kw > 0.15) or (potential_gain_pct > 0.20)

        should_check_waiting = has_time_left and is_substantial_gain

        if should_check_waiting:
            # We gebruiken nu veilig de variabelen die hierboven gedefinieerd zijn
            # En we checken nogmaals de day quality als extra eis voor de 'fijne kneepjes'
            if day_quality_ratio > day_quality_average:
                # REGEL 1: Zijn we er al bijna? (85%)
                if median_pv >= (best_power * 0.85):
                    is_waiting_worth_it = False
                else:
                    # REGEL 2: Time Decay
                    time_penalty_factor = max(0.5, 1.0 - (wait_hours * 0.10))
                    if (best_power * time_penalty_factor) > (median_pv * 1.10):
                        is_waiting_worth_it = True
            else:
                # Op slechte dagen wachten we alleen bij gigantische absolute winst
                if potential_gain_pct > 0.30:
                    is_waiting_worth_it = True

        # B. Opportunisme
        current_slot_forecast_raw = expected_now if "expected_now" in locals() else 0.0

        is_sunny_surprise = median_pv > (current_slot_forecast_raw * 1.20)
        is_viable_run = median_pv > day_floor_limit

        if is_sunny_surprise and is_viable_run and not is_waiting_worth_it:
            return self._make_result(
                SolarStatus.START,
                f"[{day_type}] Opportunisme: Feller dan forecast!",
                datetime.now(local_tz),
                context,
            )

        # C. Normale Drempel
        if median_pv > final_trigger_val:
            if is_waiting_worth_it:
                return self._make_result(
                    SolarStatus.WAIT,
                    f"[{day_type}] Wacht op piek ({best_power:.2f}kW)",
                    start_time_local,
                    context,
                )

            if (
                day_quality_ratio > day_quality_average
                and not is_stable
                and median_pv < (best_power * 0.9)
            ):
                return self._make_result(
                    SolarStatus.WAIT_STABLE,
                    f"[{day_type}] Wachten op stabiel licht",
                    start_time_local,
                    context,
                )

            return self._make_result(
                SolarStatus.START,
                f"[{day_type}] Drempel bereikt ({median_pv:.2f} > {final_trigger_val:.2f})",
                datetime.now(local_tz),
                context,
            )

        # D. Wachten
        wait_msg = f"over {wait_minutes} min" if wait_minutes > 0 else "NU"
        return self._make_result(
            SolarStatus.WAIT,
            f"[{day_type}] Piek {wait_msg} ({best_power:.2f}kW)",
            start_time_local,
            context,
        )

    def run_cycle(self):
        """Wordt elke minuut aangeroepen vanuit de main loop."""
        self._check_midnight_reset()
        now = datetime.now(timezone.utc)

        try:
            pv_val = self.ha.get_state(self.entity_pv)
            pv_kw = (
                float(pv_val) / 1000.0
                if pv_val not in ["unknown", "unavailable", None]
                else 0.0
            )
        except Exception:
            pv_kw = 0.0

        self._process_pv_sample(pv_kw)
        self._update_solcast_cache()

        raw_advice = self.get_solar_recommendation()

        self.state_buffer.append(raw_advice)

        # Buffer Logica (Fix: Altijd updaten bij stabiliteit)
        if len(self.state_buffer) == self.state_len:
            newest_action = self.state_buffer[-1]["action"]
            is_stable = all(
                item["action"] == newest_action for item in self.state_buffer
            )

            if is_stable:
                # ALTIJD updaten, zodat timestamp en reason actueel blijven
                self.last_stable_advice = raw_advice

        final_advice = self.last_stable_advice

        pending_msg = ""
        if raw_advice["action"] != final_advice["action"]:
            pending_msg = f" (Pending: {raw_advice['action'].value})"

        res_str = final_advice["action"].value
        reason = final_advice["reason"]

        p_time = (
            final_advice.get("plan_start").strftime("%H:%M")
            if final_advice.get("plan_start")
            else "--:--"
        )
        iso_date = (
            final_advice.get("plan_start").replace(second=0, microsecond=0).isoformat()
            if final_advice.get("plan_start")
            else "unknown"
        )

        logger.info(
            f"SolarAI: [{res_str}]{pending_msg} {reason} | Plan: {p_time} | Bias: {self.smoothed_bias:.2f}"
        )

        attributes = {
            "status": res_str,
            "reason": reason,
            "planned_start": iso_date,
            "bias_factor": round(self.smoothed_bias, 2),
            "last_update": now.isoformat(),
            "device_class": "timestamp",
        }

        self.ha.set_solar_prediction(iso_date, attributes)

    def get_influence_factors(self, df_row: pd.DataFrame) -> dict:
        if not self.is_fitted or self.model is None:
            return {"Status": "Model nog niet getraind"}

        try:
            X = df_row.reindex(columns=self.feature_columns)
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            base_value = float(
                explainer.expected_value[0]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

            vals = shap_values[0]
            raw_influences = {
                col: float(val) for col, val in zip(self.feature_columns, vals)
            }

            solcast_raw = df_row.iloc[0].get("pv_estimate", 0.0)
            factor = 0.6

            direct_impact = solcast_raw * 0.4
            model_impact = raw_influences.get("pv_estimate", 0) * factor
            uncertainty_impact = (
                raw_influences.get("pv_estimate10", 0)
                + raw_influences.get("pv_estimate90", 0)
                + raw_influences.get("uncertainty", 0)
            ) * factor
            time_impact = (
                raw_influences.get("hour_sin", 0)
                + raw_influences.get("hour_cos", 0)
                + raw_influences.get("doy_sin", 0)
                + raw_influences.get("doy_cos", 0)
            ) * factor

            influences = {
                "Model Basis": f"{base_value * factor:.2f}",
                "Solcast (Direct 40%)": f"{'+' if direct_impact > 0 else ''}{direct_impact:.2f}",
                "Solcast (Model 60%)": f"{'+' if model_impact > 0 else ''}{model_impact:.2f}",
            }

            if abs(uncertainty_impact) > 0.03:
                influences["Onzekerheid"] = (
                    f"{'+' if uncertainty_impact > 0 else ''}{uncertainty_impact:.2f}"
                )
            if abs(time_impact) > 0.03:
                influences["Tijd/Seizoen"] = (
                    f"{'+' if time_impact > 0 else ''}{time_impact:.2f}"
                )

            ai_part = base_value + sum(vals)
            ai_power_blended = (0.6 * ai_part) + (0.4 * solcast_raw)
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
