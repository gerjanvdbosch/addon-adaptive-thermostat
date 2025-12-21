import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from pandas.api.types import is_datetime64_any_dtype

# Project Imports
from db import fetch_solar_training_data_orm, upsert_solar_record
from ha_client import HAClient

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATIE & CONSTANTEN
# ==============================================================================
SYSTEM_MAX_KW = 2.0  # Max vermogen van je omvormer
SWW_DURATION_HOURS = 1.0  # Hoe lang duurt een run voor warm water?
AGGREGATION_MINUTES = 30  # Hoe vaak slaan we een record op in de DB?


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

        # Stabiliteits Buffer (Historie van laatste 10 min)
        self.history_len = int(600 / max(1, self.interval))
        self.pv_buffer = deque(maxlen=self.history_len)

        # Slot Aggregatie (voor DB opslag)
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

        # Cyclische Tijd Features
        hours = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        doy = df["timestamp"].dt.dayofyear

        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 366)

        # Onzekerheid (Verschil tussen Solcast P10 en P90)
        df["uncertainty"] = df["pv_estimate90"] - df["pv_estimate10"]

        features = [
            "pv_estimate",
            "pv_estimate10",
            "pv_estimate90",
            "uncertainty",
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
        ]
        return df[features]

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
        y = df["actual_pv_yield"].clip(0, SYSTEM_MAX_KW)

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

        # Aggregatie per blok van AGGREGATION_MINUTES
        slot_minute = (now.minute // AGGREGATION_MINUTES) * AGGREGATION_MINUTES
        slot_start = now.replace(minute=slot_minute, second=0, microsecond=0)

        if self.current_slot_start and slot_start > self.current_slot_start:
            if self.slot_samples:
                avg_pv = float(np.mean(self.slot_samples))
                upsert_solar_record(self.current_slot_start, actual_pv_yield=avg_pv)
            self.slot_samples = []

        self.current_slot_start = slot_start
        self.slot_samples.append(pv_kw)

    def _get_stability_stats(self):
        """Bepaalt de stabiliteit van het weer."""
        if len(self.pv_buffer) < 5:
            return 0.0, True

        arr = np.array(self.pv_buffer)
        median = float(np.median(arr))
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)

        # Als IQR > 0.4 kW bij helder weer, is het wisselend bewolkt
        threshold = 0.4 if median > 0.8 else 0.2
        is_stable = iqr < threshold

        return median, is_stable

    # ==============================================================================
    # 4. INFERENCE (Het "SWW Moment" bepalen)
    # ==============================================================================

    def get_solar_recommendation(self):
        """
        Analyseert de forecast en de huidige bias om een start-advies te geven.
        """
        if not self.cached_solcast_data:
            return {"action": "WAIT", "reason": "Wachten op Solcast data..."}

        # 1. Prepareer DataFrame
        df = pd.DataFrame(self.cached_solcast_data)
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)
        df.rename(
            columns={
                "pv_estimate": "pv_estimate",
                "pv_estimate10": "pv_estimate10",
                "pv_estimate90": "pv_estimate90",
            },
            inplace=True,
        )
        df.sort_values("timestamp", inplace=True)

        # 2. AI Predictie
        if self.is_fitted and self.model:
            X_pred = self._create_features(df)
            df["ai_power_raw"] = self.model.predict(X_pred)
            df["ai_power_raw"] = df["ai_power_raw"].clip(0, SYSTEM_MAX_KW)
        else:
            # Fallback als model nog niet getraind is
            df["ai_power_raw"] = df.get("pv_estimate", 0.0)

        # 3. Smoothed Bias Correction
        now_utc = pd.Timestamp.now(tz="UTC")
        median_pv, is_stable = self._get_stability_stats()

        # Zoek het dichtstbijzijnde forecast punt
        df["time_diff"] = (df["timestamp"] - now_utc).abs()
        nearest_idx = df["time_diff"].idxmin()

        if df.loc[nearest_idx, "time_diff"] < pd.Timedelta(minutes=45):
            expected_now = df.loc[nearest_idx, "ai_power_raw"]

            if expected_now > 0.1:
                new_bias = median_pv / expected_now
                # EWMA: 80% historie, 20% nieuwe meting (voorkomt schokken)
                self.smoothed_bias = (0.8 * self.smoothed_bias) + (0.2 * new_bias)
                self.smoothed_bias = np.clip(self.smoothed_bias, 0.4, 1.6)

            df["ai_power"] = (df["ai_power_raw"] * self.smoothed_bias).clip(
                0, SYSTEM_MAX_KW
            )
        else:
            df["ai_power"] = df["ai_power_raw"]

        # We kijken naar de piek van VANDAAG (vanaf middernacht tot middernacht)
        today_start = now_utc.normalize()
        today_end = today_start + timedelta(days=1)

        df_today = df[(df["timestamp"] >= today_start) & (df["timestamp"] < today_end)]

        if df_today.empty:
            today_peak = 0.0
        else:
            today_peak = df_today["ai_power"].max()

        # Classificeer de dag
        # We gebruiken SYSTEM_MAX_KW (2.0) als referentie
        day_quality_ratio = today_peak / max(1.0, SYSTEM_MAX_KW)

        if day_quality_ratio > 0.75:
            day_type = "Zonnig"
            # Op een topdag wachten we tot we minimaal 70% van de piek hebben
            dynamic_threshold = today_peak * 0.7
        elif day_quality_ratio > 0.4:
            day_type = "Bewolkt"
            # Op een gemiddelde dag zijn we tevreden met 60% van de piek
            dynamic_threshold = today_peak * 0.6
        else:
            day_type = "Donker"
            # Op een donkere dag pakken we alles wat we pakken kunnen (90% van die kleine piek)
            dynamic_threshold = today_peak * 0.9

        # HARDE ONDERGRENS:
        # Een warmtepompboiler heeft minimaal ~400-500W nodig voor de compressor.
        # Draaien onder de 450W heeft geen zin (dan trek je alles uit het net).
        min_technical_limit = 0.45

        # We stellen de drempel in, maar nooit lager dan wat technisch nuttig is.
        final_trigger_val = max(dynamic_threshold, min_technical_limit)

        # ==========================================================================

        # 4. Zoek beste window (Rolling Average)
        # Solcast is per 30 min, dus SWW_DURATION_HOURS * 2
        window_steps = max(1, int(SWW_DURATION_HOURS * 2))
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_steps)
        df["window_avg_power"] = df["ai_power"].rolling(window=indexer).mean()

        # 5. Filter op de toekomst
        future = df[df["timestamp"] >= (now_utc - timedelta(minutes=15))].copy()
        if future.empty:
            return {"action": "WAIT", "reason": "Einde dag."}

        best_row = future.loc[future["window_avg_power"].idxmax()]
        best_power = best_row["window_avg_power"]

        # Tijdstippen
        local_tz = datetime.now().astimezone().tzinfo
        start_time_local = best_row["timestamp"].tz_convert(local_tz)

        # 6. Besluitvorming met de NIEUWE dynamische logica

        # A. Is de beste piek van vandaag goed genoeg voor onze dynamische drempel?
        if best_power < final_trigger_val:
            # Specifieke reden geven voor loggen
            if best_power < min_technical_limit:
                msg = f"Te weinig zon (Max {best_power:.2f}kW < Min {min_technical_limit}kW)"
            else:
                msg = f"Sparen voor beter moment (Max {best_power:.2f}kW < Drempel {final_trigger_val:.2f}kW)"

            return {
                "action": "WAIT",
                "reason": f"[{day_type}] {msg}",
                "plan_start": start_time_local,
            }

        # B. Zitten we in het ideale blok? (-15min marge)
        if (
            (best_row["timestamp"] - timedelta(minutes=15))
            <= now_utc
            < (best_row["timestamp"] + timedelta(minutes=30))
        ):
            if not is_stable and median_pv < 1.0:
                return {
                    "action": "WAIT_CLOUD",
                    "reason": f"[{day_type}] Wachten op stabieler zonlicht",
                    "plan_start": start_time_local,
                }

            return {
                "action": "START",
                "reason": f"[{day_type}] Optimaal venster! ({best_power:.2f}kW >= {final_trigger_val:.2f}kW)",
                "plan_start": start_time_local,
            }

        # C. Te vroeg?
        if now_utc < best_row["timestamp"]:
            wait_min = int((best_row["timestamp"] - now_utc).total_seconds() / 60)
            return {
                "action": "WAIT",
                "reason": f"[{day_type}] Piek over {wait_min} min ({best_power:.2f}kW)",
                "plan_start": start_time_local,
            }

        return {
            "action": "WAIT",
            "reason": f"[{day_type}] Piek reeds gepasseerd",
            "plan_start": start_time_local,
        }

    def run_cycle(self):
        """Wordt elke minuut aangeroepen vanuit de main loop."""
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

        res = advice["action"]
        reason = advice["reason"]
        p_time = (
            advice.get("plan_start").strftime("%H:%M")
            if advice.get("plan_start")
            else "--:--"
        )

        logger.info(
            f"SolarAI: [{res}] {reason} | Gepland: {p_time} | Bias: {self.smoothed_bias:.2f}"
        )

        self.ha.set_solar_prediction(
            f"[{res}] {reason} | Gepland: {p_time} | Bias: {self.smoothed_bias:.2f}",
            {
                "reason": reason,
                "planned_start": p_time,
                "bias_factor": round(self.smoothed_bias, 2),
                "last_update": now.isoformat(),
            },
        )
