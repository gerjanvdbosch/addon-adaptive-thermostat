import os
import logging
import joblib
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Project Imports
from db import fetch_solar_training_data_orm, upsert_solar_record
from ha_client import HAClient

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATIE & CONSTANTEN
# ==============================================================================
SYSTEM_MAX_KW = 2.0  # Max vermogen van je omvormer (voorkomt onzin-voorspellingen)
SWW_DURATION_HOURS = 1.0  # Hoe lang duurt een run voor warm water?

class SolarAI:
    """
    Het brein voor zonne-energie management.
    Doel: Voorspel het ideale moment om SWW (warm water) te verwarmen.

    Functies:
    1. Traint een AI model op historische Solcast vs Werkelijke opbrengst.
    2. Corrigeert Solcast voorspellingen real-time ('Bias Correction').
    3. Zoekt het beste tijdslot (Rolling Window) voor energie-intensieve taken.
    """

    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}

        # --- Config ---
        self.model_path = Path(self.opts.get("model_path_solar", "/config/models/solar_model.pkl"))
        self.interval = int(self.opts.get("solar_interval_seconds", 60))

        # Sensoren
        self.entity_pv = self.opts.get("sensor_pv_power", "sensor.fuj7chn07b_pv_output_actual")
        self.entity_solcast = self.opts.get("sensor_solcast", "sensor.solcast_pv_forecast_forecast_today")
        self.entity_solcast_poll = self.opts.get("sensor_solcast_poll", "sensor.solcast_pv_forecast_api_last_polled")

        # Zorg dat de map bestaat
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # --- State ---
        self.model = None
        self.is_fitted = False
        self.last_run_ts = None

        # Solcast Cache
        self.cached_solcast_data = []
        self.last_solcast_poll_ts = None

        # Stabiliteits Buffer (Historie van laatste 10 min)
        # We gebruiken dit om te bepalen of het weer stabiel is
        self.history_len = int(600 / max(1, self.interval))
        self.pv_buffer = deque(maxlen=self.history_len)

        # Slot Aggregatie (voor DB opslag per minuut/kwartier)
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
                logger.info("SolarAI: Model loaded successfully.")
            except Exception:
                logger.exception("SolarAI: Failed to load model.")

    def _atomic_save(self, model):
        tmp_path = self.model_path.with_suffix(".tmp")
        try:
            joblib.dump(model, tmp_path)
            tmp_path.replace(self.model_path)
            logger.info("SolarAI: Model saved.")
        except Exception:
            logger.exception("SolarAI: Failed to save model.")

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Single Source of Truth voor feature engineering."""
        df = df.copy()

        # Tijd conversie naar UTC voor uniformiteit
        if not np.issubdtype(df["timestamp"], np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if df["timestamp"].dt.tz is None:
             pass # Al naive UTC
        else:
             df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Cyclische Tijd Features
        hours = df["timestamp"].dt.hour
        doy = df["timestamp"].dt.dayofyear

        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 366)

        # Onzekerheid (Verschil tussen Solcast P10 en P90)
        df["uncertainty"] = df["solcast_90"] - df["solcast_10"]

        features = [
            "solcast_est", "solcast_10", "solcast_90", "uncertainty",
            "hour_sin", "hour_cos", "doy_sin", "doy_cos"
        ]
        return df[features]

    # ==============================================================================
    # 2. TRAINING
    # ==============================================================================
    def train(self):
        """Haalt data op en traint het model (Geoptimaliseerde Versie)."""
        logger.info("SolarAI: Starting training...")

        # Haal data op (bijv. laatste 180 dagen)
        df = fetch_solar_training_data_orm(days=180)

        # Check op minimale data
        if len(df) < 50:
            logger.warning("SolarAI: Too few samples (<50). Skipping.")
            return

        # Features & Target
        X = self._create_features(df)
        y = df["actual_pv_yield"]

        # Validatie: Schoonmaken (al zou fetch_orm dit al redelijk moeten doen)
        # We zorgen dat we geen NaN targets hebben
        mask = np.isfinite(y)
        X, y = X[mask], y[mask]

        if len(y) < 50:
            return

        # Train/Val Split (85/15) voor Early Stopping
        # Dit is belangrijk zodat het model stopt VOORDAT het ruis gaat leren
        n_total = len(y)
        split_idx = int(n_total * 0.85)

        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

        # Geoptimaliseerde Configuratie (Gelijk aan ThermostatAI)
        # Geen StandardScaler nodig voor Histogram-based Gradient Boosting
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",      # Minimaliseer MAE (robuust tegen uitschieters)
            learning_rate=0.05,         # Rustig leren
            max_iter=2000,              # Hoog limiet, early stopping is de rem
            max_leaf_nodes=31,          # Standaard complexiteit
            min_samples_leaf=20,        # Minimaal 20 meetpunten per beslisregel
            l2_regularization=1.0,      # Voorkomt extreme waardes
            early_stopping=True,        # STOP als validatie niet meer verbetert
            validation_fraction=None,   # We geven handmatig X_val mee
            n_iter_no_change=20,        # Geduld
            random_state=42
        )

        try:
            # We geven expliciet de validatieset mee voor betere early stopping
            self.model.fit(X_train, y_train, val_set=(X_val, y_val))

            self.is_fitted = True
            self._atomic_save(self.model)

            # Log score
            score = self.model.score(X_val, y_val)
            logger.info(f"SolarAI: Training finished. Samples={n_total}, R2_Score={score:.3f}")

        except Exception:
            logger.exception("SolarAI: Training failed")

    # ==============================================================================
    # 3. DATA VERWERKING (Runtime)
    # ==============================================================================

    def _update_solcast_cache(self):
        """Checkt of HA nieuwe Solcast data heeft en update de cache."""
        try:
            poll_state = self.ha.get_state(self.entity_solcast_poll)
            if not poll_state: return

            current_poll_ts = poll_state.get("state")
            if current_poll_ts == self.last_solcast_poll_ts:
                return # Niets nieuws

            state = self.ha.get_state(self.entity_solcast)
            if state and "attributes" in state and "detailedForecast" in state["attributes"]:
                raw_data = state["attributes"]["detailedForecast"]
                self.cached_solcast_data = raw_data

                # Sla ook meteen op in DB voor latere training
                for item in raw_data:
                    ts = pd.to_datetime(item["period_start"]).to_pydatetime()
                    upsert_solar_record(
                        ts,
                        solcast_est=item["pv_estimate"],
                        solcast_10=item["pv_estimate10"],
                        solcast_90=item["pv_estimate90"]
                    )

                self.last_solcast_poll_ts = current_poll_ts
                logger.info(f"SolarAI: Solcast cache updated (TS: {current_poll_ts})")

        except Exception:
            logger.exception("SolarAI: Error updating Solcast cache")

    def _process_pv_sample(self, pv_kw):
        """Verwerkt huidige PV waarde, stopt in buffer en slaat op in DB."""
        now = datetime.now()
        self.pv_buffer.append(pv_kw)

        # Aggregatie logica (bijv. gemiddelde per kwartier opslaan)
        # We ronden af naar het dichtstbijzijnde blok (00 of 30)
        minute = 0 if now.minute < 30 else 30
        slot_start = now.replace(minute=minute, second=0, microsecond=0)

        if self.current_slot_start and slot_start > self.current_slot_start:
            # Vorig slot is klaar, opslaan
            if self.slot_samples:
                avg_pv = np.mean(self.slot_samples)
                upsert_solar_record(
                    self.current_slot_start,
                    actual_pv_yield=avg_pv,
                    actual_consumption=0.0 # Legacy field, op 0 zetten
                )
            self.slot_samples = []

        self.current_slot_start = slot_start
        self.slot_samples.append(pv_kw)

    def _get_stability_stats(self):
        """Geeft terug: median_pv en is_stable (boolean)."""
        if len(self.pv_buffer) < 5:
            return 0.0, True # Te weinig data, ga uit van rust

        arr = np.array(self.pv_buffer)
        median = float(np.median(arr))

        # Interkwartielafstand (IQR) als maat voor volatiliteit
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)

        # Als het verschil groot is, is het onstabiel (wolkendek)
        # Bij laag vermogen (<0.5kW) mag de variatie kleiner zijn
        threshold = 0.5 if median > 1.0 else 0.2
        is_stable = iqr < threshold

        return median, is_stable

    # ==============================================================================
    # 4. INFERENCE (Het "SWW Moment" bepalen)
    # ==============================================================================

    def get_sww_recommendation(self):
        """
        Bepaalt of NU het moment is om SWW te verwarmen.
        Returns:
            dict: { "action": "START" | "WAIT", "reason": str, "plan_start": datetime }
        """
        if not self.cached_solcast_data:
            return {"action": "WAIT", "reason": "Geen Solcast data"}

        # 1. Prepareer Forecast DataFrame
        df = pd.DataFrame(self.cached_solcast_data)
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)

        # Hernoem kolommen voor model
        rename_map = {
            "pv_estimate": "solcast_est",
            "pv_estimate10": "solcast_10",
            "pv_estimate90": "solcast_90"
        }
        df.rename(columns=rename_map, inplace=True)
        df.sort_values("timestamp", inplace=True)

        # 2. AI Voorspelling (Raw)
        if self.is_fitted:
            X_pred = self._create_features(df)
            df["ai_power_raw"] = self.model.predict(X_pred)
            df["ai_power_raw"] = df["ai_power_raw"].clip(0, SYSTEM_MAX_KW)
        else:
            df["ai_power_raw"] = df["solcast_est"] # Fallback

        # 3. Bias Correctie (Real-time aanpassing)
        # Vergelijk huidige meting met wat we dachten te hebben op DIT tijdstip
        now_utc = pd.Timestamp.now(tz="UTC")
        median_pv, is_stable = self._get_stability_stats()

        # Zoek dichtstbijzijnde punt in forecast
        df["time_diff"] = (df["timestamp"] - now_utc).abs()
        nearest_idx = df["time_diff"].idxmin()

        # Alleen corrigeren als forecast vers is (<45 min)
        if df.loc[nearest_idx, "time_diff"] < pd.Timedelta(minutes=45):
            expected_now = df.loc[nearest_idx, "ai_power_raw"]

            # Bereken bias factor (maar begrens hem om extremen te voorkomen)
            bias = 1.0
            if expected_now > 0.1: # Alleen als we zon verwachten
                bias = median_pv / expected_now
                bias = np.clip(bias, 0.5, 1.5) # Max 50% afwijking toestaan

            df["ai_power"] = df["ai_power_raw"] * bias
            df["ai_power"] = df["ai_power"].clip(upper=SYSTEM_MAX_KW)
        else:
            df["ai_power"] = df["ai_power_raw"]

        # 4. Rolling Window (Het SWW Blok)
        # We zoeken een blok van X uur met de hoogste energie
        steps = int(SWW_DURATION_HOURS * 2) # Solcast is per 30 min

        # Rolling mean geeft gemiddelde vermogen over het venster
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=steps)
        df["window_avg_power"] = df["ai_power"].rolling(window=indexer).mean()

        # 5. Beslismoment
        # Filter alleen de toekomst (vanaf 15 min geleden tot vanavond)
        future = df[df["timestamp"] >= (now_utc - timedelta(minutes=15))].copy()

        if future.empty:
            return {"action": "WAIT", "reason": "Geen data voor vandaag"}

        best_idx = future["window_avg_power"].idxmax()
        best_window_row = future.loc[best_idx]
        best_power = best_window_row["window_avg_power"]

        # Tijdstippen naar lokaal converteren voor leesbaarheid
        local_tz = datetime.now().astimezone().tzinfo
        start_time_utc = best_window_row["timestamp"]
        start_time_local = start_time_utc.tz_convert(local_tz)

        # Huidige tijd marges
        start_margin = start_time_utc - timedelta(minutes=15)
        end_margin = start_time_utc + timedelta(minutes=30)

        # 6. LOGICA: Starten of Wachten?

        # A. Is er überhaupt genoeg zon vandaag?
        if best_power < 0.5: # Bijv minder dan 500W gemiddeld
            return {
                "action": "WAIT",
                "reason": "Te weinig zon vandaag (<500W)",
                "plan_start": start_time_local
            }

        # B. Zitten we in het ideale blok?
        if start_margin <= now_utc < end_margin:
            # We moeten starten, MAAR... is het wel stabiel?

            if not is_stable and median_pv < 1.0:
                # Het is instabiel en < 1kW. Wacht even.
                return {
                    "action": "WAIT_CLOUD",
                    "reason": "Onstabiel weer (Wolkendek)",
                    "plan_start": start_time_local
                }

            return {
                "action": "START",
                "reason": f"Optimaal venster bereikt (Verwacht {best_power:.1f} kW)",
                "plan_start": start_time_local
            }

        # C. Zijn we te vroeg?
        if now_utc < start_margin:
            wait_min = int((start_time_utc - now_utc).total_seconds() / 60)
            return {
                "action": "WAIT",
                "reason": f"Wacht {wait_min} min op piek",
                "plan_start": start_time_local
            }

        # D. Zijn we te laat? (Zou niet moeten gebeuren door 'future' filter, maar toch)
        return {"action": "WAIT", "reason": "Piek voorbij", "plan_start": start_time_local}


    def run_cycle(self):
        """Main loop: Update data -> Bereken advies."""
        now = datetime.now()

        # Rate limiter
        if self.last_run_ts and (now - self.last_run_ts).total_seconds() < 5:
            return
        self.last_run_ts = now

        # 1. Lees PV
        try:
            pv_state = self.ha.get_state(self.entity_pv)
            pv_kw = float(pv_state["state"]) / 1000.0 if pv_state and pv_state["state"] not in ["unknown", "unavailable"] else 0.0
        except:
            pv_kw = 0.0

        # 2. Update System (Buffer, DB, Solcast)
        self._process_pv_sample(pv_kw)
        self._update_solcast_cache()

        # 3. Vraag Advies
        advice = self.get_sww_recommendation()

        status = advice["action"]
        reason = advice.get("reason", "")
        plan = advice.get("plan_start")
        plan_str = plan.strftime("%H:%M") if plan else "??"

        logger.info(f"SolarAI: {status} - {reason} (Plan: {plan_str})")

        # Optioneel: Stuur naar HA
        # self.ha.set_state("sensor.solar_ai_status", status, {"reason": reason, "next_run": plan_str})