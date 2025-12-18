import os
import logging
import joblib
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from db import fetch_solar_training_data_orm, upsert_solar_record

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATIE
# ==============================================================================
# Systeem limiet: 6x320Wp = ~1.92kW. We ronden af op 2.0kW.
# Dit voorkomt dat het model onrealistische waardes voorspelt.
SYSTEM_MAX_KW = 2.0


# ==============================================================================
# 1. AI MODEL
# ==============================================================================
class SolarModel:
    def __init__(self, model_path="solar_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_fitted = False
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_fitted = True
                logger.info("SolarBrain: AI Model geladen.")
            except Exception as e:
                logger.error(f"SolarBrain: Kon model niet laden: {e}")

    def _create_features(self, df):
        df = df.copy()

        # 1. Timezone Uniformiteit: Zet alles om naar UTC
        if not np.issubdtype(df["timestamp"], np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if df["timestamp"].dt.tz is None:
            pass
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # 2. Seizoensfeatures
        hours = df["timestamp"].dt.hour
        doy = df["timestamp"].dt.dayofyear

        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 366)

        df["uncertainty"] = df["solcast_90"] - df["solcast_10"]

        return df[
            [
                "solcast_est",
                "solcast_10",
                "solcast_90",
                "uncertainty",
                "hour_sin",
                "hour_cos",
                "doy_sin",
                "doy_cos",
            ]
        ]

    def train(self):
        # We halen data op. Let op: we negeren consumptie data bij training.
        df = fetch_solar_training_data_orm(days=365)
        if len(df) < 50:
            logger.warning("SolarBrain: Te weinig data om te trainen (<50 samples).")
            return

        X = self._create_features(df)
        y = df["actual_pv_yield"]

        pipeline = make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(loss="absolute_error", random_state=42),
        )

        pipeline.fit(X, y)
        self.model = pipeline
        self.is_fitted = True
        joblib.dump(self.model, self.model_path)
        logger.info(f"SolarBrain: Model getraind op {len(df)} records.")

    def predict_best_window(self, solcast_data, current_pv_kw, duration_hours=1.5):
        """
        Bepaalt het beste startmoment met Bias Correction en Timezone fixes.
        current_pv_kw: Het huidige vermogen (in kW) om de voorspelling te kalibreren.
        """
        if not solcast_data:
            return None, "Geen Solcast data"

        df = pd.DataFrame(solcast_data)

        # 1. TIMEZONE FIX: Forceer alles naar UTC
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)

        df.rename(
            columns={
                "pv_estimate": "solcast_est",
                "pv_estimate10": "solcast_10",
                "pv_estimate90": "solcast_90",
            },
            inplace=True,
        )

        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 2. AI Voorspelling (Base Prediction)
        if self.is_fitted:
            X_pred = self._create_features(df)
            df["ai_power_raw"] = self.model.predict(X_pred)

            # CLIP: Beperk tot fysieke systeem limiet (0 - 2.0 kW)
            df["ai_power_raw"] = df["ai_power_raw"].clip(lower=0, upper=SYSTEM_MAX_KW)
        else:
            df["ai_power_raw"] = df["solcast_est"]

        # 3. RECENT ERROR CORRECTION (Bias Correction)
        now_utc = pd.Timestamp.now(tz="UTC")

        try:
            time_diff = (df["timestamp"] - now_utc).abs()
            idx_now = time_diff.idxmin()
            row_now = df.loc[idx_now]

            # Alleen corrigeren als de data vers is (< 45 min oud)
            if time_diff[idx_now] < pd.Timedelta(minutes=45):
                expected_now = row_now["ai_power_raw"]

                bias_factor = 1.0
                # Alleen bias berekenen als er daadwerkelijk zon verwacht wordt (>100W)
                if expected_now > 0.1:
                    raw_ratio = current_pv_kw / expected_now
                    # We geloven correcties tussen 50% en 150%
                    bias_factor = np.clip(raw_ratio, 0.5, 1.5)

                df["ai_power"] = df["ai_power_raw"] * bias_factor

                # Ook na correctie mag het niet boven de max van de panelen komen
                df["ai_power"] = df["ai_power"].clip(upper=SYSTEM_MAX_KW)

            else:
                df["ai_power"] = df["ai_power_raw"]
        except Exception as e:
            logger.warning(f"Kon bias niet berekenen: {e}")
            df["ai_power"] = df["ai_power_raw"]

        # 4. Rolling Window (Gemiddelde over tijdsduur)
        steps = int(duration_hours * 2)  # aanname 30 min slots
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=steps)

        df["rolling_power"] = df["ai_power"].rolling(window=indexer).mean()
        df["rolling_std"] = df["ai_power"].rolling(window=indexer).std().fillna(0)

        # Score = Vermogen - (Onzekerheid * 0.5)
        df["score"] = df["rolling_power"] - (df["rolling_std"] * 0.5)

        # 5. Filter Toekomst
        future = df[df["timestamp"] >= (now_utc - timedelta(minutes=30))].copy()

        if future.empty:
            return None, "Geen toekomstige data"

        # 6. De 95% Regel (Slimste startmoment kiezen)
        max_score = future["score"].max()

        # [FIX] Als de max score < 100W is, is er geen zon meer (nacht).
        # Return None zodat status op "WAIT" blijft en niet "WAIT_CLOUD".
        if max_score < 0.1:
            return None, "Geen zon verwacht (< 100W)"

        candidates = future[future["score"] >= (max_score * 0.95)]

        if candidates.empty:
            best_idx = future["score"].idxmax()
            return future.loc[best_idx], future

        # Pak de eerste (vroegste) kandidaat die bijna even goed is als de piek
        best_row = candidates.iloc[0]

        return best_row, future


# ==============================================================================
# 2. AGGREGATOR (Met Stabiliteitscheck, Zonder Consumptie)
# ==============================================================================
class SolarAggregator:
    def __init__(self, interval_seconds=60):
        # Buffer van 10 minuten voor stabiliteit
        target_history_seconds = 600
        self.maxlen = int(target_history_seconds / max(1, interval_seconds))
        if self.maxlen < 1:
            self.maxlen = 1

        self.stability_buffer_pv = deque(maxlen=self.maxlen)

        self.min_stat_samples = int(180 / max(1, interval_seconds))
        self.current_slot_start = None
        self.slot_samples_pv = []

    def add_sample(self, timestamp, pv_kw):
        completed_slot_data = None
        self.stability_buffer_pv.append(pv_kw)

        minute = 0 if timestamp.minute < 30 else 30
        slot_start = timestamp.replace(minute=minute, second=0, microsecond=0)

        if self.current_slot_start and slot_start > self.current_slot_start:
            if len(self.slot_samples_pv) > 0:
                completed_slot_data = {
                    "timestamp": self.current_slot_start,
                    "avg_pv": np.mean(self.slot_samples_pv),
                }
            self.slot_samples_pv = []

        self.current_slot_start = slot_start
        self.slot_samples_pv.append(pv_kw)

        return completed_slot_data

    def get_solar_stats(self):
        if len(self.stability_buffer_pv) < self.min_stat_samples:
            if len(self.stability_buffer_pv) > 0:
                return {
                    "median_pv": self.stability_buffer_pv[-1],
                    "safe_pv": 0,
                    "is_stable": True,
                }
            return {"median_pv": 0, "safe_pv": 0, "is_stable": True}

        pv_arr = np.array(self.stability_buffer_pv)

        # IQR Berekening: Is het weer stabiel of schieten we alle kanten op?
        # Verschil tussen 75% en 25% percentiel.
        iqr = np.percentile(pv_arr, 75) - np.percentile(pv_arr, 25)
        is_stable = iqr < 0.5

        return {
            "median_pv": float(np.median(pv_arr)),
            "safe_pv": float(np.percentile(pv_arr, 10)),
            "is_stable": is_stable,
        }


# ==============================================================================
# 3. CONTROLLER (Pure Solar Logic)
# ==============================================================================
class SolarController:
    def __init__(self, ha_client, opts):
        self.ha = ha_client
        self.opts = opts
        self.interval = opts.get("solar_interval_seconds", 15)

        self.entity_pv = opts.get(
            "sensor_pv_power", "sensor.fuj7chn07b_pv_output_actual"
        )
        self.entity_solcast = opts.get(
            "sensor_solcast", "sensor.solcast_pv_forecast_forecast_today"
        )
        self.entity_solcast_poll = opts.get(
            "sensor_solcast_poll", "sensor.solcast_pv_forecast_api_last_polled"
        )

        self.model = SolarModel()
        self.aggregator = SolarAggregator(interval_seconds=self.interval)
        self.cached_solcast = []
        self.last_known_poll_timestamp = None

        logger.info(
            f"SolarBrain: Geconfigureerd (Pure Solar Mode). Interval {self.interval}s"
        )

    def train_model(self):
        logger.info("SolarBrain: Start training cyclus...")
        self.model.train()

    def update_solcast(self):
        try:
            poll_state = self.ha.get_state(self.entity_solcast_poll)
            if not poll_state:
                return
            current_poll_ts = poll_state.get("state")
            if current_poll_ts == self.last_known_poll_timestamp:
                return

            state = self.ha.get_state(self.entity_solcast)
            if (
                state
                and "attributes" in state
                and "detailedForecast" in state["attributes"]
            ):
                raw_data = state["attributes"]["detailedForecast"]
                self.cached_solcast = raw_data
                for item in raw_data:
                    ts = pd.to_datetime(item["period_start"]).to_pydatetime()
                    upsert_solar_record(
                        ts,
                        solcast_est=item["pv_estimate"],
                        solcast_10=item["pv_estimate10"],
                        solcast_90=item["pv_estimate90"],
                    )
                self.last_known_poll_timestamp = current_poll_ts
                logger.info(
                    f"SolarBrain: Solcast data vernieuwd (TS: {current_poll_ts})"
                )
        except Exception as e:
            logger.error(f"SolarBrain: Fout bij update_solcast: {e}")

    def tick(self):
        """Main loop."""
        now_system = datetime.now()

        # 1. Lees PV Sensor
        try:
            pv_state = self.ha.get_state(self.entity_pv)
            if pv_state and pv_state["state"] not in ["unknown", "unavailable"]:
                pv_kw = float(pv_state["state"]) / 1000.0
            else:
                pv_kw = 0.0
        except Exception:
            pv_kw = 0.0

        # 2. Update buffer & database
        # We slaan consumptie op als 0.0 omdat we het niet meer gebruiken
        finished_block = self.aggregator.add_sample(now_system, pv_kw)
        if finished_block:
            upsert_solar_record(
                finished_block["timestamp"],
                actual_pv_yield=finished_block["avg_pv"],
                actual_consumption=0.0,
            )

        self.update_solcast()
        stats = self.aggregator.get_solar_stats()

        # 3. Voorspel beste moment
        # We gebruiken de mediaan (stabiel) of de huidige waarde, wat hoger is.
        current_input_pv = max(stats["median_pv"], pv_kw)

        best_moment, _ = self.model.predict_best_window(
            self.cached_solcast,
            current_pv_kw=current_input_pv,
            duration_hours=1.5,
        )

        status = "WAIT"
        msg = "In rust"
        start_time_naive = None

        # 4. LOGICA
        if best_moment is not None:
            ts_start_utc = best_moment["timestamp"]
            ts_start_local = ts_start_utc.tz_convert(datetime.now().astimezone().tzinfo)
            start_time_naive = ts_start_local.replace(tzinfo=None)

            start_window_open = start_time_naive - timedelta(minutes=15)
            start_window_close = start_time_naive + timedelta(minutes=30)

            # De verwachte kracht (gemiddeld over 1.5 uur)
            expected_power = best_moment["rolling_power"]

            # Scenario A: We zitten in het start-tijdslot
            if start_window_open <= now_system < start_window_close:

                # STABILITEITS CHECK:
                # Als het onstabiel weer is (wolken) EN de kracht is "net aan" (< 1.0 kW),
                # dan wachten we nog even om te voorkomen dat de WP gaat pendelen.
                # Als het harder waait (>1.0 kW) mag hij ook bij onstabiel weer wel aan.
                is_risky_weather = (not stats["is_stable"]) and (
                    stats["median_pv"] < 1.0
                )

                if is_risky_weather:
                    status = "WAIT_CLOUD"
                    msg = "Te onstabiel (wolkendek)"

                elif expected_power > 0 and stats["median_pv"] > (expected_power * 0.4):
                    status = "ACTIVE_PLANNED"
                    msg = f"Starten! (Plan: {start_time_naive.strftime('%H:%M')})"

                elif stats["safe_pv"] > 1.5:
                    status = "ACTIVE_NOW"
                    msg = "Zon schijnt volop"

                else:
                    status = "WAIT_CLOUD"
                    msg = f"Wachten op zon (Plan: {start_time_naive.strftime('%H:%M')})"

            # Scenario B: Het slot is in de toekomst
            elif now_system < start_window_open:
                wait_min = int((start_time_naive - now_system).total_seconds() / 60)
                if wait_min > 120:
                    msg = f"Volgende: {start_time_naive.strftime('%H:%M')}"
                else:
                    msg = f"Wacht {wait_min} min ({start_time_naive.strftime('%H:%M')})"

        # Loggen
        if status.startswith("ACTIVE") or "WAIT_CLOUD" in status:
            logger.info(f"Solar: {status} - {msg}")

        # Optioneel: Stuur self.send_status_to_ha(...) hier
