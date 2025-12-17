import os
import logging
import joblib
import requests
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timedelta

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from database import fetch_solar_training_data_orm, upsert_solar_record

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. AI MODEL (Ongewijzigd)
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
        if not np.issubdtype(df["timestamp"], np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        hours = df["timestamp"].dt.hour
        months = df["timestamp"].dt.month

        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        df["month_sin"] = np.sin(2 * np.pi * months / 12)
        df["month_cos"] = np.cos(2 * np.pi * months / 12)
        df["uncertainty"] = df["solcast_90"] - df["solcast_10"]

        return df[
            [
                "solcast_est",
                "solcast_10",
                "solcast_90",
                "uncertainty",
                "hour_sin",
                "hour_cos",
                "month_sin",
                "month_cos",
            ]
        ]

    def train(self):
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

    def predict_best_window(self, solcast_data, duration_hours=1.0):
        if not solcast_data:
            return None, "Geen Solcast data"

        df = pd.DataFrame(solcast_data)
        df["timestamp"] = pd.to_datetime(df["period_start"])
        df.rename(
            columns={
                "pv_estimate": "solcast_est",
                "pv_estimate10": "solcast_10",
                "pv_estimate90": "solcast_90",
            },
            inplace=True,
        )

        if self.is_fitted:
            X_pred = self._create_features(df)
            df["ai_power"] = self.model.predict(X_pred)
            df["ai_power"] = df["ai_power"].clip(lower=0)
        else:
            df["ai_power"] = df["solcast_est"]

        steps = int(duration_hours * 2)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=steps)

        df["rolling_power"] = df["ai_power"].rolling(window=indexer).mean()
        df["rolling_std"] = df["ai_power"].rolling(window=indexer).std().fillna(0)
        df["score"] = df["rolling_power"] - (df["rolling_std"] * 1.0)

        if df["timestamp"].dt.tz is not None:
            now = datetime.now(df["timestamp"].dt.tz)
        else:
            now = datetime.now()

        future = df[df["timestamp"] >= (now - timedelta(minutes=30))]

        if future.empty:
            return None, "Geen toekomstige data"

        best_idx = future["score"].idxmax()
        return future.loc[best_idx], future


# ==============================================================================
# 2. AGGREGATOR (Dynamisch Buffer)
# ==============================================================================
class SolarAggregator:
    def __init__(self, interval_seconds=60):
        # We willen altijd ongeveer 10 minuten (600 seconden) historie vasthouden
        # voor de stabiliteitscheck (P10/P90).
        target_history_seconds = 600

        # Bereken buffer grootte: 600 / interval. Minimaal 1 sample.
        self.maxlen = int(target_history_seconds / max(1, interval_seconds))
        if self.maxlen < 1:
            self.maxlen = 1

        self.stability_buffer_pv = deque(maxlen=self.maxlen)
        self.stability_buffer_cons = deque(maxlen=self.maxlen)

        # Minimaal aantal samples voordat we statistiek vertrouwen (bv. na 3 minuten)
        self.min_stat_samples = int(180 / max(1, interval_seconds))

        self.current_slot_start = None
        self.slot_samples_pv = []
        self.slot_samples_cons = []

    def add_sample(self, timestamp, pv_kw, cons_kw):
        completed_slot_data = None

        self.stability_buffer_pv.append(pv_kw)
        self.stability_buffer_cons.append(cons_kw)

        # 30-min slot logica
        minute = 0 if timestamp.minute < 30 else 30
        slot_start = timestamp.replace(minute=minute, second=0, microsecond=0)

        if self.current_slot_start and slot_start > self.current_slot_start:
            # Check of we data hebben (lege lijst check)
            if len(self.slot_samples_pv) > 0:
                completed_slot_data = {
                    "timestamp": self.current_slot_start,
                    "avg_pv": np.mean(self.slot_samples_pv),
                    "avg_cons": np.mean(self.slot_samples_cons),
                }
            self.slot_samples_pv = []
            self.slot_samples_cons = []

        self.current_slot_start = slot_start
        self.slot_samples_pv.append(pv_kw)
        self.slot_samples_cons.append(cons_kw)

        return completed_slot_data

    def get_safety_stats(self):
        # Als we nog maar net gestart zijn, geef veilige defaults
        if len(self.stability_buffer_pv) < self.min_stat_samples:
            # Probeer toch iets terug te geven als er al data is
            if len(self.stability_buffer_pv) > 0:
                return {
                    "safe_pv": self.stability_buffer_pv[-1],  # Huidige waarde
                    "peak_cons": self.stability_buffer_cons[-1],
                    "median_pv": self.stability_buffer_pv[-1],
                    "is_stable": True,
                }
            return {"is_stable": True, "safe_pv": 0, "peak_cons": 0, "median_pv": 0}

        pv_arr = np.array(self.stability_buffer_pv)
        cons_arr = np.array(self.stability_buffer_cons)

        return {
            "safe_pv": float(np.percentile(pv_arr, 10)),
            "median_pv": float(np.median(pv_arr)),
            "peak_cons": float(np.percentile(cons_arr, 90)),
            "is_stable": (np.percentile(pv_arr, 75) - np.percentile(pv_arr, 25)) < 0.5,
        }


# ==============================================================================
# 3. CONTROLLER
# ==============================================================================
class SolarController:
    def __init__(self, ha_client, opts):
        self.ha = ha_client
        self.opts = opts

        # Haal het interval op uit de opties (default 60s als niet ingesteld)
        self.interval = opts.get("solar_interval_seconds", 15)

        self.entity_pv = opts.get(
            "sensor_pv_power", "sensor.fuj7chn07b_pv_output_actual"
        )
        self.entity_p1 = opts.get("sensor_p1_power", "sensor.p1_meter_power")
        self.entity_solcast = opts.get(
            "sensor_solcast", "sensor.solcast_pv_forecast_forecast_today"
        )
        self.entity_solcast_poll = opts.get(
            "sensor_solcast_poll", "sensor.solcast_pv_forecast_api_last_polled"
        )

        self.entity_output_status = "sensor.solar_brain_status"
        self.entity_output_start = "sensor.solar_brain_next_start"

        self.model = SolarModel()
        # Geef interval door aan aggregator zodat buffers kloppen
        self.aggregator = SolarAggregator(interval_seconds=self.interval)
        self.cached_solcast = []
        self.last_known_poll_timestamp = None

        logger.info(f"SolarBrain: Geconfigureerd met interval {self.interval}s")

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

    def send_status_to_ha(self, status, message, best_start=None):
        try:
            token = os.getenv("SUPERVISOR_TOKEN")
            if not token:
                return

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            base_url = "http://supervisor/core"

            payload = {
                "state": status,
                "attributes": {
                    "friendly_name": "Solar Brain Status",
                    "message": message,
                },
            }
            requests.post(
                f"{base_url}/api/states/{self.entity_output_status}",
                json=payload,
                headers=headers,
            )

            if best_start:
                payload_time = {
                    "state": best_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "attributes": {"friendly_name": "Volgende Starttijd"},
                }
                requests.post(
                    f"{base_url}/api/states/{self.entity_output_start}",
                    json=payload_time,
                    headers=headers,
                )
        except Exception:
            pass

    def tick(self):
        """Main loop: wordt aangeroepen obv inferencer_interval_seconds."""
        now = datetime.now()

        try:
            pv_state = self.ha.get_state(self.entity_pv)
            p1_state = self.ha.get_state(self.entity_p1)

            def parse_kw(state_obj):
                if state_obj and state_obj["state"] not in ["unknown", "unavailable"]:
                    try:
                        return float(state_obj["state"]) / 1000.0
                    except ValueError:
                        return 0.0
                return 0.0

            pv_kw = parse_kw(pv_state)
            p1_kw = parse_kw(p1_state)
        except Exception:
            pv_kw, p1_kw = 0, 0

        finished_block = self.aggregator.add_sample(now, pv_kw, p1_kw)
        if finished_block:
            upsert_solar_record(
                finished_block["timestamp"],
                actual_pv_yield=finished_block["avg_pv"],
                actual_consumption=finished_block["avg_cons"],
            )

        self.update_solcast()

        stats = self.aggregator.get_safety_stats()
        best_moment, _ = self.model.predict_best_window(self.cached_solcast)

        status = "WAIT"
        msg = "In rust"
        start_time = None

        if stats["peak_cons"] > 2.5:
            status = "WAIT_CONSUMPTION"
            msg = f"Verbruik hoog ({stats['peak_cons']:.1f} kW)"
        elif stats["safe_pv"] > 2.0:
            status = "ACTIVE_NOW"
            msg = f"Zon stabiel ({stats['safe_pv']:.1f} kW)"
        elif best_moment is not None:
            start = best_moment["timestamp"].to_pydatetime()
            start_time = start
            end = start + timedelta(hours=1.0)

            if start <= now < end:
                if stats["median_pv"] > (best_moment["rolling_power"] * 0.4):
                    status = "ACTIVE_PLANNED"
                    msg = f"Gepland ({start.strftime('%H:%M')})"
                else:
                    status = "WAIT_CLOUD"
                    msg = "Planning actief, maar bewolkt"
            elif now < start:
                wait_min = int((start - now).total_seconds() / 60)
                msg = f"Wacht {wait_min} min ({start.strftime('%H:%M')})"

        #         self.send_status_to_ha(status, msg, start_time)
        logger.info(
            f"SolarBrain: Tick voltooid - Status: {status}, Bericht: {msg}, Start: {start_time}"
        )
