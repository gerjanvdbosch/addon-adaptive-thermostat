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

        # 1. Timezone Uniformiteit: Zet alles om naar UTC en verwijder tz-info voor de ML-input
        if not np.issubdtype(df["timestamp"], np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Zorg dat we zeker weten dat het datetime objecten zijn
        if df["timestamp"].dt.tz is None:
            # Als het naive is, neem aan dat het UTC is (trainingsdata opslag)
            pass
        else:
            # Als het aware is, converteer naar UTC
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # 2. Betere Seizoensfeatures (Day of Year i.p.v. Month)
        hours = df["timestamp"].dt.hour
        doy = df["timestamp"].dt.dayofyear  # 1 tot 366

        df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

        # Nauwkeuriger dan maand: de zon staat anders op 1 maart dan 31 maart
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

        # Sorteer zekerheidshalve op tijd
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 2. AI Voorspelling (Base Prediction)
        if self.is_fitted:
            X_pred = self._create_features(df)
            df["ai_power_raw"] = self.model.predict(X_pred)
            df["ai_power_raw"] = df["ai_power_raw"].clip(lower=0)
        else:
            df["ai_power_raw"] = df["solcast_est"]

        # 3. RECENT ERROR CORRECTION (Bias Correction)
        # We vergelijken NU met wat Solcast NU dacht.
        now_utc = pd.Timestamp.now(tz="UTC")

        # Vind de rij die het dichtst bij 'nu' ligt (max 30 min verschil)
        # We gebruiken get_indexer met method='nearest'
        try:
            time_diff = (df["timestamp"] - now_utc).abs()
            idx_now = time_diff.idxmin()
            row_now = df.loc[idx_now]

            # Check of "dichtstbijzijnde" niet 3 uur geleden is (bv oude data)
            if time_diff[idx_now] < pd.Timedelta(minutes=45):
                expected_now = row_now["ai_power_raw"]

                # Alleen corrigeren als we verwachten dat de zon schijnt (> 100W)
                # Anders ga je delen door (bijna) nul in de nacht/ochtend.
                bias_factor = 1.0
                if expected_now > 0.1:
                    raw_ratio = current_pv_kw / expected_now
                    # Clamp de factor: we geloven correcties tussen -50% en +50%
                    # Als de ratio 0 is (storing?) of 10 (onmogelijk?), beperken we de impact.
                    bias_factor = np.clip(raw_ratio, 0.5, 1.5)

                # Pas de factor toe op de hele dataset (of alleen toekomst)
                df["ai_power"] = df["ai_power_raw"] * bias_factor

                # Debug info in log (optioneel)
                logger.info(
                    f"Bias correctie: Verwacht={expected_now:.2f}, Echt={current_pv_kw:.2f}, Factor={bias_factor:.2f}"
                )
            else:
                # Data te oud om bias te bepalen
                df["ai_power"] = df["ai_power_raw"]
        except Exception as e:
            logger.warning(f"Kon bias niet berekenen: {e}")
            df["ai_power"] = df["ai_power_raw"]

        # 4. Rolling Window
        steps = int(duration_hours * 2)  # aanname 30 min data
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=steps)

        df["rolling_power"] = df["ai_power"].rolling(window=indexer).mean()
        df["rolling_std"] = df["ai_power"].rolling(window=indexer).std().fillna(0)

        # Score berekening
        df["score"] = df["rolling_power"] - (df["rolling_std"] * 0.5)

        # 5. Filter Toekomst (UTC vergelijking!)
        # We kijken 30 min terug om "net gestarte" slots niet te missen,
        # maar plannen vooral vooruit.
        future = df[df["timestamp"] >= (now_utc - timedelta(minutes=30))].copy()

        if future.empty:
            return None, "Geen toekomstige data (UTC check)"

        # 6. De 95% Regel (Slimste startmoment kiezen)
        max_score = future["score"].max()

        if max_score < 0.1:
            return future.iloc[0], future

        candidates = future[future["score"] >= (max_score * 0.95)]

        if candidates.empty:
            best_idx = future["score"].idxmax()
            return future.loc[best_idx], future

        # Pak de eerste (vroegste) kandidaat
        best_row = candidates.iloc[0]

        return best_row, future


# ==============================================================================
# 2. AGGREGATOR (Dynamisch Buffer)
# ==============================================================================
class SolarAggregator:
    def __init__(self, interval_seconds=60):
        target_history_seconds = 600
        self.maxlen = int(target_history_seconds / max(1, interval_seconds))
        if self.maxlen < 1:
            self.maxlen = 1

        self.stability_buffer_pv = deque(maxlen=self.maxlen)
        self.stability_buffer_cons = deque(maxlen=self.maxlen)

        self.min_stat_samples = int(180 / max(1, interval_seconds))
        self.current_slot_start = None
        self.slot_samples_pv = []
        self.slot_samples_cons = []

    def add_sample(self, timestamp, pv_kw, cons_kw):
        completed_slot_data = None
        self.stability_buffer_pv.append(pv_kw)
        self.stability_buffer_cons.append(cons_kw)

        minute = 0 if timestamp.minute < 30 else 30
        slot_start = timestamp.replace(minute=minute, second=0, microsecond=0)

        if self.current_slot_start and slot_start > self.current_slot_start:
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
        if len(self.stability_buffer_pv) < self.min_stat_samples:
            if len(self.stability_buffer_pv) > 0:
                return {
                    "safe_pv": self.stability_buffer_pv[-1],
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
                    # Let op: upsert verwacht dat je weet wat voor tijd dit is.
                    # Meestal is Solcast raw data UTC.
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

        # 1. Lees sensors
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

        # 2. Update buffer & database
        finished_block = self.aggregator.add_sample(now_system, pv_kw, p1_kw)
        if finished_block:
            # Let op: finished_block timestamp is 'naive' (lokale tijd van systeemtijd)
            # In database opslaan als UTC is vaak beter, maar hangt af van je DB implementatie
            upsert_solar_record(
                finished_block["timestamp"],
                actual_pv_yield=finished_block["avg_pv"],
                actual_consumption=finished_block["avg_cons"],
            )

        self.update_solcast()
        stats = self.aggregator.get_safety_stats()

        # 3. Voorspel met BIAS CORRECTION
        # We geven stats['median_pv'] (stabieler) of pv_kw (directer) mee.
        # Median is beter om pieken door wolken/schaduw uit te middelen.
        current_stable_pv = stats["median_pv"] if stats["median_pv"] > 0 else pv_kw

        best_moment, _ = self.model.predict_best_window(
            self.cached_solcast,
            current_pv_kw=current_stable_pv,  # Nieuw argument!
            duration_hours=1.5,
        )

        status = "WAIT"
        msg = "In rust"
        start_time_naive = None

        # 4. Logica met Timezone correctie voor display
        if stats["peak_cons"] > 3.0:
            status = "WAIT_CONSUMPTION"
            msg = f"Verbruik hoog ({stats['peak_cons']:.1f} kW)"

        elif best_moment is not None:
            ts_start_utc = best_moment["timestamp"]

            # Converteer UTC timestamp naar lokale tijd voor display & vergelijking met 'now_system'
            # We nemen aan dat now_system (datetime.now()) overeenkomt met de lokale tijdzone van de server.
            ts_start_local = ts_start_utc.tz_convert(datetime.now().astimezone().tzinfo)
            start_time_naive = ts_start_local.replace(
                tzinfo=None
            )  # Strip tz voor cleane logica vergelijking

            start_window_open = start_time_naive - timedelta(minutes=15)
            start_window_close = start_time_naive + timedelta(minutes=30)

            if start_window_open <= now_system < start_window_close:
                # Hier kijken we naar de GECORRIGEERDE voorspelling (best_moment['rolling_power'])
                # Omdat 'rolling_power' nu al de bias-factor heeft, is de drempel betrouwbaarder.
                expected_power = best_moment["rolling_power"]

                # Als de voorspelling (na correctie) nog steeds zegt "er is zon", en we meten ook zon:
                if expected_power > 0 and stats["median_pv"] > (expected_power * 0.4):
                    status = "ACTIVE_PLANNED"
                    msg = f"Starten! (Plan: {start_time_naive.strftime('%H:%M')})"
                elif stats["safe_pv"] > 1.5:
                    status = "ACTIVE_NOW"
                    msg = "Zon schijnt harder dan verwacht"
                else:
                    status = "WAIT_CLOUD"
                    msg = f"Wachten op zon (Plan: {start_time_naive.strftime('%H:%M')})"

            elif now_system < start_window_open:
                wait_min = int((start_time_naive - now_system).total_seconds() / 60)
                if wait_min > 120:
                    msg = f"Volgende: {start_time_naive.strftime('%H:%M')}"
                else:
                    msg = f"Wacht {wait_min} min ({start_time_naive.strftime('%H:%M')})"

        # 4. Stuur naar HA
        #         self.send_status_to_ha(status, msg, start_time_naive)

        # Alleen loggen bij verandering of error voorkomt log spam, hier beknopt:
        #         if status.startswith("ACTIVE") or "WAIT_CLOUD" in status:
        logger.info(f"Solar: {status} - {msg} - {start_time_naive}")
