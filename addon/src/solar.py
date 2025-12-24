import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
from utils import add_cyclic_time_features

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
EARLY_START_THRESHOLD = 0.95  # Venster-optimalisatie: start bij 95% van de komende piek


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

    def _check_midnight_reset(self):
        """Wist de cache zodra er een nieuwe dag begint."""
        today = datetime.now().date()
        if today != self.last_midnight_reset_date:
            logger.info(
                "SolarAI: Nieuwe dag gedetecteerd. Solcast cache wordt geleegd."
            )
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
        """
        Bepaalt de stabiliteit van het weer.
        Schaalt dynamisch mee met SYSTEM_MAX_KW.
        """
        if len(self.pv_buffer) < 5:
            return 0.0, True

        arr = np.array(self.pv_buffer)
        median = float(np.median(arr))

        # IQR = Interkwartielafstand (verschil tussen 75% en 25% punt)
        # Dit geeft aan hoe hard de waarden heen en weer springen
        iqr = np.percentile(arr, 75) - np.percentile(arr, 25)

        # 1. Bepaal of we momenteel op "hoog vermogen" draaien
        # Grens: 40% van wat je systeem kan (bij 2.0kW is dit 0.8kW)
        high_power_limit = SYSTEM_MAX_KW * 0.4

        # 2. Bepaal de drempel voor instabiliteit
        if median > high_power_limit:
            # Bij hoog vermogen (felle zon) zijn de klappen van wolken groter.
            # We tolereren fluctuaties tot 20% van de systeemcapaciteit.
            # (Bij 2.0kW = 0.4kW fluctuatie toegestaan)
            threshold = SYSTEM_MAX_KW * 0.20
        else:
            # Bij laag vermogen (ochtend/avond/winter) is de basislijn lager.
            # We tolereren dan minder absolute schommeling (10% capaciteit).
            # (Bij 2.0kW = 0.2kW fluctuatie toegestaan)
            threshold = SYSTEM_MAX_KW * 0.10

        # Harde ondergrens voor ruis (anders wordt hij zenuwachtig van 50W verschil)
        threshold = max(threshold, 0.15)

        is_stable = iqr < threshold

        return median, is_stable

    # ==============================================================================
    # 4. INFERENCE (Het "SWW Moment" bepalen)
    # ==============================================================================

    def get_solar_recommendation(self):
        """Analyseert de forecast en de huidige bias om een start-advies te geven."""
        if not self.cached_solcast_data:
            return {"action": "WAIT", "reason": "Wachten op Solcast data..."}

        # ----------------------------------------------------------------------
        # 1. VOORBEREIDING DATA
        # ----------------------------------------------------------------------
        df = pd.DataFrame(self.cached_solcast_data)
        df["timestamp"] = pd.to_datetime(df["period_start"], utc=True)
        df.sort_values("timestamp", inplace=True)

        # Filter nachtwaarden eruit
        df = df[df["pv_estimate"] > 0.01]
        if df.empty:
            return {"action": "WAIT", "reason": "Nacht: Geen zon"}

        now_utc = pd.Timestamp.now(tz="UTC")
        local_tz = datetime.now().astimezone().tzinfo

        # ----------------------------------------------------------------------
        # 2. AI PREDICTIE
        # ----------------------------------------------------------------------
        if self.is_fitted and self.model:
            X_pred = self._create_features(df)
            pred_ai = self.model.predict(X_pred)
            # Blending: 60% AI, 40% Solcast
            df["ai_power_raw"] = (0.6 * pred_ai) + (0.4 * df["pv_estimate"])
            df["ai_power_raw"] = df["ai_power_raw"].clip(0, SYSTEM_MAX_KW)
        else:
            df["ai_power_raw"] = df.get("pv_estimate", 0.0)

        # ----------------------------------------------------------------------
        # 3. STATISTIEKEN & BIAS
        # ----------------------------------------------------------------------
        median_pv, is_stable = self._get_stability_stats()

        # Zoek dichtstbijzijnde forecast punt voor bias berekening
        df["time_diff"] = (df["timestamp"] - now_utc).abs()
        nearest_idx = df["time_diff"].idxmin()

        # Update Bias
        if df.loc[nearest_idx, "time_diff"] < pd.Timedelta(minutes=45):
            expected_now = df.loc[nearest_idx, "ai_power_raw"]

            # Voorkom delen door nul of extreme uitschieters bij zonsopkomst
            if expected_now > 0.1:
                new_bias = median_pv / expected_now
                # EWMA smoothing
                self.smoothed_bias = (0.8 * self.smoothed_bias) + (0.2 * new_bias)
                self.smoothed_bias = np.clip(self.smoothed_bias, 0.4, 1.6)

        # Pas bias toe op de hele dataframe
        df["ai_power"] = (df["ai_power_raw"] * self.smoothed_bias).clip(
            0, SYSTEM_MAX_KW
        )

        # ----------------------------------------------------------------------
        # 4. ROLLING WINDOW (SWW BLOK)
        # ----------------------------------------------------------------------
        window_steps = max(1, int(SWW_DURATION_HOURS * 2))  # Bijv. 2 stappen van 30min
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_steps)

        # We gebruiken de biased power voor het plannen
        df["window_avg_power"] = df["ai_power"].rolling(window=indexer).mean()

        # Filter: Alleen toekomst
        future = df[df["timestamp"] >= (now_utc - timedelta(minutes=15))].copy()
        future.dropna(subset=["window_avg_power"], inplace=True)

        if future.empty:
            return {"action": "WAIT", "reason": "Einde zonnige dagdeel."}

        # ----------------------------------------------------------------------
        # 5. BESTE MOMENT BEPALEN (MET 95% OPTIMALISATIE!)
        # ----------------------------------------------------------------------
        # Dit is de verbetering: We wachten niet op de absolute 100% piek,
        # maar pakken de EERSTE kans die >95% van de piek is.
        # Dit zorgt dat je op mooie dagen 1 a 2 uur eerder start.
        max_peak_power = future["window_avg_power"].max()
        threshold = max(max_peak_power * EARLY_START_THRESHOLD, 0.01)

        candidates = future[future["window_avg_power"] >= threshold]

        if not candidates.empty:
            best_row = candidates.iloc[
                0
            ]  # Pak de EERSTE (vroegste) die aan de eis voldoet
            best_idx = best_row.name
        else:
            best_idx = future["window_avg_power"].idxmax()
            best_row = future.loc[best_idx]

        best_power = best_row["window_avg_power"]
        start_time_local = best_row["timestamp"].tz_convert(local_tz)

        # ----------------------------------------------------------------------
        # 6. DAG CLASSIFICATIE
        # ----------------------------------------------------------------------
        today_start = now_utc.normalize()
        today_end = today_start + timedelta(days=1)
        df_today = df[(df["timestamp"] >= today_start) & (df["timestamp"] < today_end)]

        # Historische piek (wat was vandaag maximaal mogelijk volgens forecast?)
        forecast_peak_day = (
            df_today["window_avg_power"].max() if not df_today.empty else 0.0
        )

        # Hier telt huidige piek WEL mee: Hoe 'mooi' is de dag in het algemeen?
        actual_day_max = max(forecast_peak_day, median_pv)
        day_quality_ratio = actual_day_max / max(1.0, SYSTEM_MAX_KW)

        if day_quality_ratio > 0.75:
            day_type = "Sunny ☀️"
            percentage = 0.80  # Streng: Wachten op de top
        elif day_quality_ratio > 0.4:
            day_type = "Average ⛅"
            percentage = 0.60  # Soepel: Pakken wat kan
        else:
            day_type = "Gloomy ☁️"
            percentage = 0.90  # Zeer soepel: Alles is meegenomen

        # ----------------------------------------------------------------------
        # 6. TARGET BEPALING (De "Lat")
        # ----------------------------------------------------------------------

        # Gebruik een behoudende bias voor de toekomstverwachting.
        # Als bias < 1 (bewolkt), verlagen we de verwachting.
        # Als bias > 1 (zonniger), rekenen we ons niet rijk (max 1.0 of 1.1).
        conservative_future_bias = min(self.smoothed_bias, 1.1)

        # Wat is de ruwe (ongecorrigeerde) potentie in de toekomst?
        future_max_raw = future["ai_power_raw"].rolling(window=indexer).mean().max()
        if pd.isna(future_max_raw):
            future_max_raw = 0.0

        # De referentie is: Wat gaat de forecast doen, geschaald naar de realiteit van nu?
        # Dit lost het "Spook-Piek" probleem op.
        adjusted_future_max = future_max_raw * conservative_future_bias

        reference_peak = adjusted_future_max
        dynamic_threshold = reference_peak * percentage
        min_noise_limit = 0.15  # Minimaal 150W om überhaupt te overwegen

        final_trigger_val = max(dynamic_threshold, min_noise_limit)

        logger.info(
            f"SolarAI: Ref-Future: {reference_peak:.2f}kW | Drempel: {final_trigger_val:.2f}kW | Actueel: {median_pv:.2f}kW"
        )

        # ----------------------------------------------------------------------
        # 7. BESLUITVORMING
        # ----------------------------------------------------------------------

        # A. Absolute ondergrens (Nacht/Schemer)
        if best_power < min_noise_limit and median_pv < min_noise_limit:
            return {
                "action": "WAIT",
                "reason": f"[{day_type}] Te weinig licht (<{min_noise_limit*1000:.0f}W)",
                "plan_start": start_time_local,
            }

        # B. Opportunisme (Is het NU veel zonniger dan het MODEL dacht?)
        # Vergelijk huidige meting met de voorspelling van DIT specifieke slot (Appels met Appels)
        current_slot_forecast_raw = df.loc[nearest_idx, "ai_power_raw"]

        # Is het nu 20% zonniger dan voorspeld?
        is_sunny_surprise = median_pv > (current_slot_forecast_raw * 1.20)

        # EN: Is het gemiddelde voor het komende uur wel acceptabel?
        # (Niet starten op een piek van 1 minuut als de rest 0 is)
        is_viable_run = best_power > (SYSTEM_MAX_KW * 0.25)

        if is_sunny_surprise and is_viable_run:
            return {
                "action": "START",
                "reason": (
                    f"[{day_type}] Nu veel zonniger dan slot-voorspelling! "
                    f"(Actueel {median_pv:.2f}kW > Slot {current_slot_forecast_raw:.2f}kW)"
                ),
                "plan_start": datetime.now(local_tz),
            }

        # C. Normale Drempel Check
        # Starten we omdat we de dynamische drempel hebben bereikt?
        # Of omdat de toekomst slechter is dan nu (Sunset scenario)?

        if median_pv >= final_trigger_val:
            # We zitten boven de drempel. Maar is het stabiel genoeg?
            # Bij Gloomy dagen negeren we stabiliteit (pakken wat we pakken kunnen)
            if (
                "Gloomy" not in day_type
                and not is_stable
                and median_pv < (best_power * 0.9)
            ):
                return {
                    "action": "WAIT_CLOUD",
                    "reason": f"[{day_type}] Drempel bereikt, maar wacht op stabiel licht",
                    "plan_start": start_time_local,
                }

            return {
                "action": "START",
                "reason": f"[{day_type}] Actueel ({median_pv:.2f}kW) boven drempel ({final_trigger_val:.2f}kW)",
                "plan_start": datetime.now(local_tz),
            }

        # D. Wachten
        wait_min = int((best_row["timestamp"] - now_utc).total_seconds() / 60)
        return {
            "action": "WAIT",
            "reason": f"[{day_type}] Wachten op piek over {wait_min} min (Verwacht {best_power:.2f}kW)",
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
