import logging
import joblib
import numpy as np
import pandas as pd
import shap

from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# Machine Learning
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from pandas.api.types import is_datetime64_any_dtype

# Project Imports
from utils import add_cyclic_time_features
from ha_client import HAClient
from weather import WeatherClient
from db import fetch_solar_training_data_orm, upsert_solar_record
from utils import safe_float

logger = logging.getLogger(__name__)

# ==============================================================================
# 0. CONFIG & CONSTANTS
# ==============================================================================


class SolarStatus(Enum):
    START = "START"
    WAIT = "WAIT"
    DONE = "DONE"
    LOW_LIGHT = "LOW_LIGHT"


@dataclass
class DecisionContext:
    energy_now: float
    energy_best: float
    opportunity_cost: float
    confidence: float
    action: SolarStatus
    reason: str
    planned_start: Optional[datetime] = None
    load_now: float = 0.0


# ==============================================================================
# 1. NOWCASTER (Real-time Bias Correction)
# ==============================================================================


class NowCaster:
    def __init__(
        self, model_mae: float, system_max_kw: float, decay_hours: float = 3.0
    ):
        self.decay_hours = decay_hours
        self.current_ratio = 1.0
        # Punt 4: Data-gedreven klemmen gebaseerd op model-onzekerheid
        error_margin = (2.5 * model_mae) / (system_max_kw + 0.1)
        self.max_ratio = 1.0 + error_margin
        self.min_ratio = max(0.2, 1.0 - error_margin)

    def update(self, actual_kw: float, forecasted_kw: float):
        if forecasted_kw < 0.05:
            return
        raw_ratio = actual_kw / forecasted_kw
        raw_ratio = np.clip(raw_ratio, self.min_ratio, self.max_ratio)
        # Smoothing (0.7 / 0.3)
        self.current_ratio = (0.7 * self.current_ratio) + (0.3 * raw_ratio)

    def apply(self, df: pd.DataFrame, now: datetime, col_name: str) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        # Bereken decay vanaf NU, niet vanaf iloc[0]
        delta_hours = (df["timestamp"] - now).dt.total_seconds() / 3600.0
        delta_hours = delta_hours.clip(lower=0)

        # Bias vervaagt naarmate we verder in de toekomst kijken
        decay_factors = np.exp(-delta_hours / self.decay_hours)
        correction_vector = 1.0 + (self.current_ratio - 1.0) * decay_factors
        return (df[col_name] * correction_vector).clip(lower=0)


# ==============================================================================
# 2. MACHINE LEARNING ENGINE
# ==============================================================================


class SolarModel:
    def __init__(self, path: Path):
        self.path = path
        self.model: Optional[BaseEstimator] = None
        self.mae = 0.2
        self.feature_cols = [
            "hour_sin",
            "hour_cos",
            "doy_sin",
            "doy_cos",
            "pv_estimate",
            "pv_estimate10",
            "pv_estimate90",
            "uncertainty",
            "temp",
            "radiation",
            "diffuse",
            "cloud",
            "irradiance",
        ]
        self.is_fitted = False
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = joblib.load(self.path)
                if isinstance(data, dict):
                    self.model = data.get("model")
                    self.mae = data.get("mae", 0.2)
                else:
                    self.model = data
                self.is_fitted = True
            except Exception:
                logger.error("Solar: Model corrupt.")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = add_cyclic_time_features(df, col_name="timestamp")
        df["uncertainty"] = df["pv_estimate90"] - df["pv_estimate10"]
        X = df.reindex(columns=self.feature_cols)
        return X.apply(pd.to_numeric, errors="coerce").fillna(0)

    def train(self, df_history: pd.DataFrame, system_max: float):
        X = self._prepare_features(df_history)
        y = df_history["actual_pv_yield"].clip(0, system_max)
        self.model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.5,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
        self.model.fit(X, y)
        self.mae = mean_absolute_error(y, self.model.predict(X))
        joblib.dump({"model": self.model, "mae": self.mae}, self.path)
        self.is_fitted = True

    def predict(self, df_forecast: pd.DataFrame) -> pd.Series:
        raw_solcast = df_forecast["pv_estimate"].fillna(0)
        if not self.is_fitted:
            return raw_solcast
        X = self._prepare_features(df_forecast)
        pred_ml = np.maximum(self.model.predict(X), 0)
        # 70% ML, 30% Raw Solcast blend
        return (pred_ml * 0.7) + (raw_solcast * 0.3)

    def explain(self, df_row: pd.DataFrame) -> Dict[str, str]:
        if not self.is_fitted:
            return {"Info": "No model"}

        try:
            X = self._prepare_features(df_row)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            base = (
                explainer.expected_value
                if np.isscalar(explainer.expected_value)
                else explainer.expected_value[0]
            )

            y_pred = self.model.predict(X)

            result = {"Base": f"{base:.2f}", "Prediction": f"{y_pred[0]:.2f}"}
            for col, val in zip(self.feature_cols, shap_values[0]):
                result[col] = f"{val:+.2f}"
            return result
        except Exception:
            return {"Error": "Shap failed"}


# ==============================================================================
# 3. OPTIMIZER
# ==============================================================================


class SolarOptimizer:
    def __init__(
        self,
        system_max_kw: float,
        duration_hours: float,
        min_kwh_threshold: float = 0.3,
        avg_baseload_kw: float = 0.25,  # Standaard rustverbruik (koelkast, router, etc)
    ):
        self.system_max = system_max_kw
        self.duration = duration_hours
        self.timestep_hours = 0.25
        self.min_kwh_threshold = min_kwh_threshold
        self.avg_baseload = avg_baseload_kw

    def calculate_optimal_window(
        self, df: pd.DataFrame, current_time: datetime, current_load_kw: float
    ) -> Tuple[SolarStatus, DecisionContext]:
        # 1. Voorbereiding
        window_size = int(self.duration / self.timestep_hours)
        future = df[df["timestamp"] >= current_time].copy()

        if len(future) < window_size:
            logger.info("Solar: Geen data meer.")
            return SolarStatus.DONE, None

        # We maken een load profiel aan.
        # Voor "Nu" gebruiken we de gemeten load (bijv. wasmachine aan = 2.5kW).
        # Voor "Straks" (toekomst) nemen we aan dat de wasmachine klaar is en gebruiken we baseload.

        # We laten het hoge huidige verbruik lineair afnemen naar baseload over 45 minuten (3 kwartier)
        # Dit voorkomt dat het systeem denkt dat over 1 minuut de wasmachine spontaan uit is.
        decay_steps = 3

        future["projected_load"] = self.avg_baseload

        # Pas de eerste paar rijen aan met werkelijke load (afbouwend)
        for i in range(min(len(future), decay_steps)):
            # Lineaire interpolatie van Current Load -> Base Load
            factor = 1.0 - (i / decay_steps)
            blended_load = (current_load_kw * factor) + (
                self.avg_baseload * (1 - factor)
            )
            future.iloc[i, future.columns.get_loc("projected_load")] = max(
                blended_load, self.avg_baseload
            )

        # Bereken Netto Solar: Zonne-energie MIN Huishoudelijk verbruik
        # Als netto < 0 is (wasmachine verbruikt meer dan zon), is de 'beschikbare energie' 0.
        future["net_power"] = (
            future["power_corrected"] - future["projected_load"]
        ).clip(lower=0)

        # 2. Rolling calculations op NETTO power
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        future["rolling_energy_kwh"] = (
            future["net_power"].rolling(window=indexer).sum() * self.timestep_hours
        )

        # 3. Low Light Check: Is er vandaag genoeg OVERCAPACITEIT?
        max_attainable_energy = future["rolling_energy_kwh"].max()
        if max_attainable_energy < self.min_kwh_threshold:
            return SolarStatus.LOW_LIGHT, DecisionContext(
                energy_now=0,
                energy_best=max_attainable_energy,
                opportunity_cost=1.0,
                confidence=0,
                action=SolarStatus.LOW_LIGHT,
                reason=f"Te weinig overcapaciteit ({max_attainable_energy:.2f} kWh)",
                load_now=current_load_kw,
            )

        # 4. Score berekening
        future["rolling_uncertainty"] = (
            (future["pv_estimate90"] - future["pv_estimate10"])
            .rolling(window=indexer)
            .mean()
        )
        time_diff_hours = (
            future["timestamp"] - current_time
        ).dt.total_seconds() / 3600.0
        time_factor = np.clip(time_diff_hours / 4.0, 0.4, 1.2)
        relative_uncert = future["rolling_uncertainty"] / future[
            "rolling_energy_kwh"
        ].clip(lower=0.1)

        future["score"] = future["rolling_energy_kwh"] * (
            1.0 - relative_uncert * 0.4 * time_factor
        )

        # 5. Bepaal beste moment
        best_idx = future["score"].idxmax()
        best_row = future.loc[best_idx]
        energy_now = future["rolling_energy_kwh"].iloc[0]
        energy_best = best_row["rolling_energy_kwh"]

        # Opportunity cost
        opp_cost = (energy_best - energy_now) / max(energy_best, 0.001)

        # 6. Dynamische Confidence
        scores = future["score"].dropna()
        confidence = 0.1
        if len(scores) > 4:
            best_idx_num = scores.argmax()
            mask = np.abs(np.arange(len(scores)) - best_idx_num) > 3
            other_scores = scores[mask]
            if not other_scores.empty:
                confidence = float(
                    np.clip(
                        (scores.max() - other_scores.max()) / max(scores.max(), 0.1),
                        0.0,
                        1.0,
                    )
                )

        # Besluitvorming: drempel is strenger als de totale opbrengst laag is
        # Als energy_best laag is, moet de opp_cost NUL zijn om te starten.
        yield_weight = np.clip(
            energy_best / (self.system_max * self.duration * 0.5), 0.2, 1.0
        )
        decision_threshold = (0.02 + (0.10 * (1 - confidence))) * yield_weight

        status = SolarStatus.WAIT
        minutes_to_peak = int(
            (best_row["timestamp"] - current_time).total_seconds() / 60
        )
        reason = (
            f"Wacht op overcapaciteit ({energy_best:.2f}kWh) over {minutes_to_peak}m"
        )

        # Logica aanpassing: Als energy_now 0 is door hoge load, zal opp_cost heel hoog zijn (100%),
        # waardoor we automatisch wachten.

        if (
            opp_cost <= decision_threshold
            and energy_now >= self.min_kwh_threshold * 0.8
        ) or (opp_cost <= 0.001 and energy_now >= self.min_kwh_threshold):
            status = SolarStatus.START
            reason = f"Nu starten: verlies {opp_cost:.1%}, netto ruimte {energy_now:.2f}kWh (Load: {current_load_kw:.2f}kW)"
        elif energy_now < 0.1 and energy_best < self.min_kwh_threshold:
            status = SolarStatus.LOW_LIGHT
            reason = "Te weinig netto licht voor start"

        return status, DecisionContext(
            energy_now=round(energy_now, 2),
            energy_best=round(energy_best, 2),
            opportunity_cost=round(opp_cost, 3),
            confidence=round(confidence, 2),
            action=status,
            reason=reason,
            planned_start=best_row["timestamp"],
            load_now=round(current_load_kw, 2),
        )


# ==============================================================================
# 4. MAIN CONTROLLER
# ==============================================================================


class Solar:
    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}
        self.system_max = float(self.opts.get("system_max_kw", 2.0))
        # Config voor gemiddeld huisverbruik als apparaten uit staan
        self.avg_baseload = float(self.opts.get("avg_baseload_kw", 0.25))

        model_path = Path(
            self.opts.get("solar_model_path", "/config/models/solar_model.joblib")
        )
        self.model = SolarModel(model_path)
        self.weather = WeatherClient(opts)
        self.nowcaster = NowCaster(
            model_mae=self.model.mae, system_max_kw=self.system_max
        )
        self.optimizer = SolarOptimizer(
            self.system_max,
            float(self.opts.get("duration_hours", 1.0)),
            avg_baseload_kw=self.avg_baseload,
        )

        # Buffers voor smoothing
        self.pv_buffer = deque(maxlen=20)
        self.load_buffer = deque(maxlen=20)

        self.cached_forecast = None
        self.last_poll = None
        self.last_decision_ctx = None
        self.current_slot_start = None
        self.slot_samples = []

        logger.info("Solar: Opstarten voltooid.")

    def run_cycle(self):
        now = datetime.now(timezone.utc)

        # 1. Update PV
        val = safe_float(
            self.ha.get_state(
                self.opts.get("sensor_pv_power", "sensor.fuj7chn07b_pv_output_actual")
            )
        )
        current_pv = float(val) / 1000.0 if val else 0.0

        self.pv_buffer.append(current_pv)
        stable_pv = np.median(self.pv_buffer)

        # 1b. Update Load (Huisverbruik)
        val_load = safe_float(
            self.ha.get_state(
                self.opts.get("sensor_power_load", "sensor.stroomverbruik")
            )
        )
        current_load = float(val_load) / 1000.0 if val_load else self.avg_baseload

        #         # Filter HVAC load (indien van toepassing)
        #         hvac_mode = 'off'
        #         hvac_load = {
        #             'dwh': 3.7,
        #             'heating': 1.2
        #         }.get(hvac_mode, 0.0)
        #         current_load = max(total_load - hvac_load, 0.0)

        # Gebruik de mediaan om pieken (waterkoker) eruit te filteren
        self.load_buffer.append(current_load)
        stable_load = np.median(self.load_buffer)

        # 2. Update Forecast
        self._process_pv_sample(current_pv)
        self._update_forecast_data()
        if self.cached_forecast is None:
            logger.warning("Solar: Geen forecast data beschikbaar.")
            return

        # 3. Predictie & Nowcasting
        df_calc = self.cached_forecast.copy()
        df_calc["power_ml"] = self.model.predict(df_calc)

        # Bias ankerpunt (nu)
        idx_now = df_calc["timestamp"].searchsorted(now)
        row_now = df_calc.iloc[min(idx_now, len(df_calc) - 1)]

        self.nowcaster.update(stable_pv, row_now["power_ml"])

        df_calc["power_corrected"] = self.nowcaster.apply(df_calc, now, "power_ml")
        df_calc["power_corrected"] = df_calc["power_corrected"].clip(0, self.system_max)

        # 4. Optimalisatie (Geef stable_load mee)
        status, ctx = self.optimizer.calculate_optimal_window(df_calc, now, stable_load)

        # 5. Publiceren
        if ctx:
            self._publish_state(ctx, stable_pv)

    def train(self):
        df_history = fetch_solar_training_data_orm(days=365)
        if df_history.empty:
            return

        self.model.train(df_history, self.system_max)
        logger.info(
            f"Solar: Model getraind met MAE={self.model.mae:.2f} op {len(df_history)} records."
        )

    def _update_forecast_data(self):
        now = datetime.now(timezone.utc)
        today = now.date()

        if self.last_poll and (now - self.last_poll) < timedelta(minutes=15):
            if self.cached_forecast is not None:
                return

        payload = self.ha.get_payload(
            self.opts.get("sensor_solcast", "sensor.solcast_pv_forecast_forecast_today")
        )

        if not payload:
            logger.warning("Solar: Geen Solcast data gevonden.")
            return

        raw = payload.get("attributes", {}).get("detailedForecast", [])

        df = pd.DataFrame(raw)
        df["timestamp"] = pd.to_datetime(df["period_start"]).dt.tz_convert("UTC")
        df_sol = (
            df.set_index("timestamp")
            .apply(pd.to_numeric, errors="coerce")
            .infer_objects(copy=False)
            .resample("15min")
            .interpolate(method="linear")
            .fillna(0)
            .reset_index()
        )

        df_om = self.weather.get_forecast()

        if df_om.empty:
            logger.warning("Solar: Geen Open-Meteo data gevonden.")

        # Merge Solcast met Open-Meteo
        df_merged = pd.merge_asof(
            df_sol.sort_values("timestamp"),
            df_om.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

        df_today = df_merged[df_merged["timestamp"].dt.date == today].copy()

        if df_today.empty:
            logger.warning("Solar: Geen forecast data voor vandaag.")
            return

        for _, row in df_today.iterrows():
            ts = row["timestamp"].to_pydatetime()
            upsert_solar_record(
                ts,
                pv_estimate=float(row["pv_estimate"]),
                pv_estimate10=float(row["pv_estimate10"]),
                pv_estimate90=float(row["pv_estimate90"]),
                temp=float(row["temp"]),
                radiation=float(row["radiation"]),
                diffuse=float(row["diffuse"]),
                cloud=float(row["cloud"]),
                irradiance=float(row["irradiance"]),
            )

        self.cached_forecast = df_merged.sort_values("timestamp")
        self.last_poll = now
        logger.info("Solar: Forecast data bijgewerkt.")

    def _process_pv_sample(self, pv_kw):
        now = datetime.now(timezone.utc)
        aggregation_minutes = 15
        slot_minute = (now.minute // aggregation_minutes) * aggregation_minutes
        slot_start = now.replace(minute=slot_minute, second=0, microsecond=0)

        # Als dit de allereerste sample is
        if self.current_slot_start is None:
            self.current_slot_start = slot_start

        # Als we een nieuw kwartier zijn binnengegaan
        if slot_start > self.current_slot_start:
            if self.slot_samples:
                avg_pv = float(np.mean(self.slot_samples))
                # Sla het gemiddelde op voor het AFGELOPEN kwartier
                upsert_solar_record(self.current_slot_start, actual_pv_yield=avg_pv)
                logger.info(
                    f"Solar: Actual yield opgeslagen voor {self.current_slot_start.strftime('%H:%M')}: {avg_pv:.2f}kW"
                )

            self.slot_samples = []
            self.current_slot_start = slot_start

        self.slot_samples.append(pv_kw)

    def _publish_state(self, ctx: DecisionContext, current_pv: float):
        self.last_decision_ctx = ctx
        plan_iso = ctx.planned_start.isoformat() if ctx.planned_start else "unknown"
        logger.info(
            f"Solar: [{ctx.action.value}] {ctx.reason} | Conf: {ctx.confidence} | Load: {ctx.load_now:.2f}kW"
        )

        attrs = {
            "status": str(ctx.action.value),
            "reason": str(ctx.reason),
            "planned_start": plan_iso,
            "current_bias_ratio": float(round(self.nowcaster.current_ratio, 2)),
            "current_pv_kw": float(round(current_pv, 2)),
            "current_load_kw": float(round(ctx.load_now, 2)),
            "energy_now_kwh": float(ctx.energy_now),
            "energy_best_kwh": float(ctx.energy_best),
            "opportunity_cost": float(ctx.opportunity_cost),
            "confidence": float(ctx.confidence),
            "device_class": "timestamp",
        }
        self.ha.set_solar_prediction(plan_iso, attrs)
