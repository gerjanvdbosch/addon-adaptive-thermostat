import logging
import joblib
import numpy as np
import pandas as pd
import shap

from datetime import datetime, timezone
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
                logger.error("SolarAI: Model corrupt.")

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
            loss="absolute_error", random_state=42
        )
        self.model.fit(X, y)
        self.mae = mean_absolute_error(y, self.model.predict(X))
        joblib.dump({"model": self.model, "mae": self.mae}, self.path)
        self.is_fitted = True

    def predict(self, df_forecast: pd.DataFrame) -> pd.Series:
        raw_solcast = df_forecast["pv_estimate90"].fillna(0)
        if not self.is_fitted:
            return raw_solcast
        X = self._prepare_features(df_forecast)
        pred_ml = np.maximum(self.model.predict(X), 0)
        # 60% ML, 40% Raw Solcast blend
        return (pred_ml * 0.6) + (raw_solcast * 0.4)

    def get_shap_values(self, df_row: pd.DataFrame, solcast_col: str) -> Dict[str, str]:
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

            result = {"Base": f"{base:.2f}"}
            for col, val in zip(self.feature_cols, shap_values[0]):
                if abs(val) > 0.02:
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
    ):
        self.system_max = system_max_kw
        self.duration = duration_hours
        self.timestep_hours = 0.25
        self.min_kwh_threshold = (
            min_kwh_threshold  # Minimale energie om een start te rechtvaardigen
        )

    def calculate_optimal_window(
        self, df: pd.DataFrame, current_time: datetime
    ) -> Tuple[SolarStatus, DecisionContext]:
        # 1. Resample en voorbereiding
        df_res = (
            df.set_index("timestamp")
            .infer_objects(copy=False)
            .resample("15min")
            .interpolate("linear")
            .reset_index()
        )
        window_size = int(self.duration / self.timestep_hours)
        future = df_res[df_res["timestamp"] >= current_time].copy()

        if len(future) < window_size:
            return SolarStatus.DONE, None

        # 2. Rolling calculations
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        future["rolling_energy_kwh"] = (
            future["power_corrected"].rolling(window=indexer).sum()
            * self.timestep_hours
        )

        # 3. Low Light Check: Is er vandaag überhaupt genoeg zon?
        max_attainable_energy = future["rolling_energy_kwh"].max()
        if max_attainable_energy < self.min_kwh_threshold:
            return SolarStatus.LOW_LIGHT, DecisionContext(
                energy_now=0,
                energy_best=max_attainable_energy,
                opportunity_cost=1.0,
                confidence=0,
                action=SolarStatus.LOW_LIGHT,
                reason=f"Te weinig zon verwacht ({max_attainable_energy:.2f} kWh)",
            )

        # 4. Score berekening met onzekerheidsstraf
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

        # Opportunity cost (voorkom delen door nul bij nacht/donker)
        opp_cost = (energy_best - energy_now) / max(energy_best, 0.001)

        # 6. Dynamische Confidence & Decision Logic
        # Saliency check
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
        reason = f"Wacht op piek ({energy_best:.2f}kWh) over {minutes_to_peak}m"

        # Start condities:
        # - We zitten binnen de verliesmarge EN de huidige opbrengst is substantieel
        # - Of we zitten op de absolute piek van de dag
        if (
            opp_cost <= decision_threshold
            and energy_now >= self.min_kwh_threshold * 0.8
        ) or (opp_cost <= 0.001 and energy_now >= self.min_kwh_threshold):
            status = SolarStatus.START
            reason = (
                f"Nu starten: verlies {opp_cost:.1%}, opbrengst {energy_now:.2f}kWh"
            )
        elif energy_now < 0.05 and energy_best < self.min_kwh_threshold:
            status = SolarStatus.LOW_LIGHT
            reason = "Te weinig licht voor zinvolle start"

        return status, DecisionContext(
            energy_now=round(energy_now, 2),
            energy_best=round(energy_best, 2),
            opportunity_cost=round(opp_cost, 3),
            confidence=round(confidence, 2),
            action=status,
            reason=reason,
            planned_start=best_row["timestamp"],
        )


# ==============================================================================
# 4. MAIN CONTROLLER
# ==============================================================================


class SolarAI2:
    def __init__(self, ha_client: HAClient, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}
        self.system_max = float(self.opts.get("system_max_kw", 2.0))

        model_path = Path(
            self.opts.get("solar_model_path", "/config/models/solar_model.joblib")
        )
        self.model = SolarModel(model_path)
        self.nowcaster = NowCaster(
            model_mae=self.model.mae, system_max_kw=self.system_max
        )
        self.optimizer = SolarOptimizer(
            self.system_max, float(self.opts.get("duration_hours", 1.0))
        )

        self.pv_buffer = deque(maxlen=20)
        self.cached_forecast = None
        self.last_poll = None

    def run_cycle(self):
        now = datetime.now(timezone.utc)

        # 1. Update PV
        try:
            val = self.ha.get_state(
                self.opts.get("sensor_pv_power", "sensor.fuj7chn07b_pv_output_actual")
            )
            current_pv = (
                float(val) / 1000.0
                if val not in ["unknown", "unavailable", None]
                else 0.0
            )
        except Exception:
            current_pv = 0.0
        self.pv_buffer.append(current_pv)

        # 2. Update Forecast
        self._update_forecast_data()
        if self.cached_forecast is None:
            return

        # 3. Predictie & Nowcasting
        df_calc = self.cached_forecast.copy()
        df_calc["power_ml"] = self.model.predict(df_calc)

        # Bias ankerpunt (nu)
        idx_now = df_calc["timestamp"].searchsorted(now)
        row_now = df_calc.iloc[min(idx_now, len(df_calc) - 1)]

        stable_pv = np.median(self.pv_buffer)
        self.nowcaster.update(stable_pv, row_now["power_ml"])

        df_calc["power_corrected"] = self.nowcaster.apply(df_calc, now, "power_ml")
        df_calc["power_corrected"] = df_calc["power_corrected"].clip(0, self.system_max)

        # 4. Optimalisatie
        status, ctx = self.optimizer.calculate_optimal_window(df_calc, now)

        # 5. Publiceren
        if ctx:
            self._publish_state(ctx, stable_pv)

    def _update_forecast_data(self):
        poll = self.ha.get_state(
            self.opts.get(
                "sensor_solcast_poll", "sensor.solcast_pv_forecast_api_last_polled"
            )
        )
        if poll == self.last_poll and self.cached_forecast is not None:
            return

        payload = self.ha.get_payload(
            self.opts.get(
                "sensor_solcast", "sensor.solcast_pv_forecast_forecast_today"
            )
        )
        if payload:
            raw = payload.get("attributes", {}).get("detailedForecast", [])
            df = pd.DataFrame(raw)
            df["timestamp"] = pd.to_datetime(df["period_start"]).dt.tz_convert("UTC")
            self.cached_forecast = df.sort_values("timestamp")
            self.last_poll = poll

    def _publish_state(self, ctx: DecisionContext, current_pv: float):
        plan_iso = ctx.planned_start.isoformat() if ctx.planned_start else None
        logger.info(
            f"SolarML: [{ctx.action.value}] {ctx.reason} | Conf: {ctx.confidence}"
        )

        attrs = {
            "status": ctx.action.value,
            "reason": ctx.reason,
            "planned_start": plan_iso,
            "current_bias_ratio": round(self.nowcaster.current_ratio, 2),
            "current_pv_kw": round(current_pv, 2),
            "energy_now_kwh": ctx.energy_now,
            "energy_best_kwh": ctx.energy_best,
            "opportunity_cost": ctx.opportunity_cost,
            "confidence": ctx.confidence,
            "device_class": "timestamp",
        }
        self.ha.set_solar_prediction(plan_iso, attrs)
