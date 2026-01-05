import numpy as np
import pandas as pd
import joblib
import shap
import logging

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from utils import add_cyclic_time_features
from typing import Dict
from config import Config
from context import Context

logger = logging.getLogger(__name__)


class SolarStatus(Enum):
    START = "START"
    WAIT = "WAIT"
    LOW_LIGHT = "LOW_LIGHT"
    DONE = "DONE"


@dataclass
class SolarContext:
    load_now: float
    energy_now: float
    energy_best: float
    opportunity_cost: float
    confidence: float
    action: SolarStatus
    reason: str
    planned_start: Optional[datetime] = None
    current_bias: float = 1.0


class NowCaster:
    def __init__(self, model_mae: float, pv_max_kw: float, decay_hours: float = 3.0):
        self.decay_hours = decay_hours
        self.pv_max_kw = pv_max_kw
        self.current_ratio = 1.0

        # Veiligheidsmarges
        error_margin = (2.5 * model_mae) / (pv_max_kw + 0.1)
        self.max_ratio = 1.0 + error_margin
        self.min_ratio = max(0.2, 1.0 - error_margin)

    def update(self, actual_kw: float, forecasted_kw: float):
        # 1. Gating: Negeer ruis in de nacht/ochtend (<5% van systeem max)
        # We laten de ratio langzaam 'uitdoven' naar 1.0 als er geen zon is.
        if forecasted_kw < 0.05 * self.pv_max_kw:
            self.current_ratio = (0.95 * self.current_ratio) + (0.05 * 1.0)
            return

        # 2. Ratio met floor
        # Voorkom delen door 0, maar gebruik max(0.05) om extreme ratio's te dempen
        raw_ratio = actual_kw / max(forecasted_kw, 0.05)
        raw_ratio = np.clip(raw_ratio, self.min_ratio, self.max_ratio)

        # 3. Confidence Weighting (Cruciaal voor ochtendstabiliteit)
        # Bij laag vermogen (ochtend) vertrouwen we de meting minder -> lage alpha
        confidence = np.clip(forecasted_kw / (self.pv_max_kw * 0.4), 0.1, 1.0)
        alpha = 0.3 * confidence

        # Update (Low pass filter)
        self.current_ratio = (1 - alpha) * self.current_ratio + alpha * raw_ratio

    def apply(
        self, df: pd.DataFrame, now: datetime, col_name: str, actual_pv: float = None
    ) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        # Tijdsverschil
        delta_hours = (df["timestamp"] - now).dt.total_seconds() / 3600.0
        delta_hours = delta_hours.clip(lower=0)

        # 1. Asymmetrische Decay
        # Negatieve bias (wolk) vergeten we sneller dan positieve bias
        effective_decay = (
            self.decay_hours if self.current_ratio >= 1.0 else (self.decay_hours * 0.5)
        )
        decay_factors = np.exp(-delta_hours / effective_decay)

        # Basis correctie (vermenigvuldigen)
        correction_vector = 1.0 + (self.current_ratio - 1.0) * decay_factors
        corrected_series = df[col_name] * correction_vector

        # 2. Short-term Persistence (De '0-Forecast' Fix)
        # Dit repareert het gat als forecast=0 maar actual=2kW
        if actual_pv is not None and actual_pv > 0.1:
            idx_now = df["timestamp"].searchsorted(now)
            idx_now = min(idx_now, len(df) - 1)

            # Vergelijk kW met kW
            current_model_val = corrected_series.iloc[idx_now]
            error_gap = max(0, actual_pv - current_model_val)

            if error_gap > 0.1:
                # Voeg het verschil toe, uitstervend over ~1 uur
                steps = np.maximum(0, np.arange(len(df)) - idx_now)
                boost_vector = error_gap * np.exp(-steps / 4.0)
                corrected_series += boost_vector

        return corrected_series.clip(0, self.pv_max_kw)


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
            "wind",
            "cloud",
            "radiation",
            "diffuse",
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
                logger.error("[Solar] Model corrupt.")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
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
            return {"Solar": "Shap failed"}


class SolarOptimizer:
    def __init__(
        self,
        pv_max_kw: float,
        duration_hours: float,
        min_kwh_threshold: float = 0.1,
        avg_baseload_kw: float = 0.25,  # Standaard rustverbruik (koelkast, router, etc)
    ):
        self.system_max = pv_max_kw
        self.duration = duration_hours
        self.timestep_hours = 0.25
        self.min_kwh_threshold = min_kwh_threshold
        self.avg_baseload = avg_baseload_kw

    def calculate_optimal_window(
        self, df: pd.DataFrame, current_time: datetime, current_load_kw: float
    ):
        window_size = int(self.duration / self.timestep_hours)
        future = df[df["timestamp"] >= current_time].copy()

        if len(future) < window_size:
            logger.info("[Solar] Geen data meer.")
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

        # Bereken Netto [Solar] Zonne-energie MIN Huishoudelijk verbruik
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
            return SolarStatus.LOW_LIGHT, SolarContext(
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

        return status, SolarContext(
            energy_now=round(energy_now, 2),
            energy_best=round(energy_best, 2),
            opportunity_cost=round(opp_cost, 3),
            confidence=round(confidence, 2),
            action=status,
            reason=reason,
            planned_start=best_row["timestamp"],
            load_now=round(current_load_kw, 2),
        )


class SolarForecaster:
    def __init__(self, config: Config, context: Context):
        self.model = SolarModel(Path(config.solar_model_path))
        self.context = context
        self.config = config
        self.nowcaster = NowCaster(model_mae=self.model.mae, pv_max_kw=config.pv_max_kw)
        self.optimizer = SolarOptimizer(
            pv_max_kw=config.pv_max_kw,
            duration_hours=config.dhw_duration_hours,
            min_kwh_threshold=config.min_kwh_threshold,
            avg_baseload_kw=config.avg_baseload_kw,
        )

    def analyze(self, current_time: datetime, current_load_kw: float):
        forecast_df = self.context.forecast_df

        if forecast_df is None or forecast_df.empty:
            return SolarStatus.WAIT, None

        df_calc = forecast_df.copy()
        if self.model and self.model.is_fitted:
            df_calc["power_ml"] = self.model.predict(df_calc)
        else:
            # Fallback: Gebruik de ruwe schatting van de weerdienst
            df_calc["power_ml"] = df_calc["pv_estimate"].fillna(0)

        # Bias ankerpunt (nu)
        logger.info(
            f"[Solar] Huidige tijd voor analyse: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )
        idx_now = df_calc["timestamp"].searchsorted(current_time)
        row_now = df_calc.iloc[min(idx_now, len(df_calc) - 1)]

        self.nowcaster.update(self.context.stable_pv, row_now["power_ml"])

        df_calc["power_corrected"] = self.nowcaster.apply(
            df_calc, current_time, "power_ml", actual_pv=self.context.stable_pv
        )
        df_calc["power_corrected"] = df_calc["power_corrected"].clip(
            0, self.config.pv_max_kw
        )

        # 4. Optimalisatie (Geef stable_load mee)
        status, context = self.optimizer.calculate_optimal_window(
            df_calc, current_time, current_load_kw
        )

        return status, context
