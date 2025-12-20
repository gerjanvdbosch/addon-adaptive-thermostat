import os
import logging
import joblib
import time
import math
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

# Machine Learning
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Project imports
from db import fetch_training_setpoints, insert_setpoint
from collector import Collector, FEATURE_ORDER
from ha_client import HAClient
from utils import safe_round, safe_float

logger = logging.getLogger(__name__)

class ThermostatAI:
    """
    Het centrale brein van de slimme thermostaat.
    Pandas-versie: Robuuster door gebruik van kolomnamen i.p.v. indexen.
    """

    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}

        # Config
        self.model_path = Path(self.opts.get("model_path", "/config/models/delta_model.joblib"))
        self.random_state = int(self.opts.get("random_state", 42))
        self.feature_columns = FEATURE_ORDER # We gebruiken dit om de DF te filteren

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Runtime State
        self.model = None
        self.last_known_setpoint = None
        self.last_ai_prediction = None
        self.last_ai_action_ts = None
        self.stability_start_ts = None
        self.last_run_ts = None

        self._load_model()
        self._init_runtime_state()

    # ==============================================================================
    # 1. HELPERS & LOADING
    # ==============================================================================

    def _init_runtime_state(self):
        sp = self.ha.get_shadow_setpoint()
        if sp is not None:
            self.last_known_setpoint = safe_round(sp)

    def _load_model(self):
        if self.model_path.exists():
            try:
                payload = joblib.load(self.model_path)
                self.model = payload.get("model")
                logger.info("ThermostatAI: Model loaded (Pandas-ready).")
            except Exception:
                logger.exception("ThermostatAI: Failed to load model")

    def _atomic_save(self, model, meta):
        tmp_path = self.model_path.with_suffix(".tmp")
        try:
            payload = {"model": model, "meta": meta}
            joblib.dump(payload, tmp_path)
            tmp_path.replace(self.model_path)
            logger.info("ThermostatAI: Model saved.")
        except Exception:
            logger.exception("ThermostatAI: Save failed")

    # ==============================================================================
    # 2. TRAINING (Pandas Style)
    # ==============================================================================

    def _prepare_training_data(self):
        """
        Haalt data op en zet om naar DataFrame.
        Veel sneller en leesbaarder dan loops.
        """
        days = int(self.opts.get("buffer_days", 30))
        rows = fetch_training_setpoints(days=days)

        # We bouwen een lijst van dicts (DataFrames bouwen in een loop is traag)
        data_list = []

        for r in rows:
            # Sla over als data corrupt is
            if not r.data or not isinstance(r.data, dict):
                continue

            label = safe_float(getattr(r, "setpoint", None))
            if label is None: continue

            # Baseline bepalen
            curr_raw = getattr(r, "observed_current_setpoint", None) or r.data.get("current_setpoint")
            baseline = safe_float(curr_raw)
            if baseline is None: continue

            # Voeg target info toe aan de feature dict
            # We maken een kopie van de dict om de originele data niet te vervuilen
            row_data = r.data.copy()
            row_data["_target_label"] = label
            row_data["_baseline"] = baseline

            data_list.append(row_data)

        if not data_list:
            return None, None

        # 1. Maak DataFrame
        df = pd.DataFrame(data_list)

        # 2. Bereken Delta (Vectorized = Snel)
        df["delta"] = df["_target_label"] - df["_baseline"]

        # 3. Filter onzin (Sanity Check)
        # Bijv: temperatuurverschillen van > 10 graden in 1x zijn waarschijnlijk fouten
        df = df[(df["delta"] > -10) & (df["delta"] < 10)]
        df = df.dropna(subset=["delta"]) # Gooi rijen weg waar delta mislukte

        # 4. Selecteer Features (X) en Target (y)
        # Zorg dat alle features numeriek zijn (coerce errors to NaN)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan # Vul missende kolommen
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        X = df[self.feature_columns]
        y = df["delta"]

        return X, y

    def train(self, force=False):
        logger.info("ThermostatAI: Starting training...")
        start = time.time()

        X, y = self._prepare_training_data()

        if X is None or len(X) < 10:
            logger.info("ThermostatAI: Too few samples (<10).")
            return

        # Train/Val Split (Pandas slicing)
        split_idx = int(len(X) * 0.85)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]

        # Model Config
        new_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=2000,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=None, # We leveren handmatig X_val
            n_iter_no_change=20,
            random_state=self.random_state
        )

        try:
            # We geven de DataFrame direct aan fit().
            # Het model slaat nu de KOLOMNAMEN op.
            new_model.fit(X_train, y_train, val_set=(X_val, y_val))
        except Exception:
            logger.exception("ThermostatAI: Training crash")
            return

        # Score
        mae = 0.0
        if not X_val.empty:
            mae = mean_absolute_error(y_val, new_model.predict(X_val))

        # Opslaan
        runtime = time.time() - start
        meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "mae": float(mae),
            "samples": len(X),
            "feature_names": self.feature_columns # Handig voor debuggen
        }

        self.model = new_model
        self._atomic_save(new_model, meta)
        logger.info(f"ThermostatAI: Training done. MAE={mae:.3f} (Time: {runtime:.2f}s)")

    # ==============================================================================
    # 3. INFERENCE (Pandas Style)
    # ==============================================================================

    def _predict_delta(self, features: dict):
        if self.model is None:
            return 0.0

        # Maak 1-row DataFrame
        # Dit zorgt dat de kolomnamen matchen met training data
        df_input = pd.DataFrame([features])

        # Zorg voor dezelfde kolommen (vult missende met NaN)
        # Dit is de kracht van Pandas: volgorde maakt niet uit, namen wel!
        for col in self.feature_columns:
            if col not in df_input.columns:
                df_input[col] = np.nan

        # Filter op exact de juiste kolommen (volgorde forceren voor zekerheid)
        df_input = df_input[self.feature_columns]

        # Maak alles numeriek
        df_input = df_input.apply(pd.to_numeric, errors='coerce')

        try:
            return float(self.model.predict(df_input)[0])
        except Exception:
            logger.exception("ThermostatAI: Prediction error")
            return 0.0

    def run_cycle(self):
        # ... (De logica van run_cycle blijft exact hetzelfde als voorheen) ...
        # ... Alleen de interne aanroep naar _predict_delta is nu robuuster ...

        ts = datetime.now()
        if self.last_run_ts and (ts - self.last_run_ts).total_seconds() < 5:
            return
        self.last_run_ts = ts

        try:
            raw = self.collector.read_sensors()
            raw["current_setpoint"] = self.ha.get_shadow_setpoint()
        except: return

        curr_sp = safe_float(raw.get("current_setpoint"))
        curr_temp = safe_float(raw.get("current_temp"))
        if curr_sp is None: return
        curr_sp_rounded = safe_round(curr_sp)

        # CHANGE DETECTION
        if self.last_known_setpoint is not None and curr_sp_rounded != self.last_known_setpoint:
            cooldown = float(self.opts.get("cooldown_hours", 1)) * 3600
            is_recent_ai = self.last_ai_action_ts and (ts - self.last_ai_action_ts).total_seconds() < cooldown
            is_ai_val = self.last_ai_prediction is not None and self.last_ai_prediction == curr_sp_rounded

            if is_ai_val and is_recent_ai:
                logger.info("AI Action Confirmed.")
            else:
                prev_sp = self.last_known_setpoint
                logger.info(f"User Override: {prev_sp} -> {curr_sp_rounded}")
                feats = self.collector.features_from_raw(raw, timestamp=ts, override_setpoint=prev_sp)
                insert_setpoint(feats, setpoint=curr_sp_rounded, observed_current=prev_sp)
                self.last_ai_action_ts = ts
                self.train(force=True)

            self.last_known_setpoint = curr_sp_rounded
            self.stability_start_ts = None
            return

        if self.last_known_setpoint is None:
            self.last_known_setpoint = curr_sp_rounded

        # STABILITY
        feats = self.collector.features_from_raw(raw, timestamp=ts)
        is_stable = curr_temp is not None and curr_temp >= curr_sp

        if is_stable:
            if self.stability_start_ts is None: self.stability_start_ts = ts
            else:
                hours = (ts - self.stability_start_ts).total_seconds() / 3600
                if hours > float(self.opts.get("stability_hours", 8.0)):
                    logger.info("Stability logged.")
                    insert_setpoint(feats, setpoint=curr_sp_rounded, observed_current=curr_sp_rounded)
                    self.stability_start_ts = ts
        else:
            self.stability_start_ts = None

        # INFERENCE
        if not self.model: return
        cooldown = float(self.opts.get("cooldown_hours", 1)) * 3600
        if self.last_ai_action_ts and (ts - self.last_ai_action_ts).total_seconds() < cooldown:
            return

        pred_delta = self._predict_delta(feats) # <--- Nu via Pandas
        new_target = curr_sp + pred_delta

        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 25.0))
        new_target = max(min(new_target, max_sp), min_sp)

        if abs(new_target - curr_sp) >= float(self.opts.get("min_change_threshold", 0.25)):
            logger.info(f"AI Suggests: {curr_sp} -> {new_target:.2f}")
            self.ha.set_setpoint(new_target)
            self.last_ai_prediction = safe_round(new_target)
            self.last_known_setpoint = safe_round(new_target)
            self.last_ai_action_ts = ts
            self.stability_start_ts = None