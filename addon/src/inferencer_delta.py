import os
import logging
import joblib
import numpy as np  # <--- TOEGEVOEGD

from datetime import datetime
from db import (
    insert_setpoint,
)
from collector import Collector, FEATURE_ORDER  # <--- FEATURE_ORDER TOEGEVOEGD
from ha_client import HAClient
from utils import safe_round, safe_float
from trainer_delta import TrainerDelta

logger = logging.getLogger(__name__)


class InferencerDelta:
    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}
        self.collector = collector
        self.trainer = TrainerDelta(ha_client, opts)

        self.model_path = self.opts.get(
            "model_path", "/config/models/delta_model.joblib"
        )
        self.model = None

        # State Tracking
        self.last_known_setpoint = None
        self.last_ai_prediction = None
        self.last_ai_action_ts = None
        self.stability_start_ts = None
        self.last_run_ts = None

        self._load_model()
        self._init_state()

    def _init_state(self):
        sp = self.ha.get_shadow_setpoint()
        if sp is not None:
            self.last_known_setpoint = safe_round(sp)
        logger.info(
            f"Initialized state. Last known setpoint: {self.last_known_setpoint}"
        )

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning("No model found at %s", self.model_path)
            return
        try:
            payload = joblib.load(self.model_path)
            self.model = payload.get("model")
            logger.info("Model loaded successfully.")
        except Exception:
            logger.exception("Failed loading model")

    def run_cycle(self):
        ts = datetime.now()
        cooldown_hours = float(self.opts.get("cooldown_hours", 1))
        cooldown_seconds = cooldown_hours * 3600
        threshold = float(self.opts.get("min_change_threshold", 0.25))

        if self.last_run_ts and (ts - self.last_run_ts).total_seconds() < 5:
            logger.info("Cycle ran too recently, skipping.")
            return
        self.last_run_ts = ts

        # 1. Lees ruwe data
        try:
            raw_data = self.collector.read_sensors()
            raw_data["current_setpoint"] = self.ha.get_shadow_setpoint()
        except Exception:
            logger.exception("Sensor read failed")
            return

        curr_sp = safe_float(raw_data.get("current_setpoint"))
        curr_temp = safe_float(raw_data.get("current_temp"))

        if curr_sp is None:
            logger.info("Current setpoint is None, skipping cycle.")
            return

        curr_sp_rounded = safe_round(curr_sp)

        # --- FASE 2: VERANDERING DETECTIE ---
        if (
            self.last_known_setpoint is not None
            and curr_sp_rounded != self.last_known_setpoint
        ):
            value_match = (
                self.last_ai_prediction is not None
                and self.last_ai_prediction == curr_sp_rounded
            )

            is_recent = False
            if self.last_ai_action_ts is not None:
                delta_t = (ts - self.last_ai_action_ts).total_seconds()
                if delta_t < cooldown_seconds:
                    is_recent = True

            is_ai_work = value_match and is_recent

            if is_ai_work:
                logger.info(f"Confirmed AI setpoint update to {curr_sp_rounded}")
            else:
                # USER OVERRIDE!
                prev_sp = self.last_known_setpoint
                logger.info(f"User Override Detected: {prev_sp} -> {curr_sp_rounded}")

                features = self.collector.features_from_raw(
                    raw_data, timestamp=ts, override_setpoint=prev_sp
                )

                try:
                    insert_setpoint(
                        features, setpoint=curr_sp_rounded, observed_current=prev_sp
                    )
                    logger.info("Saved labeled training sample (User Override).")
                except Exception:
                    logger.exception("DB Save failed")

                self.last_ai_action_ts = ts
                self.trainer.train_job(force=True)
                logger.info("Retraining complete.")
                self._load_model()

            self.last_known_setpoint = curr_sp_rounded
            self.stability_start_ts = None
            return

        if self.last_known_setpoint is None:
            self.last_known_setpoint = curr_sp_rounded

        # --- FASE 3: STABILITEIT CHECK ---
        features = self.collector.features_from_raw(raw_data, timestamp=ts)

        is_stable_temp = (
            curr_temp is not None and curr_sp is not None and curr_temp >= curr_sp
        )

        if is_stable_temp:
            if self.stability_start_ts is None:
                self.stability_start_ts = ts
                logger.info("Stable temperature detected, starting stability timer.")
            else:
                duration = (ts - self.stability_start_ts).total_seconds()
                hours_required = float(self.opts.get("stability_hours", 8.0))

                if duration > (hours_required * 3600):
                    logger.info(
                        f"Stability detected ({duration/3600:.1f}h). User implies satisfaction."
                    )
                    try:
                        insert_setpoint(
                            features,
                            setpoint=curr_sp_rounded,
                            observed_current=curr_sp_rounded,
                        )
                        self.stability_start_ts = ts
                    except Exception:
                        logger.exception("Stability log failed")
                else:
                    logger.info(
                        f"Stable temp detected, but duration {duration/3600:.1f}h < {hours_required:.1f}h required."
                    )
        else:
            self.stability_start_ts = None
            logger.info("Temperature not stable, resetting stability timer.")

        # --- FASE 4: AI INFERENCE ---
        if not self.model:
            return

        # Cooldown check voor NIEUWE acties
        if self.last_ai_action_ts is not None:
            time_since_last = (ts - self.last_ai_action_ts).total_seconds()
            if time_since_last < cooldown_seconds:
                # We loggen dit niet als warning, maar als info/debug om spam te voorkomen
                # of we returnen gewoon stilzwijgend.
                wait_hours = time_since_last / 3600
                logger.info(
                    f"Cooldown active ({wait_hours:.1f}h < {cooldown_hours:.1f}h), skipping AI action."
                )
                return

        try:
            # --- BELANGRIJKE AANPASSING HIERONDER ---
            # We bouwen de feature vector handmatig om consistentie met de Trainer te garanderen.
            # De trainer gebruikt np.nan, de collector doet misschien 0.0 of None.
            vec = []
            for k in FEATURE_ORDER:
                v = features.get(k)
                if v is None:
                    vec.append(np.nan)  # Gebruik NaN net als in de trainer
                else:
                    try:
                        vec.append(float(v))
                    except Exception:
                        vec.append(np.nan)

            # Maak er een 2D array van (1 rij, N kolommen) voor scikit-learn
            X = np.array([vec], dtype=float)

            # Predictie (Delta)
            pred_delta = float(self.model.predict(X)[0])
            new_target = curr_sp + pred_delta

            # Limits & Thresholds
            min_sp = float(self.opts.get("min_setpoint", 15.0))
            max_sp = float(self.opts.get("max_setpoint", 25.0))
            new_target = max(min(new_target, max_sp), min_sp)

            if abs(new_target - curr_sp) >= threshold:
                logger.info(
                    f"AI suggests change: {curr_sp} -> {new_target:.2f} (Delta {pred_delta:.2f})"
                )
                self.ha.set_setpoint(new_target)
                self.last_ai_prediction = safe_round(new_target)
                self.last_known_setpoint = safe_round(new_target)
                self.stability_start_ts = None
                self.last_ai_action_ts = ts
            else:
                logger.info(
                    f"AI change {new_target:.2f} within threshold {threshold}, no action taken."
                )

        except Exception:
            logger.exception("Inference failed")
