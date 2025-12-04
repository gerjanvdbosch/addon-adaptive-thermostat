import os
import logging
import joblib
import numpy as np
from datetime import datetime

from db import (
    fetch_unlabeled_setpoints,
    update_setpoint,
    fetch_setpoints,
)
from collector import FEATURE_ORDER, Collector
from ha_client import HAClient
from utils import safe_round, safe_float

logger = logging.getLogger(__name__)


class InferencerDelta:
    """
    Inferencer for models trained on delta (predicted_delta).

    Target Calculation:
      predicted_setpoint = baseline_current_setpoint + pred_delta

    Features:
      We do NOT mask current_setpoint. The model needs the current state
      to determine if an adjustment (positive or negative delta) is needed.
    """

    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}
        self.model_path = self.opts.get(
            "model_path", "/config/models/full_model_delta.joblib"
        )
        self.model_payload = None

        # State tracking for inference (debounce/cooldown)
        self.last_pred_ts = None
        self.last_pred_value = None
        self.last_pred_model = None
        self.last_eval_value = None
        self.last_eval_ts = None

        # State tracking for 'implied satisfaction' (learning delta=0)
        # We track these in memory for reliability and speed
        self.stable_candidate_sp = None  # The setpoint currently being monitored
        self.stable_start_ts = None  # When this setpoint became active/stable

        self.load_model()

    def load_model(self):
        self.model_payload = None
        path = self.model_path
        if not path or not os.path.exists(path):
            logger.debug("No model found at %s", path)
            return
        try:
            payload = joblib.load(path)
        except Exception:
            logger.exception("Failed loading model file %s", path)
            return
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if meta.get("feature_order") and meta.get("feature_order") != FEATURE_ORDER:
            logger.warning("Model feature_order mismatch; ignoring model at %s", path)
            return
        if meta.get("target") and meta.get("target") != "delta":
            logger.warning("Model target != delta; ignoring model at %s", path)
            return
        if "model" not in payload or payload["model"] is None:
            logger.warning("Model payload missing model; ignoring %s", path)
            return
        self.model_payload = payload
        logger.info("Loaded model from %s (mae=%s)", path, meta.get("mae"))

    def check_and_label_user_override(self):
        """
        STRATEGY 1: Explicit User Override.
        If the user manually changed the setpoint (via HA), add a labeled sample with user_override=True.
        Record the pre-override baseline on the Setpoint row using observed_current.
        """
        try:
            now = datetime.now()
            interval = float(self.opts.get("sample_interval_seconds", 300))
            if (
                self.last_pred_ts
                and (now - self.last_pred_ts).total_seconds() < interval
            ):
                return False

            rows = fetch_setpoints(limit=1)
            if not rows:
                return False
            last_row = rows[0]

            current_sp, *_ = self.ha.get_setpoint()
            min_sp = float(self.opts.get("min_setpoint", 15.0))
            max_sp = float(self.opts.get("max_setpoint", 25.0))
            if not (min_sp <= current_sp <= max_sp):
                return False

            last_sample_sp = (
                last_row.setpoint
                if safe_float(last_row.setpoint) is not None
                else (last_row.data.get("current_setpoint") if last_row.data else None)
            )
            if last_sample_sp is None:
                return False

            current_rounded = safe_round(current_sp)
            last_sample_rounded = safe_round(last_sample_sp)
            last_pred_rounded = (
                safe_round(self.last_pred_value)
                if self.last_pred_value is not None
                else None
            )

            # If current matches DB log, no change happened
            if current_rounded == last_sample_rounded:
                return False

            # If current matches what WE predicted, it's not a user override
            if last_pred_rounded is not None and current_rounded == last_pred_rounded:
                return False

            # Persist setpoint log and ensure observed_current_setpoint is saved on the Setpoint row
            # This creates a training sample: Baseline=last_sample_sp, Target=current_rounded
            try:
                update_setpoint(
                    last_row.id,
                    setpoint=current_rounded,
                    observed_current=last_sample_sp,
                )
            except Exception:
                logger.exception("Failed updating setpoint log")

            logger.info(
                "Detected user override: inserted labeled sample %.1f (was %.1f)",
                current_rounded,
                last_sample_rounded,
            )
            return True
        except Exception:
            logger.exception("Error detecting user override")
            return False

    def check_and_label_stability(self):
        """
        STRATEGY 3: Implied Satisfaction (In-Memory Tracker).
        If the setpoint remains unchanged for 'stability_hours' AND the temperature is reached,
        we assume the user is satisfied. We label the current moment as delta=0.

        This balances the dataset so the model learns that "doing nothing" is also a valid action.
        """
        try:
            # 1. Configuration
            # Default to 6.0 hours to avoid making the model too lazy with too many '0' samples.
            stability_hours = float(self.opts.get("stability_hours", 4.0))
            temp_threshold = float(self.opts.get("stability_temp_threshold", 0.3))

            # 2. Fetch current status
            unl = fetch_unlabeled_setpoints(limit=1)
            if not unl:
                return False
            current_row = unl[0]

            feat = current_row.data
            if not feat:
                return False

            curr_sp = safe_float(feat.get("current_setpoint"))
            curr_temp = safe_float(feat.get("current_temp"))

            if curr_sp is None or curr_temp is None:
                return False

            curr_sp_rounded = safe_round(curr_sp)
            now = datetime.now()

            # 3. Logic Check

            # RESET CONDITION 1: Setpoint changed since we last checked
            if (self.stable_candidate_sp is None) or (
                safe_round(self.stable_candidate_sp) != curr_sp_rounded
            ):
                self.stable_candidate_sp = curr_sp
                self.stable_start_ts = now
                logger.info(
                    "Stability tracker started/reset for setpoint %.1f", curr_sp
                )
                return False

            # RESET CONDITION 2: Temperature is not yet reached (or overshot)
            # If room is 18C and setpoint is 20C, the system is working, not stable.
            if abs(curr_temp - curr_sp) > temp_threshold:
                # Reset timer; we only count stability starting from when the target temp is reached
                logger.info(
                    "Stability tracker paused for setpoint %.1f; temp %.1f not within threshold %.2f",
                    curr_sp,
                    curr_temp,
                    temp_threshold,
                )
                self.stable_start_ts = now
                return False

            # 4. Duration Check
            duration = (now - self.stable_start_ts).total_seconds()
            required_seconds = stability_hours * 3600

            if duration < required_seconds:
                # Not stable long enough yet
                logger.info(
                    "Stability tracker waiting: setpoint %.1f stable for %.1f/%.1f hours",
                    curr_sp,
                    (duration / 3600.0),
                    stability_hours,
                )
                return False

            # 5. Success: Stable for > X hours
            try:
                update_setpoint(
                    current_row.id,
                    setpoint=curr_sp,  # Target = Current
                    observed_current=curr_sp,  # Baseline = Current (Result: Delta 0)
                )
                logger.info(
                    "Detected stability: Setpoint %.1f stable for %.1f hours. Labeled as delta=0.",
                    curr_sp,
                    (duration / 3600.0),
                )

                # IMPORTANT: Reset timer after logging to avoid spamming '0' samples every cycle.
                # It must be stable for another full period before logging again.
                self.stable_start_ts = now

                return True
            except Exception:
                logger.exception("Failed updating stability log")
                return False

        except Exception:
            logger.exception("Error in stability check")
            # In case of error, reset state to be safe
            self.stable_candidate_sp = None
            return False

    def _fetch_current_vector_masked(self):
        """
        Fetch latest unlabelled sample and build feature vector.

        NOTE: For the Delta model, we do NOT mask 'current_setpoint'.
        The model relies on the current setpoint to calculate the relative change (delta).
        """
        try:
            unl = fetch_unlabeled_setpoints(limit=1)
            if not unl:
                return None, None
            last = unl[0]
            feat = last.data if last.data and isinstance(last.data, dict) else None
            if not feat:
                return None, None
            vec = []
            for k in FEATURE_ORDER:
                v = feat.get(k)
                if v is None:
                    vec.append(0.0)
                else:
                    try:
                        vec.append(float(v))
                    except Exception:
                        logger.debug("Coercing non-numeric feature %s to 0.0", k)
                        vec.append(0.0)
            return vec, feat
        except Exception:
            logger.exception("Failed fetching current vector")
            return None, None

    def inference_job(self):
        try:
            # 1. Check for User Override (High Priority)
            if self.check_and_label_user_override():
                return

            # 2. Check for Stability / Implied Satisfaction
            if self.check_and_label_stability():
                return
        except Exception:
            logger.exception("Error in checks; continuing")

        self.load_model()
        if self.model_payload is None:
            logger.warning("No model loaded for inference")
            return

        Xvec, featdict = self._fetch_current_vector_masked()
        if Xvec is None or featdict is None:
            logger.warning("No features available for inference")
            return
        X = np.array([Xvec], dtype=float)

        feature_sp = safe_float(featdict.get("current_setpoint"))
        current_sp, *_ = self.ha.get_setpoint()

        if current_sp is None:
            logger.warning(
                "Failed to read current setpoint from HA; skipping inference"
            )
            return

        if feature_sp is not None and abs(current_sp - feature_sp) > 0.1:
            logger.warning(
                "Mismatch! current setpoint (%.2f) != database (%.2f); skipping inference",
                current_sp,
                feature_sp,
            )
            return

        baseline = current_sp

        # Inference Parameters
        min_sp = float(self.opts.get("min_setpoint", 5.0))
        max_sp = float(self.opts.get("max_setpoint", 30.0))
        threshold = float(self.opts.get("min_change_threshold", 0.25))
        stable_seconds = float(self.opts.get("stable_seconds", 600))
        shadow_mode = self.opts.get("shadow_mode")

        # Predict
        try:
            model = self.model_payload.get("model")
            pred_raw = model.predict(X)
            pred_delta = (
                float(pred_raw[0]) if hasattr(pred_raw, "__len__") else float(pred_raw)
            )
            p = float(baseline) + pred_delta
            logger.debug(
                "Predicted delta=%.2f, reconstructed_setpoint=%.2f, current_setpoint=%.2f",
                pred_delta,
                p,
                baseline,
            )
        except Exception:
            logger.exception("Prediction failed")
            return

        # Validation
        if not np.isfinite(p):
            logger.debug("Invalid prediction")
            return
        if p < min_sp or p > max_sp:
            logger.warning("Predicted setpoint outside plausible range: %.3f", p)
            return
        p = float(max(min(p, max_sp), min_sp))
        logger.info("Prediction raw delta (%.2f)", p)
        rounded_p = safe_round(p)
        return

        # Stability Timer (prevent flip-flopping)
        # This is distinct from check_and_label_stability (which is for training)
        now = datetime.now()
        if (
            self.last_eval_value is None
            or safe_round(self.last_eval_value) != rounded_p
        ):
            self.last_eval_value = p
            self.last_eval_ts = now
            logger.info("Starting stability timer for predicted value %.2f", p)
            return
        if (now - self.last_eval_ts).total_seconds() < stable_seconds:
            logger.info(
                "Prediction %.2f not yet stable; waiting (%.0fs remaining)",
                p,
                stable_seconds - (now - self.last_eval_ts).total_seconds(),
            )
            return

        # Change Threshold (ignore micro-adjustments)
        if not shadow_mode and abs(p - float(baseline)) < threshold:
            logger.info(
                "Prediction (%.2f), change %.3f < threshold %.3f; skipping",
                p,
                abs(p - float(baseline)),
                threshold,
            )
            return

        # Cooldown Check
        cooldown = float(self.opts.get("cooldown_seconds", 3600))
        if self.last_pred_ts and (now - self.last_pred_ts).total_seconds() < cooldown:
            logger.info("Cooldown active; skipping predicted setpoint %.2f", p)
            return

        # Apply Setpoint to HA
        try:
            self.ha.set_setpoint(p)
            self.last_pred_ts = now
            self.last_pred_value = p
            self.last_pred_model = self.model_path
            logger.info("Applied predicted setpoint %.2f (baseline %.2f)", p, baseline)
        except Exception:
            logger.exception("Failed to apply setpoint via HA client")
