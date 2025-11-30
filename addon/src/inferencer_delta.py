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
    It masks current_setpoint in the feature-vector during predict so the model cannot trivially copy it.
    It reconstructs predicted_setpoint = baseline_current_setpoint + pred_delta and applies checks.
    """

    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}
        self.model_path = self.opts.get(
            "model_path", "/config/models/full_model_delta.joblib"
        )
        self.model_payload = None
        self.last_pred_ts = None
        self.last_pred_value = None
        self.last_pred_model = None
        self.last_eval_value = None
        self.last_eval_ts = None
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

            if current_rounded == last_sample_rounded:
                return False
            if last_pred_rounded is not None and current_rounded == last_pred_rounded:
                # user matched our last prediction -> not a human override
                return False

            # Persist setpoint log and ensure observed_current_setpoint is saved on the Setpoint row
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

    def _fetch_current_vector_masked(
        self,
    ):
        """
        Fetch latest unlabelled sample and build feature vector.
        Mask the current_setpoint feature (set to 0.0) so model cannot trivially echo it.
        Return both vector and original featdict (unchanged) so we can reconstruct baseline.
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
                if k == "current_setpoint":
                    vec.append(0.0)
                    continue
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
            logger.exception("Failed fetching current vector (masked)")
            return None, None

    def inference_job(self):
        try:
            if self.check_and_label_user_override():
                return
        except Exception:
            logger.exception("Error in override check; continuing")

        self.load_model()
        if self.model_payload is None:
            logger.debug("No model loaded for inference")
            return

        Xvec, featdict = self._fetch_current_vector_masked()
        if Xvec is None:
            logger.debug("No features available for inference")
            return
        X = np.array([Xvec], dtype=float)

        # prefer baseline from featdict
        baseline_raw = None
        if featdict:
            baseline_raw = featdict.get(
                "observed_current_setpoint", featdict.get("current_setpoint")
            )

        # fallback: try latest setpoint log row
        if baseline_raw is None:
            try:
                rows = fetch_setpoints(1)
                if rows:
                    sp_row = rows[0]
                    if getattr(sp_row, "observed_current_setpoint", None) is not None:
                        baseline_raw = getattr(sp_row, "observed_current_setpoint")
                    else:
                        baseline_raw = (
                            sp_row.data.get("current_setpoint")
                            if isinstance(sp_row.data, dict)
                            else None
                        )
            except Exception:
                logger.exception("Failed fetching fallback setpoint row for baseline")

        baseline = None
        try:
            if baseline_raw is not None:
                baseline = safe_float(baseline_raw)
        except Exception:
            logger.exception("safe_float failed on baseline_raw=%r", baseline_raw)
            baseline = None

        if baseline is None:
            logger.warning(
                "No usable baseline found (baseline_raw=%r); skipping inference",
                baseline_raw,
            )
            return

        min_sp = float(self.opts.get("min_setpoint", 5.0))
        max_sp = float(self.opts.get("max_setpoint", 30.0))
        threshold = float(self.opts.get("min_change_threshold", 0.25))
        stable_seconds = float(self.opts.get("stable_seconds", 600))
        shadow_mode = self.opts.get("shadow_mode")

        # debug
        logger.debug(
            "DEBUG: featdict keys = %s", sorted(featdict.keys()) if featdict else None
        )
        logger.debug("DEBUG: Xvec = %s", Xvec)
        logger.debug("DEBUG: model type = %s", type(self.model_payload.get("model")))

        try:
            model = self.model_payload.get("model")
            pred_raw = model.predict(X)
            pred_delta = (
                float(pred_raw[0]) if hasattr(pred_raw, "__len__") else float(pred_raw)
            )
            p = float(baseline) + pred_delta
            logger.debug(
                "DEBUG: predicted_delta=%.2f reconstructed_setpoint=%.2f raw=%s",
                pred_delta,
                p,
                pred_raw,
            )
        except Exception:
            logger.exception("Prediction failed")
            return

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

        # stability timer
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

        # threshold and cooldown
        if not shadow_mode and abs(p - float(baseline)) < threshold:
            logger.info(
                "Prediction (%.2f), change %.3f < threshold %.3f; skipping",
                p,
                abs(p - float(baseline)),
                threshold,
            )
            return
        cooldown = float(self.opts.get("cooldown_seconds", 3600))
        if self.last_pred_ts and (now - self.last_pred_ts).total_seconds() < cooldown:
            logger.info("Cooldown active; skipping predicted setpoint %.2f", p)
            return

        # apply setpoint in HA
        try:
            self.ha.set_setpoint(p)
            self.last_pred_ts = now
            self.last_pred_value = p
            self.last_pred_model = self.model_path
            logger.info("Applied predicted setpoint %.2f (baseline %.2f)", p, baseline)
        except Exception:
            logger.exception("Failed to apply setpoint via HA client")
