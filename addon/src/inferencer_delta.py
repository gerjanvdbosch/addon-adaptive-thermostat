import os
import logging
import joblib
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List

from db import (
    fetch_unlabeled,
    update_sample_prediction,
    insert_sample,
    fetch,
    insert_setpoint,
)
from collector import FEATURE_ORDER, Collector
from ha_client import HAClient
from utils import safe_round

logger = logging.getLogger(__name__)


class InferencerDelta:
    """
    Inferencer for models trained on delta (predicted_delta).
    It masks current_setpoint in the feature-vector during predict so the model cannot trivially copy it.
    It reconstructs predicted_setpoint = current_setpoint + pred_delta and applies checks.
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

    def check_and_label_user_override(self) -> bool:
        """
        If the user manually changed the setpoint (via HA), add a labeled sample with user_override=True.
        This function should only label when it's a genuine user action (not when our model applied it).
        """
        try:
            now = datetime.now()
            interval = float(self.opts.get("sample_interval_seconds", 300))
            if (
                self.last_pred_ts
                and (now - self.last_pred_ts).total_seconds() < interval
            ):
                return False

            rows = fetch(limit=1)
            if not rows:
                return False
            last_row = rows[0]
            current_sp, *_ = self.ha.get_setpoint()
            min_sp = float(self.opts.get("min_setpoint", 5.0))
            max_sp = float(self.opts.get("max_setpoint", 30.0))
            if not (min_sp <= current_sp <= max_sp):
                return False

            last_sample_sp = (
                last_row.data.get("current_setpoint") if last_row.data else None
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

            features = self.collector.get_features(ts=now)
            insert_sample(features, label_setpoint=current_sp, user_override=True)

            features["current_setpoint"] = last_sample_sp
            insert_setpoint(features, setpoint=current_sp)
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
    ) -> Tuple[Optional[List[float]], Optional[dict]]:
        """
        Fetch latest unlabelled sample and build feature vector.
        Mask the current_setpoint feature (set to 0.0) so model cannot trivially echo it.
        Return both vector and original featdict (unchanged) so we can reconstruct current_setpoint.
        """
        try:
            unl = fetch_unlabeled(limit=1)
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

        current_sp = featdict.get("current_setpoint") if featdict else None
        if current_sp is None:
            logger.warning("Current setpoint not available; skipping inference")
            return

        min_sp = float(self.opts.get("min_setpoint", 5.0))
        max_sp = float(self.opts.get("max_setpoint", 30.0))
        threshold = float(self.opts.get("min_change_threshold", 0.25))
        stable_seconds = float(self.opts.get("stable_seconds", 600))

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
            p = float(current_sp) + pred_delta
            logger.info(
                "DEBUG: predicted_delta=%.4f reconstructed_setpoint=%.4f", pred_delta, p
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
        rounded_p = safe_round(p)

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
        if abs(p - float(current_sp)) < threshold:
            logger.info(
                "Prediction (%.2f), change %.3f < threshold %.3f; skipping",
                p,
                abs(p - float(current_sp)),
                threshold,
            )
            return
        cooldown = float(self.opts.get("cooldown_seconds", 3600))
        if self.last_pred_ts and (now - self.last_pred_ts).total_seconds() < cooldown:
            logger.info("Cooldown active; skipping predicted setpoint %.2f", p)
            return

        # persist prediction (absolute)
        try:
            unl = fetch_unlabeled(limit=1)
            if unl:
                latest = unl[0]
                sample_ts = getattr(latest, "timestamp", None)
                age = (now - sample_ts).total_seconds() if sample_ts else float("inf")
                age_thresh = float(self.opts.get("sample_age_threshold", 300))
                if age <= age_thresh:
                    update_sample_prediction(
                        latest.id, predicted_setpoint=p, prediction_error=None
                    )
                else:
                    features = self.collector.get_features(ts=now)
                    sid = insert_sample(features)
                    update_sample_prediction(
                        sid, predicted_setpoint=p, prediction_error=None
                    )
            else:
                features = self.collector.get_features(ts=now)
                sid = insert_sample(features)
                update_sample_prediction(
                    sid, predicted_setpoint=p, prediction_error=None
                )
        except Exception:
            logger.exception("Failed to persist prediction")

        # apply setpoint in HA
        try:
            self.ha.set_setpoint(p)
            self.last_pred_ts = now
            self.last_pred_value = p
            self.last_pred_model = self.model_path
            logger.info("Applied predicted setpoint %.2f (was %.2f)", p, current_sp)
        except Exception:
            logger.exception("Failed to apply setpoint via HA client")
