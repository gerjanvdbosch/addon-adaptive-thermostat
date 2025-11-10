import os
import logging
import joblib
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List

from db import fetch_unlabeled, update_sample_prediction, insert_sample
from collector import FEATURE_ORDER, Collector
from ha_client import HAClient
from utils import safe_round

logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}
        if not self.opts.get("model_path_full") or not self.opts.get(
            "model_path_partial"
        ):
            raise RuntimeError(
                "model_path_full and model_path_partial must be provided in opts."
            )
        self.model_obj = None
        self.last_pred_ts: Optional[datetime] = None
        self.last_pred_value: Optional[float] = None
        self.load_model()
    
    def check_and_label_user_override(self) -> bool:
        rows = fetch_unlabeled(limit=1)
        if not rows:
            return False
        row = rows[0]
    
        current_sp, _ = self.ha.get_setpoint()
        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 24.0))
        if current_sp < min_sp or current_sp > max_sp:
            logger.warning("Setpoint outside plausible range: %s", current_sp)
            return False
            
        current_rounded = round(current_sp, 1)
        predicted = getattr(row, "predicted_setpoint", None)
        sample_sp = row.data.get("features", {}).get("current_setpoint") if row.data else None
        sample_rounded = round(sample_sp, 1) if sample_sp is not None else None
    
        if predicted is not None and sample_rounded == round(predicted, 1):
            return False
    
        if sample_rounded is not None and sample_rounded != current_rounded:
            features = self.collector.get_features()
            insert_sample(
                {"features": features},
                label_setpoint=current_sp,
                user_override=True
            )
            logger.debug(
                "Labeled sample as user_override: sample %.1f != current %.1f",
                sample_rounded,
                current_rounded,
            )
            return True
    
        return False


    def load_model(self):
        try:
            full_path = self.opts.get("model_path_full")
            partial_path = self.opts.get("model_path_partial")
            if full_path and os.path.exists(full_path):
                self.model_obj = joblib.load(full_path)
                meta = self.model_obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logger.warning(
                        "Full model feature_order mismatch; ignoring full model"
                    )
                    self.model_obj = None
                else:
                    logger.info("Loaded full model from %s", full_path)
                    return
            if partial_path and os.path.exists(partial_path):
                self.model_obj = joblib.load(partial_path)
                meta = self.model_obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logger.warning(
                        "Partial model feature_order mismatch; ignoring partial model"
                    )
                    self.model_obj = None
                else:
                    logger.info("Loaded partial model from %s", partial_path)
                    return
            logger.info(
                "No compatible model loaded; inference will be skipped until a model is available"
            )
            self.model_obj = None
        except Exception:
            logger.exception("Error loading model")
            self.model_obj = None

    def _fetch_current_vector(self) -> Tuple[Optional[List[float]], Optional[dict]]:
        unl = fetch_unlabeled(limit=1)
        if not unl:
            return None, None
        last = unl[0]
        feat = (
            last.data.get("features")
            if last.data and isinstance(last.data, dict)
            else None
        )
        if not feat:
            return None, None
        # ensure all keys present
        vec = [feat.get(k) if feat.get(k) is not None else 0.0 for k in FEATURE_ORDER]
        return vec, feat

    def inference_job(self):
        try:
            if self.check_and_label_user_override():
                return
        except Exception:
            logger.exception("Error during override check; continuing with inference")

        # reload model each run to pick up new full model
        self.load_model()
        if not self.model_obj:
            return

        obj = self.model_obj
        model = obj.get("model")
        scaler = obj.get("scaler")

        try:
            Xvec, featdict = self._fetch_current_vector()
            if Xvec is None:
                logger.info("No current features for inference")
                return

            X = np.array([Xvec], dtype=float)

            # if model is a pipeline predict directly
            if hasattr(model, "predict") and scaler is None:
                pred = model.predict(X)[0]
            else:
                # expect scaler + model stored separately
                if scaler is None or model is None:
                    logger.warning("Model object incomplete; skipping inference")
                    return
                Xs = scaler.transform(X)
                pred = model.predict(Xs)[0]
        except Exception:
            logger.exception("Inference failed")
            return

        logger.info(
            "Prediction raw=%s last_pred=%s last_ts=%s",
            pred,
            self.last_pred_value,
            self.last_pred_ts,
        )
        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 24.0))
        threshold = float(self.opts.get("min_change_threshold", 0.3))
        stable_seconds = float(self.opts.get("stable_seconds", 600))
        current_sp = featdict.get("current_setpoint", None) if featdict else None
        if current_sp is None:
            logger.warning("Current setpoint unknown, skipping action")
            return

        if pred < min_sp or pred > max_sp:
            logger.warning("Predicted setpoint outside plausible range: %s", pred)
            return

        pred = max(min(pred, max_sp), min_sp)
        now = datetime.utcnow()

        if self.last_pred_value is not None and self.last_pred_ts:
            if abs(self.last_pred_value - pred) < threshold:
                if (now - self.last_pred_ts).total_seconds() < stable_seconds:
                    logger.info("Change not yet stable; skipping apply")
                    return

        try:
            unl = fetch_unlabeled(limit=1)
            if unl:
                latest = unl[0]
                sample_ts = getattr(latest, "timestamp", None)
                age = (now - sample_ts).total_seconds() if sample_ts else float("inf")
                age_thresh = float(self.opts.get("sample_interval_seconds", 300))
                if age <= age_thresh:
                    #update_sample_prediction(
                    #    latest.id, predicted_setpoint=pred, prediction_error=None
                    #)
                    #sid = latest.id
                else:
                    #features = self.collector.get_features(ts=now)
                    #sid = insert_sample({"features": features})
                    #update_sample_prediction(
                    #    sid, predicted_setpoint=pred, prediction_error=None
                    #)
            else:
                #features = self.collector.get_features(ts=now)
                #sid = insert_sample({"features": features})
                #update_sample_prediction(
                #    sid, predicted_setpoint=pred, prediction_error=None
                #)
        except Exception:
            logger.exception("Failed to persist predicted_setpoint; continuing")

        if abs(pred - current_sp) < threshold:
            logger.info(
                "Predicted change below threshold (%.2f < %.2f)",
                abs(pred - current_sp),
                threshold,
            )
            return
            
        # apply setpoint
        try:
            self.ha.set_setpoint(pred)
            self.last_pred_ts = now
            self.last_pred_value = pred
            logger.info("Applied predicted setpoint %.1f (was %.1f)", pred, current_sp)
        except Exception:
            logger.exception("Failed to apply setpoint via HAClient")
