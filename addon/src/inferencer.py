import os
import logging
import joblib
import numpy as np
import datetime

from db import fetch_unlabeled, update_label
from feature_extractor import FEATURE_ORDER
from ha_client import HAClient
from utils import round_half

logger = logging.getLogger(__name__)

class Inferencer:
    def __init__(self, ha_client: HAClient, opts: dict):
        if not opts.get("model_path_full") or not opts.get("model_path_partial"):
            raise RuntimeError("model_path_full and model_path_partial must be provided in opts.")
        self.ha = ha_client
        self.opts = opts
        self.model_obj = None
        self.load_model()

    def check_and_label_user_override(self):
        rows = fetch_unlabeled(limit=1)
        if not rows:
            return False
        row = rows[0]
    
        interval = int(self.opts.get("sample_interval_seconds", 300))
        sample_ts = getattr(row, "timestamp", None)
        age = (datetime.datetime.utcnow() - sample_ts).total_seconds() if sample_ts else None
    
        try:
            current_sp, _ = self.ha.get_setpoint()
        except Exception:
            return False
    
        def safe_round(v):
            try:
                return round(float(v), 1)
            except Exception:
                return None
    
        rounded_current = safe_round(current_sp)
        predicted = getattr(row, "predicted_setpoint", None)
    
        if predicted is not None:
            rounded_pred = safe_round(predicted)
            if rounded_current is not None and rounded_pred is not None:
                if rounded_current == rounded_pred:
                    return False
                else:
                    update_label(row.id, float(current_sp), user_override=True)
                    return True
    
        # fallback: compare sample snapshot if recent
        sample_sp = None
        if row.data and isinstance(row.data, dict):
            feat = row.data.get("features")
            if isinstance(feat, dict):
                sample_sp = feat.get("current_setpoint")
            else:
                sensors = row.data.get("sensors") if isinstance(row.data.get("sensors"), dict) else {}
                sample_sp = sensors.get("current_setpoint")
    
        rounded_sample = safe_round(sample_sp)
        if age is not None and age <= interval * 1.5:
            if rounded_sample is not None and rounded_current is not None and rounded_sample != rounded_current:
                update_label(row.id, float(current_sp), user_override=True)
                return True
            return False
    
        return False

    def load_model(self):
        try:
            if os.path.exists(self.opts.get("model_path_full")):
                self.model_obj = joblib.load(self.opts.get("model_path_full"))
                meta = self.model_obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logger.warning("Full model feature_order mismatch; ignoring full model")
                    self.model_obj = None
                else:
                    logger.info("Loaded full model")
            elif os.path.exists(self.opts.get("model_path_partial")):
                self.model_obj = joblib.load(self.opts.get("model_path_partial"))
                meta = self.model_obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logger.warning("Partial model feature_order mismatch; ignoring partial model")
                    self.model_obj = None
                else:
                    logger.info("Loaded partial model")
            else:
                self.model_obj = None
                logger.info("No model present, inference skipped")
        except Exception:
            logger.exception("Error loading model")
            self.model_obj = None

    def get_current_features(self):
        unl = fetch_unlabeled(limit=1)
        if unl:
            last = unl[0]
            feat = last.data.get("features") if last.data else None
            return [feat[k] for k in FEATURE_ORDER] if feat else None, feat
        return None, None

    def inference_job(self):
        try:
            if self.check_and_label_user_override():
                return
        except Exception:
            logger.exception("Error during override check; continuing with inference")
            
        self.load_model()
        if not self.model_obj:
            return
        obj = self.model_obj
        model = obj.get("model")
        scaler = obj.get("scaler") if isinstance(obj.get("model"), (type,)) else None

        try:
            Xvec, featdict = self.get_current_features()
            if Xvec is None:
                logger.info("No current features for inference")
                return
            # Full model case: stored pipeline in "model"
            if hasattr(model, "predict"):
                X = np.array([Xvec])
                pred = model.predict(X)[0]
            else:
                scaler = obj.get("scaler")
                model = obj.get("model")
                if scaler is None or model is None:
                    logger.warning("Model object incomplete")
                    return
                Xs = scaler.transform([Xvec])
                pred = model.predict(Xs)[0]
        except Exception:
            logger.exception("Inference failed")
            return

        min_sp = self.opts.get("min_setpoint", 15.0)
        max_sp = self.opts.get("max_setpoint", 24.0)
        threshold = self.opts.get("min_change_threshold", 0.3)
        current_sp = featdict.get("current_setpoint", None) if featdict else None
        if current_sp is None:
            logger.warning("Current setpoint unknown, skipping action")
            return
        pred = max(min(pred, max_sp), min_sp)
        if abs(pred - current_sp) < threshold:
            logger.info("Predicted change below threshold (%.2f < %.2f)", abs(pred - current_sp), threshold)
            return
        # persist prediction: update recent unlabeled sample else insert new
        now = datetime.datetime.utcnow()
        age_thresh = float(self.opts.get("sample_interval_seconds", 300)) * 1.5
        
        # snapshot current states
        snapshot = {}
        try:
            cs, ct = self.ha.get_setpoint()
            snapshot["current_setpoint"] = cs
            snapshot["current_temp"] = ct
        except Exception:
            snapshot = {}
        
        for key, ent in (self.opts.get("sensors") or {}).items():
            try:
                st = self.ha.get_state(ent)
                snapshot[key] = float(st.get("state")) if st and "state" in st else None
            except Exception:
                snapshot[key] = None
        
        try:
            unl = fetch_unlabeled(limit=1)
            if unl:
                latest = unl[0]
                sample_ts = getattr(latest, "timestamp", None)
                age = (now - sample_ts).total_seconds() if sample_ts else float("inf")
                if age <= age_thresh:
                    update_sample_prediction(latest.id, predicted_setpoint=pred, prediction_error=None)
                    sid = latest.id
                else:
                    fe = FeatureExtractor()
                    features_for_pred = fe.features_from_raw(snapshot, timestamp=now)
                    sid = insert_sample({"timestamp": now.isoformat(), "sensors": snapshot, "features": features_for_pred})
                    update_sample_prediction(sid, predicted_setpoint=pred, prediction_error=None)
            else:
                fe = FeatureExtractor()
                features_for_pred = fe.features_from_raw(snapshot, timestamp=now)
                sid = insert_sample({"timestamp": now.isoformat(), "sensors": snapshot, "features": features_for_pred})
                update_sample_prediction(sid, predicted_setpoint=pred, prediction_error=None)
        except Exception:
            logger.exception("Failed to persist predicted_setpoint; continuing")

        self.ha.set_setpoint(pred)
        logger.info("Applied predicted setpoint %.1f (was %.1f)", pred, current_sp)
