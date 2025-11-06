import os
import logging
import joblib
import numpy as np

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
        interval = int(self.opts.get("sample_interval_seconds"))
        sample_ts = getattr(row, "timestamp", None)
        if sample_ts:
            age = (datetime.datetime.utcnow() - sample_ts).total_seconds()
            if age > interval:
                logger.info("Latest unlabeled sample is older than interval (%ds > %ds); skipping override check", age, interval)
                return False
        feat = row.data.get("features") if row.data else None
        sample_sp = feat.get("current_setpoint") if feat else None
        if sample_sp is None:
            logger.warning("Latest unlabeled sample %s has no current_setpoint", getattr(row, "id"))
            return False
        current_sp, current_temp = self.ha.get_setpoint()
        rounded_sample = round(round_half(sample_sp), 1)
        rounded_current = round(round_half(current_sp), 1)
        
        if rounded_sample != rounded_current:
            update_label(row.id, float(current_sp), user_override=True)
            logger.info("Detected external setpoint change; updated label for sample_id=%s as user override -> %.2f", row.id, current_sp)
            return True
        return False
    
    def load_model(self):
        try:
            if os.path.exists(self.opts.get("model_path_full")):
                self.model_obj = joblib.load(self.opts.get("model_path_full"))
                meta = self.model_obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logging.warning("Full model feature_order mismatch; ignoring full model")
                    self.model_obj = None
                else:
                    logging.info("Loaded full model")
            elif os.path.exists(self.opts.get("model_path_partial")):
                self.model_obj = joblib.load(self.opts.get("model_path_partial"))
                meta = self.model_obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logging.warning("Partial model feature_order mismatch; ignoring partial model")
                    self.model_obj = None
                else:
                    logging.info("Loaded partial model")
            else:
                self.model_obj = None
                logging.info("No model present, inference skipped")
        except Exception:
            logging.exception("Error loading model")
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
        self.ha.set_setpoint(pred)
        logger.info("Applied predicted setpoint %.1f to %s (was %.1f)", pred, climate, current_sp)
