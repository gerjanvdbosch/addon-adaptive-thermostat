import os
import logging
import joblib
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List

from db import fetch, fetch_unlabeled, update_sample_prediction, insert_sample
from collector import FEATURE_ORDER, Collector
from ha_client import HAClient
from utils import safe_round

logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}
        self.models = {}  # {'full': {...}, 'partial': {...}}
        self.last_pred_ts: Optional[datetime] = None
        self.last_pred_value: Optional[float] = None
        self.last_pred_model: Optional[str] = None
        self.last_eval_value: Optional[float] = None
        self.last_eval_ts: Optional[datetime] = None
        self.load_models()

    def check_and_label_user_override(self) -> bool:
        now = datetime.utcnow()
        interval = float(self.opts.get("sample_interval_seconds", 300))
        if (
            self.last_pred_ts is not None
            and (now - self.last_pred_ts).total_seconds() < interval
        ):
            return False

        rows = fetch(limit=1)
        if not rows:
            return False
        row = rows[0]

        current_sp, _ = self.ha.get_setpoint()
        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 24.0))
        if not (min_sp <= current_sp <= max_sp):
            logger.warning("Setpoint outside plausible range: %s", current_sp)
            return False

        last_sample_sp = row.data.get("current_setpoint") if row.data else None
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
            logger.info(
                "Current setpoint matches last predicted value; not user override"
            )
            return False

        features = self.collector.get_features(ts=now)
        insert_sample(features, label_setpoint=current_sp, user_override=True)
        logger.info(
            "Detected user override: last %.1f, current %.1f",
            last_sample_rounded,
            current_rounded,
        )
        return True

    def load_models(self):
        try:
            for name, path in (
                ("full", self.opts.get("model_path_full")),
                ("partial", self.opts.get("model_path_partial")),
            ):
                if path and os.path.exists(path):
                    obj = joblib.load(path)
                    meta = obj.get("meta", {})
                    if meta.get("feature_order") != FEATURE_ORDER:
                        logger.warning(
                            "%s model feature_order mismatch; ignoring", name
                        )
                        continue
                    self.models[name] = obj
                    logger.debug("Loaded %s model from %s", name, path)
        except Exception:
            logger.exception("Error loading models")

    def _fetch_current_vector(self) -> Tuple[Optional[List[float]], Optional[dict]]:
        unl = fetch_unlabeled(limit=1)
        if not unl:
            return None, None
        last = unl[0]
        feat = last.data if last.data and isinstance(last.data, dict) else None
        if not feat:
            return None, None
        # ensure all keys present
        vec = [feat.get(k) if feat.get(k) is not None else 0.0 for k in FEATURE_ORDER]
        return vec, feat

    def _predict_with_model(self, name, obj, X):
        try:
            model = obj.get("model")
            scaler = obj.get("scaler")
            if model is None:
                return None
            if scaler is not None:
                Xs = scaler.transform(X)
                p = model.predict(Xs)[0]
            else:
                p = model.predict(X)[0]
            mae = obj.get("meta", {}).get("mae")
            logger.info("Model %s predicted %.2f (MAE=%s)", name, p, mae)
            return p
        except Exception:
            logger.exception("Prediction failed for model %s", name)
            return None

    def inference_job(self):
        try:
            if self.check_and_label_user_override():
                return
        except Exception:
            logger.exception("Error during override check; continuing")

        self.load_models()
        if not self.models:
            return

        Xvec, featdict = self._fetch_current_vector()
        if Xvec is None:
            logger.info("No current features for inference")
            return
        X = np.array([Xvec], dtype=float)

        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 24.0))
        threshold = float(self.opts.get("min_change_threshold", 0.3))
        stable_seconds = float(self.opts.get("stable_seconds", 600))
        current_sp = featdict.get("current_setpoint") if featdict else None
        if current_sp is None:
            logger.warning("Current setpoint unknown, skipping action")
            return
        now = datetime.utcnow()

        # sort models by MAE ascending
        sorted_models = sorted(
            self.models.items(), key=lambda kv: kv[1].get("meta", {}).get("mae", np.inf)
        )

        pred = None
        model_used = None

        for name, obj in sorted_models:
            p = self._predict_with_model(name, obj, X)
            if p is None or p < min_sp or p > max_sp:
                logger.warning("Predicted change outside plausible range: %s", p)
                continue
            p = max(min(p, max_sp), min_sp)
            rounded_p = safe_round(p)

            if (
                self.last_eval_value is None
                or safe_round(self.last_eval_value) != rounded_p
            ):
                self.last_eval_value = p
                self.last_eval_ts = now
                logger.info("Starting stability timer for predicted value %.2f", p)
                continue

            if (now - self.last_eval_ts).total_seconds() < stable_seconds:
                logger.info("Predicted change from %s not yet stable; skipping", name)
                continue

            if abs(p - current_sp) < threshold:
                logger.info(
                    "Predicted change from %s below threshold (%.2f < %.2f); skipping",
                    name,
                    abs(p - current_sp),
                    threshold,
                )
                continue

            pred = p
            model_used = name
            break

        if pred is None:
            logger.warning("No valid model prediction found")
            return

        cooldown = float(self.opts.get("cooldown_seconds", 3600))
        if self.last_pred_ts and (now - self.last_pred_ts).total_seconds() < cooldown:
            logger.info("Cooldown active; skipping predicted setpoint %.1f", pred)
            return

        try:
            unl = fetch_unlabeled(limit=1)
            if unl:
                latest = unl[0]
                sample_ts = getattr(latest, "timestamp", None)
                age = (now - sample_ts).total_seconds() if sample_ts else float("inf")
                age_thresh = float(self.opts.get("sample_interval_seconds", 300))
                if age <= age_thresh:
                    update_sample_prediction(
                        latest.id, predicted_setpoint=pred, prediction_error=None
                    )
                    sid = latest.id
                else:
                    features = self.collector.get_features(ts=now)
                    sid = insert_sample(features)
                    update_sample_prediction(
                        sid, predicted_setpoint=pred, prediction_error=None
                    )
            else:
                features = self.collector.get_features(ts=now)
                sid = insert_sample(features)
                update_sample_prediction(
                    sid, predicted_setpoint=pred, prediction_error=None
                )
        except Exception:
            logger.exception("Failed to persist predicted_setpoint; continuing")

        # apply setpoint
        try:
            self.ha.set_setpoint(pred)
            self.last_pred_ts = now
            self.last_pred_value = pred
            self.last_pred_model = model_used

            logger.info(
                "Applied predicted setpoint %.1f (was %.1f) using model %s",
                pred,
                current_sp,
                model_used,
            )
        except Exception:
            logger.exception("Failed to apply setpoint via HAClient")
