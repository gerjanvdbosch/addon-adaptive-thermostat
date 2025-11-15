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


class Inferencer2:
    """
    Inferencer die verwacht dat het model een delta voorspelt:
      predicted_setpoint = current_setpoint + model.predict(X)[0]

    Verwachte model payload op disk: joblib dump van dict {"model": <estimator>, "meta": {...}}
    Meta moet "feature_order" en "target" bevatten; voor deze implementatie target == "delta"
    """

    def __init__(self, ha_client: HAClient, collector: Collector, opts: dict):
        self.ha = ha_client
        self.collector = collector
        self.opts = opts or {}
        # self.model_path = self.opts.get("model_path_full")
        self.model_path = "/config/models/full_model2.joblib"
        self.model_payload = None  # loaded payload dict {"model","meta"}
        self.last_pred_ts: Optional[datetime] = None
        self.last_pred_value: Optional[float] = None
        self.last_pred_model: Optional[str] = None
        self.last_eval_value: Optional[float] = None
        self.last_eval_ts: Optional[datetime] = None

        self.load_model()

    def load_model(self):
        """Laad model van schijf en valideer feature_order / target."""
        self.model_payload = None
        path = self.model_path
        if not path or not os.path.exists(path):
            logger.debug("No model path configured or file missing: %s", path)
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
        # require metadata target == "delta"
        if meta.get("target") and meta.get("target") != "delta":
            logger.warning("Model target is not 'delta'; ignoring model at %s", path)
            return
        if "model" not in payload or payload["model"] is None:
            logger.warning("Model payload missing 'model'; ignoring %s", path)
            return
        self.model_payload = payload
        logger.info("Loaded model from %s (mae=%s)", path, meta.get("mae"))

    def check_and_label_user_override(self) -> bool:
        """Detecteer en label een echte gebruikeroverride; return True als gelabeld."""
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
            row = rows[0]

            current_sp, *rest = self.ha.get_setpoint()
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
                "Detected user override and inserted labeled sample: last %.1f -> current %.1f",
                last_sample_rounded,
                current_rounded,
            )
            return True
        except Exception:
            logger.exception("Error detecting/labeling user override")
            return False

    def _fetch_current_vector(self) -> Tuple[Optional[List[float]], Optional[dict]]:
        """Haal de laatste unlabelled sample features op en maak vector volgens FEATURE_ORDER."""
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
                if v is None:
                    v = 0.0
                else:
                    try:
                        v = float(v)
                    except Exception:
                        logger.warning(
                            "Feature %s value not numeric: %r; coercing to 0.0", k, v
                        )
                        v = 0.0
                vec.append(v)
            return vec, feat
        except Exception:
            logger.exception("Failed fetching current vector")
            return None, None

    def inference_job(self):
        """Hoofdlogica: label overrides, laad model, voorspel delta, reconstrueer setpoint en toepassen."""
        try:
            if self.check_and_label_user_override():
                return
        except Exception:
            logger.exception("Error during override check; continuing")

        # refresh model each run to pick up new saved model
        self.load_model()
        if self.model_payload is None:
            logger.debug("No model available for inference")
            return

        Xvec, featdict = self._fetch_current_vector()
        if Xvec is None:
            logger.debug("No current features for inference")
            return
        X = np.array([Xvec], dtype=float)

        min_sp = float(self.opts.get("min_setpoint", 15.0))
        max_sp = float(self.opts.get("max_setpoint", 24.0))
        threshold = float(self.opts.get("min_change_threshold", 0.3))
        stable_seconds = float(self.opts.get("stable_seconds", 600))
        current_sp = featdict.get("current_setpoint") if featdict else None
        if current_sp is None:
            logger.warning("Current setpoint unknown; skipping inference")
            return
        now = datetime.now()

        # --- debug logging: inspect features, vector en model type before predict
        logger.debug(
            "DEBUG: featdict keys = %s", sorted(featdict.keys()) if featdict else None
        )
        logger.debug(
            "DEBUG: featdict sample = %s",
            (
                {k: featdict.get(k) for k in list(featdict.keys())[:10]}
                if featdict
                else None
            ),
        )
        logger.debug("DEBUG: Xvec = %s", Xvec)
        logger.debug(
            "DEBUG: X shape/dtype = %s %s",
            getattr(X, "shape", None),
            getattr(X, "dtype", None),
        )

        # inspect model
        model = self.model_payload.get("model")
        logger.debug("DEBUG: model type = %s", type(model))
        try:
            # model must predict delta
            pred_raw = model.predict(X)
            logger.debug("DEBUG: model.predict returned = %s", pred_raw)
            pred_delta = (
                float(pred_raw[0]) if hasattr(pred_raw, "__len__") else float(pred_raw)
            )
            p = float(current_sp) + pred_delta
            logger.debug(
                "DEBUG: predicted_delta = %s, reconstructed_setpoint = %s",
                pred_delta,
                p,
            )
        except Exception:
            logger.exception("Prediction failed during debug predict")
            return

        if p is None or not np.isfinite(p):
            logger.debug("No valid prediction")
            return

        if p < min_sp or p > max_sp:
            logger.warning("Predicted value outside plausible range: %.3f", p)
            return
        p = float(max(min(p, max_sp), min_sp))
        rounded_p = safe_round(p)

        # stability timer logic
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
                "Prediction (%.2f) not yet stable; waiting (%.0fs remaining)",
                p,
                stable_seconds - (now - self.last_eval_ts).total_seconds(),
            )
            return

        if abs(p - current_sp) < threshold:
            logger.info(
                "Prediction (%.2f), change %.3f below threshold %.3f; skipping",
                p,
                abs(p - current_sp),
                threshold,
            )
            return

        # cooldown
        cooldown = float(self.opts.get("cooldown_seconds", 3600))
        if self.last_pred_ts and (now - self.last_pred_ts).total_seconds() < cooldown:
            logger.info("Cooldown active; skipping predicted setpoint %.2f", p)
            return

        # persist prediction (store absolute predicted_setpoint)
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
            logger.exception("Failed to persist predicted_setpoint; continuing")

        # apply setpoint
        try:
            self.ha.set_setpoint(p)
            self.last_pred_ts = now
            self.last_pred_value = p
            self.last_pred_model = self.model_path
            logger.info("Applied predicted setpoint %.2f (was %.2f)", p, current_sp)
        except Exception:
            logger.exception("Failed to apply setpoint via HAClient")
