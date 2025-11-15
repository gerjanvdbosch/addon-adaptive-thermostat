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
from sklearn.pipeline import Pipeline

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
        now = datetime.now()
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
            "Detected user override: last %.1f, current %.1f",
            last_sample_rounded,
            current_rounded,
        )
        return True

    def load_models(self):
        """
        Load models and normalize into {'model','scaler','meta'}.
        If meta lacks 'mae' assign inf to avoid using unevaluated partials before full.
        """
        try:
            self.models = {}
            for name, path in (
                ("full", self.opts.get("model_path_full")),
                ("partial", self.opts.get("model_path_partial")),
            ):
                if not path:
                    continue
                if not os.path.exists(path):
                    logger.debug("Model path %s for %s does not exist", path, name)
                    continue
                try:
                    obj = joblib.load(path)
                except Exception:
                    logger.exception(
                        "Failed to joblib.load %s model from %s", name, path
                    )
                    continue

                if not isinstance(obj, dict):
                    logger.warning("Loaded %s model is not a dict; ignoring", name)
                    continue

                meta = obj.get("meta", {}) or {}
                if meta.get("feature_order") != FEATURE_ORDER:
                    logger.warning(
                        "%s model feature_order mismatch; ignoring %s", name, path
                    )
                    continue

                meta.setdefault("mae", float("inf"))

                model = obj.get("model")
                scaler = obj.get("scaler") if "scaler" in obj else None

                if isinstance(model, Pipeline) and scaler is not None:
                    logger.debug(
                        "%s: model is a Pipeline and an external scaler was present; ignoring external scaler",
                        name,
                    )
                    scaler = None

                # Sanity checks
                try:
                    feat_len = len(FEATURE_ORDER)
                    if scaler is not None and hasattr(scaler, "mean_"):
                        if len(getattr(scaler, "mean_")) != feat_len:
                            logger.warning(
                                "%s scaler mean length (%d) != FEATURE_ORDER length (%d); ignoring model",
                                name,
                                len(getattr(scaler, "mean_")),
                                feat_len,
                            )
                            continue
                except Exception:
                    logger.exception("Sanity check failed for %s model; skipping", name)
                    continue

                # attach normalized object
                self.models[name] = {"model": model, "scaler": scaler, "meta": meta}
                logger.info(
                    "Loaded %s model from %s (MAE=%s)", name, path, meta.get("mae")
                )
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
        vec = [feat.get(k) if feat.get(k) is not None else 0.0 for k in FEATURE_ORDER]
        return vec, feat

    def _predict_with_model(
        self, name: str, obj: dict, X: np.ndarray
    ) -> Optional[float]:
        """
        Predict with strong diagnostics:
        - logs raw features, shapes
        - logs scaler.mean_, scaled features (first 10 values) if scaler present
        - logs model.coef_ and intercept if available
        """
        try:
            model = obj.get("model")
            scaler = obj.get("scaler")
            meta = obj.get("meta", {}) or {}

            if model is None:
                logger.debug("Model %s has no estimator; skipping", name)
                return None

            # Ensure X is numpy 2D
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)

            logger.debug(
                "Model %s predict called. X.shape=%s FEATURE_ORDER_len=%d",
                name,
                X.shape,
                len(FEATURE_ORDER),
            )

            # Log raw feature sample values (first 20 chars)
            try:
                logger.debug("Model %s raw X sample: %s", name, X[0].tolist())
            except Exception:
                logger.debug("Model %s unable to list raw X", name)

            # If model is a pipeline -> pipeline handles scaling internally
            if isinstance(model, Pipeline):
                try:
                    preds = model.predict(X)
                    p = float(preds[0])
                except Exception:
                    logger.exception("Pipeline predict failed for model %s", name)
                    return None
            else:
                # Bare estimator path
                # Log model coefficients if present
                try:
                    if hasattr(model, "coef_"):
                        logger.debug(
                            "Model %s coef sample (first10): %s",
                            name,
                            np.array(model.coef_).tolist()[:10],
                        )
                    if hasattr(model, "intercept_"):
                        logger.debug(
                            "Model %s intercept: %s", name, getattr(model, "intercept_")
                        )
                except Exception:
                    logger.debug("Could not read model coef/intercept for %s", name)

                if scaler is not None:
                    # log scaler.mean_ if available
                    if hasattr(scaler, "mean_"):
                        try:
                            logger.debug(
                                "Model %s scaler.mean_ (first10): %s",
                                name,
                                np.array(scaler.mean_).tolist()[:10],
                            )
                        except Exception:
                            logger.debug(
                                "Model %s scaler.mean_ present but failed to list", name
                            )

                    # Attempt transform with fallback
                    try:
                        Xs = scaler.transform(X)
                        logger.debug("Raw X sample: %s", X[0].tolist())
                        if hasattr(scaler, "mean_"):
                            logger.debug(
                                "scaler.mean_ (first10): %s", scaler.mean_[:10].tolist()
                            )
                        logger.debug("Scaled X sample (first20): %s", Xs[0].tolist())
                        logger.debug(
                            "model.intercept: %s coef (first10): %s",
                            model.intercept_,
                            model.coef_[:10].tolist(),
                        )
                        logger.debug(
                            "linear_term: %s",
                            float(Xs.dot(model.coef_.reshape(-1, 1))[0]),
                        )
                        logger.debug(
                            "Model %s scaled X sample (first20): %s",
                            name,
                            Xs[0].tolist()[:20],
                        )
                    except Exception:
                        logger.exception(
                            "Scaler.transform failed for model %s; falling back to raw features",
                            name,
                        )
                        Xs = X

                    try:
                        preds = model.predict(Xs)
                        p = float(preds[0])
                    except Exception:
                        logger.exception(
                            "Estimator predict failed for model %s after scaling", name
                        )
                        return None
                else:
                    # No scaler present
                    try:
                        preds = model.predict(X)
                        p = float(preds[0])
                    except Exception:
                        logger.exception(
                            "Estimator predict failed for model %s (no scaler)", name
                        )
                        return None

            logger.info("Model %s predicted %.4f (MAE=%s)", name, p, meta.get("mae"))

            # Additional debug: if prediction out of expected human setpoint range, log details
            min_sp = float(self.opts.get("min_setpoint", 15.0))
            max_sp = float(self.opts.get("max_setpoint", 24.0))
            if p < min_sp or p > max_sp:
                logger.warning(
                    "Model %s produced out-of-range prediction %.4f (expected %.1f-%.1f). Dumping diagnostics.",
                    name,
                    p,
                    min_sp,
                    max_sp,
                )
                # Log full coefficient vector if present
                try:
                    if hasattr(model, "coef_"):
                        logger.warning(
                            "Model %s full coef: %s",
                            name,
                            np.array(model.coef_).tolist(),
                        )
                except Exception:
                    logger.warning("Unable to log full coefs for model %s", name)

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
            logger.info("No models available for inference")
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
        now = datetime.now()

        sorted_models = sorted(
            self.models.items(),
            key=lambda kv: kv[1].get("meta", {}).get("mae", float("inf")),
        )

        pred = None
        model_used = None

        for name, obj in sorted_models:
            p = self._predict_with_model(name, obj, X)
            if p is None:
                logger.debug("Model %s returned no prediction; trying next", name)
                continue

            # enforce plausible range
            if p < min_sp or p > max_sp:
                logger.warning(
                    "Predicted change outside plausible range from model %s: %s",
                    name,
                    p,
                )
                continue

            p = max(min(p, max_sp), min_sp)
            rounded_p = safe_round(p)

            if (
                self.last_eval_value is None
                or safe_round(self.last_eval_value) != rounded_p
            ):
                self.last_eval_value = p
                self.last_eval_ts = now
                logger.info(
                    "Starting stability timer for predicted value %.2f (model=%s)",
                    p,
                    name,
                )
                continue

            if (now - self.last_eval_ts).total_seconds() < stable_seconds:
                logger.info(
                    "Predicted value from model %s not yet stable; skipping", name
                )
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
