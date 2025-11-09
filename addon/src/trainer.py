import os
import logging
import joblib
import numpy as np
import datetime

from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error

from db import fetch_training_data, insert_metric, update_sample_prediction
from collector import FEATURE_ORDER

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, ha_client, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}
        if not self.opts.get("model_path_full") or not self.opts.get("model_path_partial"):
            raise RuntimeError("model_path_full and model_path_partial must be provided in opts.")
        self.partial = None
        self.scaler = None

        # try load partial model if present and compatible
        partial_path = self.opts.get("model_path_partial")
        if partial_path and os.path.exists(partial_path):
            try:
                obj = joblib.load(partial_path)
                self.partial = obj.get("model")
                self.scaler = obj.get("scaler")
                meta = obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logger.warning("Partial model feature_order mismatch; ignoring partial model")
                    self.partial = None
                    self.scaler = None
                else:
                    logger.info("Loaded partial model from %s", partial_path)
            except Exception:
                logger.exception("Failed loading partial model; starting fresh")
                self.partial = None
                self.scaler = None

    def partial_fit_job(self):
        rows = fetch_training_data(days=self.opts.get("buffer_days", 30))
        # only keep labeled user overrides
        rows = [r for r in rows if (r.label_setpoint is not None and getattr(r, "user_override", False))]
        if not rows:
            logger.info("No training rows available for partial_fit")
            return

        X = []
        y = []
        for r in rows:
            feat = (r.data.get("features") if r.data and isinstance(r.data, dict) else None)
            if not feat:
                continue
            try:
                vec = [feat.get(k) if feat.get(k) is not None else 0.0 for k in FEATURE_ORDER]
                X.append(vec)
                y.append(float(r.label_setpoint))
            except Exception:
                logger.exception("Skipping corrupt row %s in partial_fit", getattr(r, "id", None))

        if not X:
            logger.info("No labeled rows after filtering for partial_fit")
            return

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        # fit scaler on batch (safer than partial fitting scaler)
        if self.scaler is None:
            self.scaler = StandardScaler()
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        if self.partial is None:
            self.partial = SGDRegressor(
                max_iter=1,
                tol=None,
                learning_rate="constant",
                eta0=float(self.opts.get("partial_eta0", 0.01)),
                penalty="l2",
                alpha=float(self.opts.get("partial_alpha", 0.0001)),
                warm_start=True
            )

        try:
            self.partial.partial_fit(Xs, y)
        except Exception:
            logger.exception("partial_fit failed")
            return

        # persist partial model + scaler + meta
        try:
            joblib.dump({
                "model": self.partial,
                "scaler": self.scaler,
                "meta": {
                    "feature_order": FEATURE_ORDER,
                    "trained_at": datetime.datetime.datetime.utcnow().isoformat() if hasattr(datetime, "datetime") else datetime.datetime.utcnow().isoformat()
                }
            }, self.opts.get("model_path_partial"))
            logger.info("Partial model updated with %d samples", len(X))
        except Exception:
            logger.exception("Failed saving partial model")

    def full_retrain_job(self):
        rows = fetch_training_data(days=self.opts.get("buffer_days", 30))
        rows = [r for r in rows if (r.label_setpoint is not None and getattr(r, "user_override", False))]
        if not rows:
            logger.info("No training rows available for full retrain")
            return

        X = []
        y = []
        used_rows = []
        for r in rows:
            feat = (r.data.get("features") if r.data and isinstance(r.data, dict) else None)
            if not feat:
                continue
            try:
                vec = [feat.get(k) if feat.get(k) is not None else 0.0 for k in FEATURE_ORDER]
                X.append(vec)
                y.append(float(r.label_setpoint))
                used_rows.append(r)
            except Exception:
                logger.exception("Skipping corrupt row %s in full_retrain", getattr(r, "id", None))

        if not X:
            logger.info("No labeled rows after filtering for full retrain")
            return

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        n_splits = min(3, max(2, len(X) // 10))
        tss = TimeSeriesSplit(n_splits=n_splits)
        gs = GridSearchCV(pipe, param_grid, cv=tss, scoring="neg_mean_absolute_error", n_jobs=1)

        try:
            gs.fit(X, y)
        except Exception:
            logger.exception("GridSearchCV failed; attempting single fit of default pipeline")
            pipe.fit(X, y)
            best = pipe
        else:
            best = gs.best_estimator_

        # OOF MAE estimate
        try:
            oof_preds = np.zeros_like(y, dtype=float)
            tss_oof = TimeSeriesSplit(n_splits=n_splits)
            for train_idx, test_idx in tss_oof.split(X):
                clone = gs.best_estimator_ if hasattr(gs, "best_estimator_") else best
                clone.fit(X[train_idx], y[train_idx])
                oof_preds[test_idx] = clone.predict(X[test_idx])
            mae = float(mean_absolute_error(y, oof_preds))
        except Exception:
            try:
                preds_all = best.predict(X)
                mae = float(mean_absolute_error(y, preds_all))
            except Exception:
                mae = None

        metadata = {
            "feature_order": FEATURE_ORDER,
            "best_params": getattr(gs, "best_params_", None),
            "trained_at": datetime.datetime.datetime.utcnow().isoformat() if hasattr(datetime, "datetime") else datetime.datetime.utcnow().isoformat(),
            "mae": mae
        }

        try:
            joblib.dump({"model": best, "meta": metadata}, self.opts.get("model_path_full"))
            logger.info("Full model trained on %d labeled user samples (OOF MAE=%s) and saved", len(X), mae)
        except Exception:
            logger.exception("Failed saving full model")

        # persist metric record
        try:
            insert_metric(model_type="full", mae=mae, n_samples=len(X), meta=metadata)
        except Exception:
            logger.exception("Failed to insert metric record")

        # update per-sample predictions
        try:
            preds_all = best.predict(X)
            for i, row in enumerate(used_rows):
                try:
                    pred = float(preds_all[i])
                    err = abs(pred - float(y[i])) if y is not None else None
                    update_sample_prediction(row.id, predicted_setpoint=pred, prediction_error=err)
                except Exception:
                    logger.exception("Failed updating sample prediction for sample %s", getattr(row, "id", None))
        except Exception:
            logger.exception("Failed to compute/update per-sample predictions")
