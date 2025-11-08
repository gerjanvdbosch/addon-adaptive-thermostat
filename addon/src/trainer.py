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
from feature_extractor import FEATURE_ORDER

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, ha_client, opts):
        if not opts.get("model_path_full") or not opts.get("model_path_partial"):
            raise RuntimeError("model_path_full and model_path_partial must be provided in opts.")
        self.ha = ha_client
        self.opts = opts
        self.partial = None
        self.scaler = None
        if os.path.exists(self.opts.get("model_path_partial")):
            try:
                obj = joblib.load(self.opts.get("model_path_partial"))
                self.partial = obj.get("model")
                self.scaler = obj.get("scaler")
                meta = obj.get("meta", {})
                if meta.get("feature_order") != FEATURE_ORDER:
                    logger.warning("Partial model feature_order mismatch; ignoring partial model")
                    self.partial = None
                    self.scaler = None
            except Exception:
                logger.exception("Failed loading partial model; starting fresh")
                self.partial = None
                self.scaler = None

    def partial_fit_job(self):
        rows = fetch_training_data(days=self.opts.get("buffer_days", 30))
        rows = [r for r in rows if (r.label_setpoint is not None and getattr(r, "user_override", False) == True)]
        if not rows:
            logger.info("No training rows available for partial_fit")
            return
        X = []
        y = []
        for r in rows:
            feat = r.data.get("features") if r.data else None
            if feat and r.label_setpoint is not None:
                X.append([feat[k] for k in FEATURE_ORDER])
                y.append(r.label_setpoint)
        if not X:
            logger.info("No labeled rows for partial_fit")
            return
        X = np.array(X)
        y = np.array(y)
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        else:
            try:
                self.scaler.partial_fit(X)
            except Exception:
                self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        if self.partial is None:
            self.partial = SGDRegressor(max_iter=1, tol=None, learning_rate="invscaling")
            self.partial.partial_fit(Xs, y)
        else:
            self.partial.partial_fit(Xs, y)
        joblib.dump({"model": self.partial, "scaler": self.scaler, "meta": {"feature_order": FEATURE_ORDER, "trained_at": datetime.datetime.utcnow().isoformat()}}, self.opts.get("model_path_partial"))
        logger.info("Partial model updated with %d samples", len(X))

    def full_retrain_job(self):
        rows = fetch_training_data(days=self.opts.get("buffer_days", 30))
        rows = [r for r in rows if (r.label_setpoint is not None and getattr(r, "user_override", False) == True)]
        if not rows:
            logger.info("No training rows available for full retrain")
            return
        X = []
        y = []
        # used_rows holds the corresponding ORM rows used for X,y
        used_rows = []
        for r in rows:
            feat = r.data.get("features") if r.data else None
            if feat:
                X.append([feat[k] for k in FEATURE_ORDER])
                y.append(r.label_setpoint)
                used_rows.append(r)
        
        if not X:
            logger.info("No labeled rows for full retrain")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
        param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
        n_splits = min(5, max(2, len(X)//10))
        tss = TimeSeriesSplit(n_splits=n_splits)
        gs = GridSearchCV(pipe, param_grid, cv=tss, scoring="neg_mean_absolute_error", n_jobs=1)
        gs.fit(X, y)
        best = gs.best_estimator_
        
        # produce out-of-fold predictions for MAE estimate
        oof_preds = np.zeros_like(y, dtype=float)
        tss_oof = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tss_oof.split(X):
            clone = gs.best_estimator_
            clone.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx] = clone.predict(X[test_idx])
        
        try:
            mae = float(mean_absolute_error(y, oof_preds))
        except Exception:
            preds_all = best.predict(X)
            mae = float(mean_absolute_error(y, preds_all))
        
        metadata = {"feature_order": FEATURE_ORDER, "best_params": gs.best_params_, "trained_at": datetime.datetime.utcnow().isoformat(), "mae": mae}
        joblib.dump({"model": best, "meta": metadata}, self.opts.get("model_path_full"))
        logger.info("Full model trained on %d labeled user samples (OOF MAE=%.3f) and saved", len(X), mae)
        
        # persist metric record
        try:
            insert_metric(model_type="full", mae=mae, n_samples=len(X), meta=metadata)
        except Exception:
            logger.exception("Failed to insert metric record")
        
        # update per-sample predicted_setpoint and prediction_error
        try:
            preds_all = best.predict(X)
            for i, row in enumerate(used_rows):
                try:
                    pred = float(preds_all[i])
                    err = abs(pred - float(y[i]))
                    update_sample_prediction(row.id, predicted_setpoint=pred, prediction_error=err)
                except Exception:
                    logger.exception("Failed updating sample prediction for sample %s", getattr(row, "id", None))
        except Exception:
            logger.exception("Failed to compute/update per-sample predictions")

