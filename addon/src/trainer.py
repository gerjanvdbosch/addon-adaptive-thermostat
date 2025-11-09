import os
import logging
import joblib
import numpy as np
from datetime import datetime

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
        if not self.opts.get("model_path_full") or not self.opts.get(
            "model_path_partial"
        ):
            raise RuntimeError(
                "model_path_full and model_path_partial must be provided in opts."
            )
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
                    logger.warning(
                        "Partial model feature_order mismatch; ignoring partial model"
                    )
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
        rows = [
            r
            for r in rows
            if (r.label_setpoint is not None and getattr(r, "user_override", False))
        ]
        if not rows:
            logger.info("No training rows available for partial_fit")
            return

        X = []
        y = []
        for r in rows:
            feat = (
                r.data.get("features") if r.data and isinstance(r.data, dict) else None
            )
            if not feat:
                continue
            try:
                vec = [
                    feat.get(k) if feat.get(k) is not None else 0.0
                    for k in FEATURE_ORDER
                ]
                X.append(vec)
                y.append(float(r.label_setpoint))
            except Exception:
                logger.exception(
                    "Skipping corrupt row %s in partial_fit", getattr(r, "id", None)
                )

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
                warm_start=True,
            )

        try:
            self.partial.partial_fit(Xs, y)
        except Exception:
            logger.exception("partial_fit failed")
            return

        # persist partial model + scaler + meta
        try:
            joblib.dump(
                {
                    "model": self.partial,
                    "scaler": self.scaler,
                    "meta": {
                        "feature_order": FEATURE_ORDER,
                        "trained_at": (datetime.utcnow().isoformat()),
                    },
                },
                self.opts.get("model_path_partial"),
            )
            logger.info("Partial model updated with %d samples", len(X))
        except Exception:
            logger.exception("Failed saving partial model")

def full_retrain_job(self):
    use_unlabeled = bool(self.opts.get("use_unlabeled", True))
    pseudo_limit = int(self.opts.get("pseudo_limit", 1000))
    weight_label = float(self.opts.get("weight_label", 1.0))
    weight_pseudo = float(self.opts.get("weight_pseudo", 0.25))

    rows = fetch_training_data(days=self.opts.get("buffer_days", 30))

    # only keep labeled user overrides (these are high-quality labels)
    labeled_rows = [
        r
        for r in rows
        if (r.label_setpoint is not None and getattr(r, "user_override", False))
    ]
    if not labeled_rows:
        logger.info("No labeled user_override rows available for full retrain")
        return

    X = []
    y = []
    used_rows = []
    for r in labeled_rows:
        feat = r.data.get("features") if r.data and isinstance(r.data, dict) else None
        if not feat:
            continue
        try:
            vec = [feat.get(k) if feat.get(k) is not None else 0.0 for k in FEATURE_ORDER]
            X.append(vec)
            # use the explicit label_setpoint for labeled samples
            y.append(float(r.label_setpoint))
            used_rows.append(r)
        except Exception:
            logger.exception("Skipping corrupt row %s in full_retrain", getattr(r, "id", None))

    # Collect pseudo-labeled samples from unlabeled rows using current_setpoint only.
    # Never use predicted_setpoint as a target for training.
    pseudo_X = []
    pseudo_y = []
    pseudo_count = 0
    if use_unlabeled:
        try:
            from db import fetch_unlabeled

            unlabeled_rows = fetch_unlabeled(limit=pseudo_limit)
            for r in unlabeled_rows:
                # Skip rows that already have an explicit label
                if getattr(r, "label_setpoint", None) is not None:
                    continue

                # Never use predicted_setpoint as training target
                if getattr(r, "predicted_setpoint", None) is not None:
                    logger.debug(
                        "Skipping unlabeled row %s because predicted_setpoint is present (never train on predictions)",
                        getattr(r, "id", None),
                    )
                    continue

                feat = r.data.get("features") if r.data and isinstance(r.data, dict) else None
                if not feat:
                    continue

                # Use current_setpoint from features as pseudo-label if present
                pseudo_label = feat.get("current_setpoint")
                if pseudo_label is None:
                    continue

                try:
                    vec = [feat.get(k) if feat.get(k) is not None else 0.0 for k in FEATURE_ORDER]
                    pseudo_X.append(vec)
                    pseudo_y.append(float(pseudo_label))
                except Exception:
                    logger.exception("Skipping corrupt unlabeled row %s in full_retrain", getattr(r, "id", None))

            pseudo_count = len(pseudo_X)
            logger.info("Collected %d pseudo samples for potential inclusion", pseudo_count)
        except Exception:
            logger.exception("Failed fetching unlabeled samples for pseudo-labeling")
            pseudo_X = []
            pseudo_y = []
            pseudo_count = 0
    else:
        logger.info("Pseudo-labeling disabled by configuration (use_unlabeled=False)")

    if pseudo_X:
        X.extend(pseudo_X)
        y.extend(pseudo_y)

    if not X:
        logger.info("No training data after aggregation (labeled + pseudo). Aborting full retrain.")
        return

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Prepare sample weights: labeled rows get weight_label, pseudo rows get weight_pseudo
    n_labeled = len(labeled_rows)
    sample_weight = np.ones(len(y), dtype=float) * weight_label
    if pseudo_X:
        n_total = len(y)
        sample_weight[n_labeled:n_total] = weight_pseudo

    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

    # Determine sensible n_splits for TimeSeriesSplit (based on labeled count)
    n_splits = min(3, max(2, n_labeled // 10))
    if n_splits >= max(2, n_labeled):
        n_splits = None

    if n_splits:
        tss = TimeSeriesSplit(n_splits=n_splits)
        gs = GridSearchCV(pipe, param_grid, cv=tss, scoring="neg_mean_absolute_error", n_jobs=1)
    else:
        gs = None

    try:
        if gs is not None:
            fit_params = {"model__sample_weight": sample_weight}
            gs.fit(X, y, **fit_params)
            best = gs.best_estimator_
        else:
            # single weighted fit; pass sample_weight to final estimator via pipeline fit kwargs
            pipe.fit(X, y, model__sample_weight=sample_weight)
            best = pipe
    except Exception:
        logger.exception("GridSearchCV or weighted fit failed; attempting unweighted single fit")
        try:
            pipe.fit(X, y)
            best = pipe
        except Exception:
            logger.exception("Full retrain failed completely")
            return

    # OOF MAE estimate: evaluate only on labeled samples
    try:
        if n_splits:
            oof_preds = np.zeros(n_labeled, dtype=float)
            tss_oof = TimeSeriesSplit(n_splits=n_splits)
            for train_idx, test_idx in tss_oof.split(X[:n_labeled]):
                clone = gs.best_estimator_ if hasattr(gs, "best_estimator_") else best
                clone.fit(X[train_idx], y[train_idx], model__sample_weight=sample_weight[train_idx])
                oof_preds[test_idx] = clone.predict(X[test_idx])
            mae = float(mean_absolute_error(y[:n_labeled], oof_preds[:n_labeled]))
        else:
            preds_all = best.predict(X[:n_labeled])
            mae = float(mean_absolute_error(y[:n_labeled], preds_all))
    except Exception:
        try:
            preds_all = best.predict(X[:n_labeled])
            mae = float(mean_absolute_error(y[:n_labeled], preds_all))
        except Exception:
            mae = None

    metadata = {
        "feature_order": FEATURE_ORDER,
        "best_params": getattr(gs, "best_params_", None) if gs is not None else None,
        "trained_at": (datetime.utcnow().isoformat()),
        "mae": mae,
        "n_samples": n_labeled,
        "pseudo_samples_used": pseudo_count,
        "use_unlabeled": bool(use_unlabeled),
    }

    try:
        joblib.dump({"model": best, "meta": metadata}, self.opts.get("model_path_full"))
        logger.info(
            "Full model trained on %d labeled user samples (OOF MAE=%s) and saved (with %d pseudo samples)",
            n_labeled,
            mae,
            pseudo_count,
        )
    except Exception:
        logger.exception("Failed saving full model")

    # persist metric record (n_samples refers to labeled samples only)
    try:
        insert_metric(model_type="full", mae=mae, n_samples=n_labeled, meta=metadata)
    except Exception:
        logger.exception("Failed to insert metric record")

    # update per-sample predictions for the labeled rows
    try:
        preds_labeled = best.predict(X[:n_labeled])
        for i, row in enumerate(used_rows):
            try:
                pred = float(preds_labeled[i])
                err = abs(pred - float(y[i])) if y is not None else None
                update_sample_prediction(row.id, predicted_setpoint=pred, prediction_error=err)
            except Exception:
                logger.exception("Failed updating sample prediction for sample %s", getattr(row, "id", None))
    except Exception:
        logger.exception("Failed to compute/update per-sample predictions")

