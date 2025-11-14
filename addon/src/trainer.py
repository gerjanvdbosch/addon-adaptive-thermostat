import os
import logging
import joblib
import numpy as np
from datetime import datetime

from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from db import fetch_training_data, update_sample_prediction, fetch_unlabeled
from collector import FEATURE_ORDER

logger = logging.getLogger(__name__)


def _scale_features_for_train(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    sigma_safe = np.where(sigma < 1e-8, 1.0, sigma)
    Xn = (X - mu) / sigma_safe
    return Xn, mu, sigma_safe


def _walk_forward_val(
    X: np.ndarray, y: np.ndarray, n_splits: int, min_train_size: int, lambda_reg: float
):
    if n_splits < 1:
        return None, []
    tss = TimeSeriesSplit(n_splits=n_splits)
    val_mses = []
    for train_idx, test_idx in tss.split(X):
        if train_idx.shape[0] < min_train_size:
            continue
        try:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler().fit(X_train)
            Xt = scaler.transform(X_train)
            Xv = scaler.transform(X_test)
            model = Ridge(alpha=float(lambda_reg))
            model.fit(Xt, y_train)
            preds = model.predict(Xv)
            mse = float(mean_squared_error(y_test, preds))
            val_mses.append(mse)
        except Exception:
            logger.exception("Walk-forward fold failed; skipping fold")
            continue
    if not val_mses:
        return None, []
    return float(np.mean(val_mses)), val_mses


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

        # Try load partial model if present and compatible
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
        rows = [
            r
            for r in rows
            if r.label_setpoint is not None and getattr(r, "user_override", False)
        ]
        if not rows:
            logger.info("No training rows available for partial_fit")
            return

        X, y = [], []
        for r in rows:
            feat = r.data if r.data and isinstance(r.data, dict) else None
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

        if self.scaler is None:
            self.scaler = StandardScaler()
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        if self.partial is None:
            self.partial = SGDRegressor(
                max_iter=1,
                tol=None,
                learning_rate="invscaling",
                eta0=0.002,
                alpha=0.001,
                power_t=0.25,
                penalty="l2",
                warm_start=True,
            )

        try:
            self.partial.partial_fit(Xs, y)
        except Exception:
            logger.exception("partial_fit failed")
            return

        try:
            meta = {
                "feature_order": FEATURE_ORDER,
                "trained_at": datetime.now().isoformat(),
                "n_samples": len(X),
            }
            joblib.dump(
                {"model": self.partial, "scaler": self.scaler, "meta": meta},
                self.opts.get("model_path_partial"),
            )
            logger.info("Partial model updated with %d samples", len(X))
        except Exception:
            logger.exception("Failed saving partial model")

    def full_retrain_job(self, force: bool = False):
        use_unlabeled = bool(self.opts.get("use_unlabeled", True))
        pseudo_limit = int(self.opts.get("pseudo_limit", 1000))
        weight_label = float(self.opts.get("weight_label", 1.0))
        weight_pseudo = float(self.opts.get("weight_pseudo", 0.1))

        rows = fetch_training_data(days=self.opts.get("buffer_days", 30))
        labeled_rows = [
            r
            for r in rows
            if r.label_setpoint is not None and getattr(r, "user_override", False)
        ]
        if not labeled_rows:
            logger.info("No labeled user_override rows available for full retrain")
            return

        X_list, y_list, used_rows = [], [], []
        for r in labeled_rows:
            feat = r.data if r.data and isinstance(r.data, dict) else None
            if not feat:
                continue
            try:
                vec = [
                    feat.get(k) if feat.get(k) is not None else 0.0
                    for k in FEATURE_ORDER
                ]
                X_list.append(vec)
                y_list.append(float(r.label_setpoint))
                used_rows.append(r)
            except Exception:
                logger.exception(
                    "Skipping corrupt row %s in full_retrain", getattr(r, "id", None)
                )

        pseudo_X, pseudo_y = [], []
        pseudo_count = 0
        if use_unlabeled:
            try:
                unlabeled_rows = fetch_unlabeled(limit=pseudo_limit)
                for r in unlabeled_rows:
                    if getattr(r, "label_setpoint", None) is not None:
                        continue
                    feat = r.data if r.data and isinstance(r.data, dict) else None
                    if not feat:
                        continue
                    pseudo_label = feat.get("current_setpoint")
                    if pseudo_label is None:
                        continue
                    try:
                        vec = [
                            feat.get(k) if feat.get(k) is not None else 0.0
                            for k in FEATURE_ORDER
                        ]
                        pseudo_X.append(vec)
                        pseudo_y.append(float(pseudo_label))
                    except Exception:
                        logger.exception(
                            "Skipping corrupt unlabeled row %s in full_retrain",
                            getattr(r, "id", None),
                        )
                pseudo_count = len(pseudo_X)
                logger.info(
                    "Collected %d pseudo samples for potential inclusion", pseudo_count
                )
            except Exception:
                logger.exception(
                    "Failed fetching unlabeled samples for pseudo-labeling"
                )
                pseudo_X, pseudo_y, pseudo_count = [], [], 0
        else:
            logger.info(
                "Pseudo-labeling disabled by configuration (use_unlabeled=False)"
            )

        if pseudo_X:
            X_list.extend(pseudo_X)
            y_list.extend(pseudo_y)

        if not X_list:
            logger.info(
                "No training data after aggregation (labeled + pseudo). Aborting full retrain."
            )
            return

        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)

        n_labeled = len(used_rows)
        n_total = len(y)
        sample_weight = np.ones(n_total, dtype=float) * weight_label
        if n_total > n_labeled:
            sample_weight[n_labeled:n_total] = weight_pseudo

        # Adaptive lambda
        base_lambda = float(self.opts.get("base_lambda", 1.0))
        min_lambda = float(self.opts.get("min_lambda", 0.01))
        max_lambda = float(self.opts.get("max_lambda", 5.0))
        warmup_size = int(self.opts.get("warmup_size", 20))
        scale = min(1.0, float(n_total) / max(1, warmup_size))
        adaptive_lambda = float(base_lambda) * (
            min_lambda + (max_lambda - min_lambda) * scale
        )
        adaptive_lambda = max(min_lambda, min(max_lambda, adaptive_lambda))

        cond_est = None
        try:
            Xn_full, mu_full, sigma_full = _scale_features_for_train(X)
            X_aug = np.hstack([Xn_full, np.ones((Xn_full.shape[0], 1))])
            cond_est = np.linalg.cond(X_aug)
            if np.isfinite(cond_est):
                cond_adj = min(1e8, cond_est)
                lambda_cond = min(max_lambda, max(min_lambda, cond_adj / 1e5))
                adaptive_lambda = max(adaptive_lambda, lambda_cond)
        except Exception:
            cond_est = None

        mean_val_mse = None
        try:
            min_train_size = int(self.opts.get("min_train_size", 50))
            walk_n_splits = int(self.opts.get("walk_n_splits", 5))
            if n_total > min_train_size and walk_n_splits >= 1:
                mean_val_mse = _walk_forward_val(
                    X,
                    y,
                    n_splits=walk_n_splits,
                    min_train_size=min_train_size,
                    lambda_reg=adaptive_lambda,
                )
        except Exception:
            mean_val_mse = None

        logger.info(
            "Adaptive lambda=%.6f cond=%s walk_val_mse=%s n=%d",
            adaptive_lambda,
            str(cond_est),
            str(mean_val_mse),
            n_total,
        )

        # Param grid centered on adaptive_lambda
        alphas = sorted(list({adaptive_lambda * f for f in (0.25, 0.5, 1.0, 2.0, 4.0)}))
        param_grid = {"model__alpha": [a for a in alphas if a > 0]}

        # single pipeline instance to reuse
        pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

        # Determine TimeSeries CV splits based on labeled count
        n_splits = None
        if n_labeled >= 10:
            n_splits = min(3, max(2, n_labeled // 10))
            if n_splits >= n_labeled:
                n_splits = None

        tss = TimeSeriesSplit(n_splits=n_splits) if n_splits else None
        gs = (
            GridSearchCV(
                pipe, param_grid, cv=tss, scoring="neg_mean_absolute_error", n_jobs=1
            )
            if tss
            else None
        )

        try:
            if gs is not None:
                fit_params = {"model__sample_weight": sample_weight}
                logger.info(
                    "Running GridSearchCV on alphas=%s n_splits=%s",
                    param_grid["model__alpha"],
                    n_splits,
                )
                gs.fit(X, y, **fit_params)
                best = gs.best_estimator_
                chosen_alpha = gs.best_params_.get("model__alpha")
            else:
                # fallback: set alpha directly on pipeline then fit
                logger.info(
                    "No CV configured; using adaptive_lambda=%.6f for Ridge",
                    adaptive_lambda,
                )
                pipe.set_params(model__alpha=adaptive_lambda)
                pipe.fit(X, y, model__sample_weight=sample_weight)
                best = pipe
                chosen_alpha = adaptive_lambda
        except Exception:
            logger.exception(
                "GridSearchCV or weighted fit failed; attempting unweighted single fit"
            )
            try:
                # final fallback: unweighted fit on pipeline defaults
                pipe.set_params(model__alpha=adaptive_lambda)
                pipe.fit(X, y)
                best = pipe
                chosen_alpha = adaptive_lambda
            except Exception:
                logger.exception("Full retrain failed completely")
                return

        logger.info("Chosen alpha: %s", str(chosen_alpha))

        # OOF MAE estimate: evaluate only on labeled samples
        try:
            if n_labeled == 0:
                mae = None
            elif tss:
                oof_preds = np.zeros(n_labeled, dtype=float)
                tss_oof = TimeSeriesSplit(n_splits=n_splits)
                for train_idx, test_idx in tss_oof.split(X[:n_labeled]):
                    clone = gs.best_estimator_ if gs else best
                    try:
                        clone.fit(
                            X[train_idx],
                            y[train_idx],
                            model__sample_weight=sample_weight[train_idx],
                        )
                    except TypeError:
                        clone.fit(X[train_idx], y[train_idx])
                    oof_preds[test_idx] = clone.predict(X[test_idx])
                mae = float(mean_absolute_error(y[:n_labeled], oof_preds[:n_labeled]))
            else:
                preds_all = best.predict(X[:n_labeled])
                mae = float(mean_absolute_error(y[:n_labeled], preds_all))
        except Exception:
            try:
                if n_labeled == 0:
                    mae = None
                else:
                    preds_all = best.predict(X[:n_labeled])
                    mae = float(mean_absolute_error(y[:n_labeled], preds_all))
            except Exception:
                mae = None

        # Compare to existing model MAE
        existing_mae = None
        full_path = self.opts.get("model_path_full")
        if full_path and os.path.exists(full_path):
            try:
                obj = joblib.load(full_path)
                existing_mae = obj.get("meta", {}).get("mae")
            except Exception:
                logger.exception(
                    "Failed reading existing full model for MAE comparison"
                )

        if not force:
            if mae is None or (existing_mae is not None and mae >= existing_mae):
                logger.info(
                    "New model: MAE %s not better than existing %s; skipping overwrite",
                    mae,
                    existing_mae,
                )
                return
        else:
            logger.info(
                "Force flag set: bypassing MAE comparison and forcing overwrite (new_mae=%s existing_mae=%s)",
                mae,
                existing_mae,
            )

        # Top features (best effort)
        try:
            coefs = best.named_steps["model"].coef_
            top_idx = np.argsort(np.abs(coefs))[-3:]
            top_features = {FEATURE_ORDER[i]: float(coefs[i]) for i in top_idx}
        except Exception:
            top_features = {}

        metadata = {
            "feature_order": FEATURE_ORDER,
            "best_params": (
                getattr(gs, "best_params_", None)
                if gs
                else {"model__alpha": chosen_alpha}
            ),
            "trained_at": datetime.now().isoformat(),
            "mae": mae,
            "n_samples": n_labeled,
            "pseudo_samples_used": pseudo_count,
            "use_unlabeled": bool(use_unlabeled),
            "adaptive_lambda": float(adaptive_lambda),
            "chosen_alpha": float(chosen_alpha) if chosen_alpha is not None else None,
            "walk_val_mse": mean_val_mse,
            "cond_estimate": float(cond_est) if cond_est is not None else None,
            "top_features": top_features,
        }

        try:
            joblib.dump({"model": best, "meta": metadata}, full_path)
            logger.info(
                "Full model updated: OOF-MAE %s improved over %s (trained on %d labeled + %d pseudo samples)",
                mae,
                existing_mae,
                n_labeled,
                pseudo_count,
            )
        except Exception:
            logger.exception("Failed saving full model")
            return

        # update per-sample predictions for the labeled rows
        try:
            if n_labeled > 0:
                preds_labeled = best.predict(X[:n_labeled])
                for i, row in enumerate(used_rows):
                    try:
                        pred = float(preds_labeled[i])
                        err = abs(pred - float(y[i])) if y is not None else None
                        update_sample_prediction(
                            row.id, predicted_setpoint=pred, prediction_error=err
                        )
                    except Exception:
                        logger.exception(
                            "Failed updating sample prediction for sample %s",
                            getattr(row, "id", None),
                        )
        except Exception:
            logger.exception("Failed to compute/update per-sample predictions")
