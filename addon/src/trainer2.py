import os
import shutil
import logging
import joblib
import time
import numpy as np
from datetime import datetime

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

from db import (
    fetch_training_data,
    fetch_unlabeled,
    update_sample_prediction,
)
from collector import FEATURE_ORDER

logger = logging.getLogger(__name__)


def _atomic_dump(obj, path):
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _assemble_matrix(rows, feature_order):
    X = []
    y = []
    used_rows = []
    for r in rows:
        feat = r.data if r.data and isinstance(r.data, dict) else None
        if not feat:
            continue
        try:
            label = float(r.label_setpoint)
            if not (14 <= label <= 25.0):
                logger.info("MLTrainer: invalid temp %s", label)
                continue

            vec = [
                feat.get(k) if feat.get(k) is not None else 0.0 for k in feature_order
            ]
            X.append(vec)
            y.append(label)
            used_rows.append(r)
        except Exception:
            logger.exception(
                "Skipping corrupt row %s in assemble_matrix", getattr(r, "id", None)
            )
    if not X:
        return None, None, []
    return np.array(X, dtype=float), np.array(y, dtype=float), used_rows


class Trainer2:
    """
    Production-ready sklearn ML trainer using HistGradientBoostingRegressor.

    Optional opts keys (use defaults as needed):
      - model_path_full (required)
      - buffer_days: 30
      - use_unlabeled: True
      - pseudo_limit: 1000
      - weight_label: 1.0
      - weight_pseudo: 0.1
      - val_fraction: 0.15
      - n_jobs: 1
      - n_iter_search: 20
      - search_mode: 'compact' or 'extended'
      - min_train_size: 30
      - min_search_labels: 50
      - refit_on_full: False
      - early_stopping_rounds: 50
      - promotion_delta_mae: 0.0
    """

    def __init__(self, ha_client, opts: dict):
        self.ha = ha_client
        self.opts = opts or {}
        # self.model_path = self.opts.get("model_path_full")
        self.model_path = "/config/models/full_model2.joblib"
        if not self.model_path:
            raise RuntimeError("model_path_full must be provided in opts.")
        self.feature_order = FEATURE_ORDER
        self.backend = "sklearn_histgb"
        self.random_state = int(self.opts.get("random_state", 42))

    def _fetch_data(self):
        buffer_days = int(self.opts.get("buffer_days", 30))
        rows = fetch_training_data(days=buffer_days)
        # labeled = confirmed user overrides (collector should ensure confirm)
        labeled_rows = [
            r
            for r in rows
            if r.label_setpoint is not None and getattr(r, "user_override", False)
        ]
        X_lab, y_lab, used_rows = _assemble_matrix(labeled_rows, self.feature_order)
        X_list, y_list = ([], [])
        if X_lab is not None:
            X_list.append(X_lab)
            y_list.append(y_lab)

        pseudo_count = 0
        if bool(self.opts.get("use_unlabeled", True)):
            try:
                unl = fetch_unlabeled(limit=int(self.opts.get("pseudo_limit", 1000)))
                pseudo_rows = []
                for r in unl:
                    if getattr(r, "label_setpoint", None) is not None:
                        continue
                    feat = r.data if r.data and isinstance(r.data, dict) else None
                    if not feat:
                        continue
                    pseudo_label = feat.get("current_setpoint")
                    if pseudo_label is None:
                        continue
                    r.label_setpoint = pseudo_label  # temporary for assembly
                    pseudo_rows.append(r)
                X_pseudo, y_pseudo, _ = _assemble_matrix(
                    pseudo_rows, self.feature_order
                )
                if X_pseudo is not None:
                    X_list.append(X_pseudo)
                    y_list.append(y_pseudo)
                    pseudo_count = len(X_pseudo)
                    logger.info("MLTrainer: collected %d pseudo samples", pseudo_count)
            except Exception:
                logger.exception(
                    "MLTrainer: failed fetching unlabeled for pseudo labeling"
                )
                pseudo_count = 0

        if not X_list:
            return None, None, None, None, 0

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        n_labeled = len(used_rows)
        n_total = len(y)

        weight_label = float(self.opts.get("weight_label", 1.0))
        weight_pseudo = float(self.opts.get("weight_pseudo", 0.1))
        sample_weight = np.ones(n_total, dtype=float) * weight_label
        if n_total > n_labeled:
            sample_weight[n_labeled:n_total] = weight_pseudo

        # logging
        logger.info(
            "Data fetched: n_labeled=%d n_total=%d n_pseudo=%d",
            n_labeled,
            n_total,
            pseudo_count,
        )

        return X, y, used_rows, sample_weight, int(pseudo_count)

    def _search_estimator(self):
        base = HistGradientBoostingRegressor(random_state=self.random_state)

        user_dist = self.opts.get("search_param_dist")
        if user_dist:
            return base, user_dist

        mode = self.opts.get("search_mode", "extended")

        compact = {
            "max_iter": [200, 400, 800],
            "max_leaf_nodes": [31, 63],
            "learning_rate": [0.01, 0.03, 0.05],
            "min_data_in_leaf": [10, 30],
            "l2_regularization": [0.0, 0.01, 0.1],
            "feature_fraction": [0.7, 0.9],
            "subsample": [0.8, 1.0],
        }
        extended = {
            "max_iter": [300, 600, 1000, 1500],
            "max_leaf_nodes": [15, 31, 63, 127],
            "learning_rate": [0.005, 0.01, 0.02, 0.05],
            "min_data_in_leaf": [5, 10, 20, 50],
            "l2_regularization": [0.0, 1e-3, 0.01, 0.1, 1.0],
            "feature_fraction": [0.6, 0.8, 1.0],
            "subsample": [0.6, 0.8, 1.0],
        }

        param_dist = compact if mode == "compact" else extended
        return base, param_dist

    def _time_splits(self, n_labeled):
        min_train = int(self.opts.get("min_train_size", 30))
        if n_labeled < min_train:
            return None
        n_splits = min(3, max(2, n_labeled // 10))
        if n_splits >= n_labeled:
            return None
        return n_splits

    def _report_household_drift(self, used_rows, preds, y_true):
        """
        Compute MAE per household cohort and log results. Expects used_rows items may have household_id attribute.
        """
        try:
            hh_map = {}
            for i, r in enumerate(used_rows):
                hh = getattr(r, "household_id", None) or "unknown"
                hh_map.setdefault(hh, []).append((preds[i], y_true[i]))
            for hh, vals in hh_map.items():
                errors = [
                    abs(p - t) for p, t in vals if np.isfinite(p) and np.isfinite(t)
                ]
                if not errors:
                    continue
                hh_mae = float(np.mean(errors))
                logger.debug("Household %s MAE=%.4f (n=%d)", hh, hh_mae, len(errors))
        except Exception:
            logger.exception("Failed computing per-household drift")

    def train_job(self, force: bool = False):
        start_time = time.time()
        X, y, used_rows, sample_weight, pseudo_count = self._fetch_data()
        if X is None:
            logger.info("MLTrainer: no training data")
            return

        n_total = len(y)
        n_labeled = len(used_rows)

        # reduce search if too few labeled samples
        min_search_labels = int(self.opts.get("min_search_labels", 50))
        if n_labeled < min_search_labels:
            # shrink n_iter and force compact mode
            self.opts["search_mode"] = "compact"
            n_iter = max(5, int(self.opts.get("n_iter_search", 20) // 2))
            logger.info(
                "Few labeled samples (%d), shrinking hypersearch to n_iter=%d and compact mode",
                n_labeled,
                n_iter,
            )
        else:
            n_iter = int(self.opts.get("n_iter_search", 20))

        # chronological validation slice for early stopping / final eval
        val_frac = float(self.opts.get("val_fraction", 0.15))
        val_size = max(1, int(n_total * val_frac))
        train_idx = slice(0, n_total - val_size)
        val_idx = slice(n_total - val_size, n_total)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        sw_train = sample_weight[train_idx] if sample_weight is not None else None

        logger.info(
            "MLTrainer: train %d, val %d (labeled=%d pseudo=%d)",
            len(y_train),
            len(y_val),
            n_labeled,
            pseudo_count,
        )

        # pipeline without scaler (trees don't need it)
        pipe = Pipeline(
            [("model", HistGradientBoostingRegressor(random_state=self.random_state))]
        )

        base_est, param_dist = self._search_estimator()
        # Filter param_dist to only parameters that exist on the estimator
        allowed = set(base_est.get_params().keys())
        filtered = {}
        ignored = []
        for k, v in (param_dist or {}).items():
            if k in allowed:
                filtered[k] = v
            else:
                ignored.append(k)
        if ignored:
            logger.warning(
                "Ignoring unsupported hyperparam keys for estimator: %s",
                ", ".join(sorted(ignored)),
            )
        param_dist_pipe = {f"model__{k}": v for k, v in filtered.items()}

        n_jobs = int(self.opts.get("n_jobs", 1))
        tss_splits = self._time_splits(n_labeled)
        cv = TimeSeriesSplit(n_splits=tss_splits) if tss_splits else None

        chosen_params = None
        best_pipe = None
        best_score = None
        search_failed = False
        edge_flag = False
        best_iteration = None

        try:
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_dist_pipe,
                n_iter=n_iter,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=0,
            )
            fit_kwargs = {}
            if sw_train is not None:
                fit_kwargs["model__sample_weight"] = sw_train
            logger.info(
                "MLTrainer: running hyperparameter search (n_iter=%d, cv=%s)",
                n_iter,
                "TimeSeriesSplit" if cv else "None",
            )
            search.fit(X_train, y_train, **fit_kwargs)
            best_pipe = search.best_estimator_
            chosen_params = getattr(search, "best_params_", None)
            best_score = getattr(search, "best_score_", None)
            logger.info(
                "Hypersearch complete: best_score=%s best_params=%s",
                best_score,
                chosen_params,
            )
        except Exception:
            search_failed = True
            logger.exception(
                "MLTrainer: search failed; falling back to default estimator"
            )
            # fallback to reasonable default estimator params
            best_pipe = Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            max_iter=300,
                            max_leaf_nodes=63,
                            learning_rate=0.05,
                            random_state=self.random_state,
                        ),
                    )
                ]
            )
            try:
                if "reg_lambda" in self.opts:
                    best_pipe.set_params(
                        model__l2_regularization=float(self.opts.get("reg_lambda", 1.0))
                    )
                best_pipe.fit(X_train, y_train, model__sample_weight=sw_train)
            except Exception:
                logger.exception("MLTrainer: fallback fit failed")
                return

        # After search/fallback, try capture best_iteration if model supports it
        try:
            model_obj = best_pipe.named_steps.get("model")
            if hasattr(model_obj, "n_iter_"):
                best_iteration = int(getattr(model_obj, "n_iter_"))
            elif hasattr(model_obj, "best_iteration_"):
                best_iteration = int(getattr(model_obj, "best_iteration_"))
            else:
                best_iteration = None
            if best_iteration is not None:
                logger.info("Best iteration: %s", best_iteration)
                max_iter_chosen = None
                if chosen_params:
                    max_iter_chosen = chosen_params.get("model__max_iter")
                else:
                    max_iter_chosen = getattr(model_obj, "max_iter", None)
                if max_iter_chosen is not None and best_iteration >= int(
                    max_iter_chosen
                ):
                    msg = f"best_iteration ({best_iteration}) reached max_iter ({max_iter_chosen})"
                    logger.warning(msg)
                    # mark to expand grid next run
                    self.opts["expand_search_next"] = True
        except Exception:
            logger.exception("Failed to capture best_iteration")

        # chosen params edge detection
        try:
            if chosen_params:
                for k, v in chosen_params.items():
                    key = k.replace("model__", "")
                    vals = filtered.get(key) or param_dist.get(key)
                    if vals and (np.isclose(v, min(vals)) or np.isclose(v, max(vals))):
                        edge_flag = True
                        logger.warning("Chosen param %s=%s is on grid edge", key, v)
        except Exception:
            logger.exception("Edge detection failed")

        # final refit option
        try:
            final_refit = bool(self.opts.get("refit_on_full", False))
            if final_refit:
                fit_kwargs_all = {}
                if sample_weight is not None:
                    fit_kwargs_all["model__sample_weight"] = sample_weight
                best_pipe.fit(X, y, **fit_kwargs_all)

                def predict_fn(Xq):
                    return best_pipe.predict(Xq)

            else:

                def predict_fn(Xq):
                    return best_pipe.predict(Xq)

        except Exception:
            logger.exception("MLTrainer: failed final refit/predict setup")
            return

        # Compute OOF MAE on labeled samples only
        mae = None
        try:
            if n_labeled == 0:
                mae = None
            else:
                preds_labeled = predict_fn(X[:n_labeled])
                mae = float(mean_absolute_error(y[:n_labeled], preds_labeled))
                # per-household drift
                try:
                    self._report_household_drift(
                        used_rows, preds_labeled, y[:n_labeled]
                    )
                except Exception:
                    logger.exception("Per-household drift reporting failed")
        except Exception:
            logger.exception("MLTrainer: failed OOF MAE computation")
            mae = None

        # log/train/val MAE if val set exists
        val_mae = None
        try:
            if X_val is not None and len(X_val):
                val_preds = predict_fn(X_val)
                val_mae = float(mean_absolute_error(y_val, val_preds))
        except Exception:
            logger.exception("Failed computing val MAE")
            val_mae = None

        # metrics/logs
        runtime_seconds = time.time() - start_time
        logger.info(
            "Training finished runtime_seconds=%.2f n_labeled=%d n_pseudo=%d mae=%s val_mae=%s chosen_params=%s best_iteration=%s",
            runtime_seconds,
            n_labeled,
            pseudo_count,
            mae,
            val_mae,
            chosen_params,
            best_iteration,
        )

        # Compare to existing model
        existing_mae = None
        existing_meta = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                obj = joblib.load(self.model_path)
                existing_meta = obj.get("meta", {})
                existing_mae = existing_meta.get("mae")
            except Exception:
                logger.exception(
                    "MLTrainer: failed reading existing model for MAE comparison"
                )

        # promotion logic
        promotion_delta = float(self.opts.get("promotion_delta_mae", 0.0))
        promote = force or (
            mae is not None
            and (existing_mae is None or (mae + promotion_delta) < existing_mae)
        )
        if not promote:
            logger.info(
                "MLTrainer: new MAE %s not better than existing %s; skipping save",
                mae,
                existing_mae,
            )
            return

        # metadata
        metadata = {
            "feature_order": self.feature_order,
            "backend": self.backend,
            "trained_at": datetime.utcnow().isoformat(),
            "mae": mae,
            "val_mae": val_mae,
            "n_samples": n_labeled,
            "pseudo_samples_used": int(pseudo_count),
            "use_unlabeled": bool(self.opts.get("use_unlabeled", True)),
            "chosen_params": chosen_params,
            "search_best_score": best_score,
            "best_iteration": best_iteration,
            "edge_on_param": bool(edge_flag),
            "search_failed": bool(search_failed),
            "random_state": self.random_state,
            "runtime_seconds": runtime_seconds,
        }

        # persist model and meta
        try:
            if self.model_path and os.path.exists(self.model_path):
                try:
                    shutil.copy2(self.model_path, self.model_path + ".bak")
                except Exception:
                    logger.warning(
                        "MLTrainer: failed to create .bak for existing model; continuing"
                    )
            payload = {"model": best_pipe, "meta": metadata}
            _atomic_dump(payload, self.model_path)
            logger.info(
                "MLTrainer: saved model (MAE %s -> was %s) trained on %d labeled + %d pseudo samples",
                mae,
                existing_mae,
                n_labeled,
                pseudo_count,
            )
        except Exception:
            logger.exception("MLTrainer: failed saving model")
            return

        # update per-sample predictions
        try:
            if n_labeled > 0:
                preds = predict_fn(X[:n_labeled])
                for i, row in enumerate(used_rows):
                    try:
                        pred = float(preds[i])
                        err = abs(pred - float(y[i])) if y is not None else None
                        update_sample_prediction(
                            row.id, predicted_setpoint=pred, prediction_error=err
                        )
                    except Exception:
                        logger.exception(
                            "MLTrainer: failed updating sample prediction for %s",
                            getattr(row, "id", None),
                        )
        except Exception:
            logger.exception(
                "MLTrainer: failed computing/updating per-sample predictions"
            )
