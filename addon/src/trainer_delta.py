import os
import shutil
import logging
import joblib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import loguniform, randint

from db import fetch_training_setpoints
from collector import FEATURE_ORDER

logger = logging.getLogger(__name__)


def _atomic_dump(obj: Any, path: str) -> None:
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def _assemble_matrix_delta(
    rows: List[Any], feature_order: List[str]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Any]]:
    """
    Build X and y where y = setpoint - current_setpoint (delta).
    Skip rows that lack current_setpoint or have trivial delta and are not override.
    """
    X: List[List[float]] = []
    y: List[float] = []
    used_rows: List[Any] = []

    for r in rows:
        feat = r.data if r.data and isinstance(r.data, dict) else None
        if not feat:
            continue
        try:
            if r.setpoint is None:
                continue
            label = _safe_float(r.setpoint, None)
            if label is None:
                continue
            # sanity bounds on absolute setpoint
            if not (5 <= label <= 30):
                logger.debug(
                    "Skipping row %s: label out of plausible bounds %s",
                    getattr(r, "id", None),
                    label,
                )
                continue
            curr_raw = feat.get("current_setpoint")
            if curr_raw is None:
                logger.debug(
                    "Skipping row %s: missing current_setpoint", getattr(r, "id", None)
                )
                continue
            current = _safe_float(curr_raw, None)
            if current is None:
                logger.debug(
                    "Skipping row %s: non-numeric current_setpoint %r",
                    getattr(r, "id", None),
                    curr_raw,
                )
                continue

            delta = label - current
            # skip trivial deltas unless trusted override or auto_label flag
            if (
                abs(delta) < 1e-6
                and not getattr(r, "override", False)
                and not getattr(r, "auto_label", False)
            ):
                logger.debug(
                    "Skipping row %s: trivial delta and not override",
                    getattr(r, "id", None),
                )
                continue

            vec: List[float] = []
            for k in feature_order:
                v = feat.get(k)
                if v is None:
                    vec.append(0.0)
                else:
                    try:
                        vec.append(float(v))
                    except Exception:
                        logger.debug(
                            "Coercing non-numeric feature %s in row %s to 0.0",
                            k,
                            getattr(r, "id", None),
                        )
                        vec.append(0.0)

            X.append(vec)
            y.append(delta)
            used_rows.append(r)
        except Exception:
            logger.exception(
                "Skipping corrupt row %s in assemble_matrix_delta",
                getattr(r, "id", None),
            )

    if not X:
        return None, None, []
    return np.array(X, dtype=float), np.array(y, dtype=float), used_rows


def _clean_train_arrays(
    Xa: np.ndarray, ya: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure numeric, finite arrays.
    Drops rows where any feature or y is non-finite.
    Raises ValueError if nothing remains.
    """
    Xa = np.asarray(Xa, dtype=float)
    ya = np.asarray(ya, dtype=float)

    if Xa.ndim != 2:
        raise ValueError(f"X must be 2D array, got ndim={Xa.ndim}")
    if ya.ndim != 1:
        raise ValueError(f"y must be 1D array, got ndim={ya.ndim}")
    if Xa.shape[0] != ya.shape[0]:
        raise ValueError(f"Row count mismatch X ({Xa.shape[0]}) vs y ({ya.shape[0]})")

    finite_mask = np.isfinite(ya) & np.all(np.isfinite(Xa), axis=1)
    dropped = np.count_nonzero(~finite_mask)
    if dropped:
        logger.warning("Dropping %d non-finite training rows", int(dropped))

    if not np.any(finite_mask):
        raise ValueError("No finite training rows after cleaning")

    Xa_clean = Xa[finite_mask]
    ya_clean = ya[finite_mask]
    return Xa_clean, ya_clean


class TrainerDelta:
    """
    Trainer that learns delta = setpoint - current_setpoint.
    """

    def __init__(self, ha_client=None, opts: Optional[Dict] = None):
        self.ha = ha_client
        self.opts = opts or {}
        self.model_path = self.opts.get(
            "model_path", "/config/models/full_model_delta.joblib"
        )
        self.feature_order = FEATURE_ORDER
        self.random_state = int(self.opts.get("random_state", 42))

        # safe defaults
        if "warning_std_threshold" not in self.opts:
            self.opts["warning_std_threshold"] = 1e-3

    def _fetch_data(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[Any], int]:
        rows = fetch_training_setpoints(days=int(self.opts.get("buffer_days", 30)))
        labeled_rows = [r for r in rows if r.setpoint is not None]
        X_lab, y_lab, used_rows = _assemble_matrix_delta(
            labeled_rows, self.feature_order
        )

        if X_lab is None:
            return None, None, [], 0

        X = X_lab
        y = y_lab

        # diagnostics and constant-check
        try:
            if y is not None and len(y):
                mean = float(np.mean(y))
                std = float(np.std(y))
                logger.info(
                    "Training deltas: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f",
                    len(y),
                    mean,
                    std,
                    float(np.min(y)),
                    float(np.max(y)),
                )
                if std < float(self.opts.get("warning_std_threshold", 1e-3)):
                    logger.warning(
                        "Training deltas near-constant (std=%.6f); likely labels == current_setpoint across dataset",
                        std,
                    )
        except Exception:
            logger.exception("Failed logging training delta stats")

        return X, y, used_rows, 0

    def _search_param_dist(self):
        compact = {
            "max_iter": [100, 200, 400],
            "max_leaf_nodes": [15, 31, 63],
            "learning_rate": [0.001, 0.01, 0.03],
            "min_samples_leaf": [10, 20, 40],
            "l2_regularization": [1e-6, 0.001, 0.01],
            "max_features": [0.5, 0.7, 1.0],
            "validation_fraction": [0.05, 0.1, 0.15],
        }
        extended = {
            "max_iter": [300, 600, 1000, 1500, 2000],
            "max_leaf_nodes": [15, 31, 63, 127, 255],
            "learning_rate": [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05],
            "min_samples_leaf": [5, 10, 20, 40, 80],
            "l2_regularization": [1e-6, 1e-3, 0.01, 0.1, 1.0],
            "max_features": [0.4, 0.6, 0.8, 1.0],
            "validation_fraction": [0.05, 0.1, 0.15],
        }
        return compact, extended

    def train_job(self, force: bool = False):
        start = time.time()
        X, y_delta, used_rows, _ = self._fetch_data()
        if X is None:
            logger.info("TrainerDelta: no training data")
            return

        n_total = len(y_delta)
        n_labeled = len(used_rows)
        self.opts["last_n_labeled"] = n_labeled

        # simple train/val split (temporal): last fraction is val
        val_frac = float(self.opts.get("val_fraction", 0.15))
        val_size = max(1, int(n_total * val_frac))
        val_size = min(val_size, max(1, n_total - 1))
        train_idx = slice(0, n_total - val_size)
        val_idx = slice(n_total - val_size, n_total)

        X_train, y_train = X[train_idx], y_delta[train_idx]
        X_val, y_val = X[val_idx], y_delta[val_idx]

        logger.info(
            "TrainerDelta: train %d val %d (labeled=%d)",
            len(X_train),
            len(X_val) if X_val is not None else 0,
            n_labeled,
        )

        # Clean training arrays before any fit
        try:
            X_train, y_train = _clean_train_arrays(X_train, y_train)
        except Exception as e:
            logger.exception("Training data invalid after cleaning: %s", e)
            return

        # Clean validation arrays (non-finite rows removed)
        try:
            if X_val is not None and len(X_val):
                X_val = np.asarray(X_val, dtype=float)
                y_val = np.asarray(y_val, dtype=float)
                finite_val_mask = np.isfinite(y_val) & np.all(
                    np.isfinite(X_val), axis=1
                )
                if not np.any(finite_val_mask):
                    logger.warning(
                        "Validation set contains no finite rows; skipping val MAE"
                    )
                    X_val = None
                    y_val = None
                else:
                    X_val = X_val[finite_val_mask]
                    y_val = y_val[finite_val_mask]
        except Exception:
            logger.exception(
                "Validation cleaning failed; proceeding without validation metrics"
            )
            X_val = None
            y_val = None

        pipe = Pipeline(
            [("model", HistGradientBoostingRegressor(random_state=self.random_state))]
        )
        compact, extended = self._search_param_dist()
        mode = self.opts.get("search_mode", "compact")
        param_dist = compact if mode == "compact" else extended

        sampled = {}
        sampled["learning_rate"] = loguniform(1e-4, 1e-1)
        sampled["l2_regularization"] = loguniform(1e-6, 1.0)
        mi_min, mi_max = min(param_dist["max_iter"]), max(param_dist["max_iter"])
        sampled["max_iter"] = randint(mi_min, mi_max + 1)
        mln_min, mln_max = min(param_dist["max_leaf_nodes"]), max(
            param_dist["max_leaf_nodes"]
        )
        sampled["max_leaf_nodes"] = randint(mln_min, mln_max + 1)
        ms_min, ms_max = min(param_dist["min_samples_leaf"]), max(
            param_dist["min_samples_leaf"]
        )
        sampled["min_samples_leaf"] = randint(ms_min, ms_max + 1)
        sampled["max_features"] = param_dist.get("max_features", [1.0])
        sampled["validation_fraction"] = param_dist.get("validation_fraction", [0.1])
        param_dist_pipe = {f"model__{k}": v for k, v in sampled.items()}

        n_iter = int(self.opts.get("n_iter_compact", 20))
        if n_labeled >= int(self.opts.get("min_labels_to_expand", 100)) and bool(
            self.opts.get("expand_search_next", False)
        ):
            n_iter = int(self.opts.get("n_iter_extended", 100))

        n_jobs = int(self.opts.get("n_jobs", 1))
        tss_splits = self._time_splits(n_labeled)
        cv = (
            TimeSeriesSplit(n_splits=tss_splits)
            if tss_splits and X_train.shape[0] > tss_splits
            else None
        )

        best_pipe = None
        chosen_params = None
        best_score = None

        min_train_for_search = int(self.opts.get("min_train_for_search", 10))
        if X_train.shape[0] < min_train_for_search:
            logger.info(
                "Skipping hypersearch; too few train samples (%d)", X_train.shape[0]
            )
            best_pipe = Pipeline(
                [
                    (
                        "model",
                        HistGradientBoostingRegressor(random_state=self.random_state),
                    )
                ]
            )
            try:
                best_pipe.fit(X_train, y_train)
            except Exception:
                logger.exception("Fallback fit failed")
                return
        else:
            try:
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_dist_pipe,
                    n_iter=n_iter,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                    n_jobs=n_jobs,
                    random_state=self.random_state,
                )
                logger.info(
                    "Running hyperparameter search (n_iter=%d cv=%s)",
                    n_iter,
                    "TimeSeriesSplit" if cv else "None",
                )
                search.fit(X_train, y_train)
                best_pipe = search.best_estimator_
                chosen_params = getattr(search, "best_params_", None)
                best_score = getattr(search, "best_score_", None)
                logger.info(
                    "Hypersearch complete: best_score=%s best_params=%s",
                    str(best_score),
                    str(chosen_params),
                )
            except Exception:
                logger.exception(
                    "Hypersearch failed; falling back to default estimator"
                )
                best_pipe = Pipeline(
                    [
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                random_state=self.random_state
                            ),
                        )
                    ]
                )
                try:
                    best_pipe.fit(X_train, y_train)
                except Exception:
                    logger.exception("Fallback fit failed")
                    return

        # final refit on full data (optional) -- clean full arrays first
        try:
            if bool(self.opts.get("refit_on_full", True)):
                try:
                    X_full, y_full = _clean_train_arrays(X, y_delta)
                except Exception as e:
                    logger.exception("Full-data cleaning failed: %s", e)
                    return
                best_pipe.fit(X_full, y_full)

            def predict_fn(Xq):
                return best_pipe.predict(Xq)

        except Exception:
            logger.exception("Final refit failed")
            return

        # OOF: predict on labeled portion and reconstruct absolute preds for human-readable MAE
        mae_abs = None
        try:
            if n_labeled > 0:
                preds_delta = predict_fn(X[:n_labeled])
                current_arr = np.array(
                    [
                        _safe_float(r.data.get("current_setpoint", 0.0))
                        for r in used_rows[:n_labeled]
                    ],
                    dtype=float,
                )
                preds_abs = current_arr + np.array(preds_delta, dtype=float)
                y_true_abs = np.array(
                    [_safe_float(r.setpoint, 0.0) for r in used_rows[:n_labeled]],
                    dtype=float,
                )
                mae_abs = float(mean_absolute_error(y_true_abs, preds_abs))
                self._report_household_drift(
                    used_rows[:n_labeled], preds_abs, y_true_abs
                )
        except Exception:
            logger.exception("Failed OOF MAE computation")

        val_mae_abs = None
        try:
            if X_val is not None and len(X_val):
                val_preds_delta = predict_fn(X_val)
                ci = (
                    self.feature_order.index("current_setpoint")
                    if "current_setpoint" in self.feature_order
                    else None
                )
                if ci is not None:
                    curr_val = X_val[:, ci]
                    val_preds_abs = curr_val + np.array(val_preds_delta, dtype=float)
                    val_true_abs = curr_val + np.array(y_val, dtype=float)
                    val_mae_abs = float(
                        mean_absolute_error(val_true_abs, val_preds_abs)
                    )
                else:
                    val_mae_abs = float(mean_absolute_error(y_val, val_preds_delta))
        except Exception:
            logger.exception("Failed val MAE computation")

        runtime = time.time() - start
        logger.info(
            "Training finished runtime_seconds=%.2f n_labeled=%d mae_abs=%s val_mae_abs=%s chosen_params=%s",
            runtime,
            n_labeled,
            str(mae_abs),
            str(val_mae_abs),
            str(chosen_params),
        )

        # persist model + meta
        try:
            meta: Dict[str, Any] = {
                "feature_order": self.feature_order,
                "backend": "sklearn_histgb",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "mae": mae_abs,
                "val_mae": val_mae_abs,
                "n_samples": n_labeled,
                "use_unlabeled": False,
                "chosen_params": chosen_params,
                "search_best_score": best_score,
                "random_state": self.random_state,
                "runtime_seconds": runtime,
                "target": "delta",
            }
            if self.model_path and os.path.exists(self.model_path):
                try:
                    shutil.copy2(self.model_path, self.model_path + ".bak")
                except Exception:
                    logger.warning("Failed creating model backup")
            payload = {"model": best_pipe, "meta": meta}
            _atomic_dump(payload, self.model_path)
            logger.info("Saved model to %s (mae_abs=%s)", self.model_path, str(mae_abs))
        except Exception:
            logger.exception("Failed saving model")
            return

    def _time_splits(self, n_labeled: int) -> Optional[int]:
        min_train = int(self.opts.get("min_train_size", 30))
        if n_labeled < min_train:
            return None
        n_splits = min(3, max(2, n_labeled // 10))
        if n_splits >= n_labeled:
            return None
        return n_splits

    def _report_household_drift(
        self, used_rows: List[Any], preds_abs: np.ndarray, y_true_abs: np.ndarray
    ):
        try:
            hh_map = {}
            for i, r in enumerate(used_rows):
                hh = getattr(r, "household_id", None) or "unknown"
                hh_map.setdefault(hh, []).append((preds_abs[i], y_true_abs[i]))
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
